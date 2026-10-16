"""One-command submission builder (ensemble + optional threshold calibration + validation).

This is the single entry point for producing an official submission.  It runs the
whole chain deterministically and prints everything needed for the submission log:

  1. per-checkpoint valid metrics (AUROC / AUPRC / balACC@0.5) with flip TTA
  2. ensemble metrics + expected leaderboard value under both metric hypotheses
  3. test predictions (flip TTA) averaged with the same weights
  4. optional monotone calibration so the optimal operating point lands on 0.5
     (ranking metrics are provably unchanged -> safe either way)
  5. strict format validation of the written CSV

Usage:
  python make_submission.py --ckpt A/best.pt B/best.pt --out submission.csv
  python make_submission.py --ckpt A/best.pt B/best.pt --weights 1 0.8 --calibrate --out submission.csv
  python make_submission.py --ckpt A/best.pt --tta 1 --no-calibrate --out submission.csv   # quick check

Every run also writes a sidecar JSON report next to --out for the submission ledger.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torchvision

from train import FundusNet, load_rows, age_tensor, CACHE_ROOT, CSV_ROOT

AGE_MEDIAN = 58.0


class DS(Dataset):
    """Reads preprocessed cache; flip index for TTA."""

    def __init__(self, split, rows, img_size, flip=0, cache_root=None):
        self.dir = os.path.join(cache_root or CACHE_ROOT, split)
        self.rows = rows
        self.img_size = img_size
        self.flip = flip
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age, _ = self.rows[i]
        img = Image.open(os.path.join(self.dir, stem + ".jpg")).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.flip & 1:
            img = torchvision.transforms.functional.hflip(img)
        if self.flip & 2:
            img = torchvision.transforms.functional.vflip(img)
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        return t, age_tensor(age), i


@torch.no_grad()
def predict_split(model, split, rows, img_size, flips, workers, bs=64, cache_root=None):
    total = np.zeros(len(rows), dtype=np.float64)
    for fl in flips:
        ds = DS(split, rows, img_size, flip=fl, cache_root=cache_root)
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
        for x, age, idx in dl:
            x = x.cuda(non_blocking=True)
            age = age.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, age)
            total[idx.numpy()] += torch.sigmoid(logits.float()).cpu().numpy()
    return total / len(flips)


def threshold_metrics(y, p, thr=0.5):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-9, prec + sens)
    return {"balacc": 0.5 * (sens + spec), "sens": sens, "spec": spec, "prec": prec, "f1": f1,
            "acc": (tp + tn) / max(1, len(y)), "pos_rate": float(pred.mean())}


def ranking_metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def optimal_shift(y, p, target=0.5):
    """Return the logit shift that maps the max-balanced-accuracy point onto `target`."""
    order = np.argsort(-p)
    ys, ss = y[order], p[order]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx]
    fps = np.cumsum(1 - ys)[idx]
    P, N = ys.sum(), len(ys) - ys.sum()
    score = tps / P + (1 - fps / N)
    t_star = float(ss[idx][int(np.argmax(score))])
    eps = 1e-7
    t_star = min(max(t_star, eps), 1 - eps)
    return np.log(target / (1 - target)) - np.log(t_star / (1 - t_star)), t_star


def apply_shift(p, shift):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    z = np.log(p / (1 - p)) + shift
    return 1.0 / (1.0 + np.exp(-z))


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    use = ckpt.get("use")
    if use not in ("ema", "model"):
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    m = FundusNet(arch, pretrained=False).to(device)
    m.load_state_dict(ckpt[use], strict=True)
    m.eval()
    cache_root = (ckpt.get("args") or {}).get("cache_root") or CACHE_ROOT
    return m, ckpt.get("img_size", 384), use, ckpt, cache_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--img-sizes", type=int, nargs="+", default=None,
                    help="multi-scale TTA: average over several input scales (e.g. 448 512 576)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--calibrate", dest="calibrate", action="store_true", default=True)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    w = np.array(args.weights, dtype=float) if args.weights else np.ones(len(args.ckpt))
    w = w / w.sum()
    assert len(w) == len(args.ckpt), "--weights must match --ckpt count"

    valid_rows = load_rows("valid")
    y = np.array([r[2] for r in valid_rows])
    test_rows = load_rows("test")

    report = {"ckpt": args.ckpt, "weights": w.tolist(), "tta": args.tta,
              "calibrate": args.calibrate, "models": [], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    val_probs, test_probs = [], []
    for i, ck in enumerate(args.ckpt):
        model, img_size, use, ckpt, cache_root = load_model(ck)
        if ckpt.get("train_on_all"):
            print(f"[{i}] !! WARNING: {ck} is a --train-on-all checkpoint. Valid rows were part of "
                  f"its training set, so its AUROC/AUPRC/balACC are contaminated and any "
                  f"calibration fitted on them is unreliable.", flush=True)
        # 多尺度 TTA（#43）：每尺度内做 flip TTA，跨尺度平均（与 evaluate.py 语义一致）
        scales = args.img_sizes or [img_size]
        pv = np.zeros(len(valid_rows), dtype=np.float64)
        pt = np.zeros(len(test_rows), dtype=np.float64)
        for s in scales:
            pv += predict_split(model, "valid", valid_rows, s, flips, args.workers, cache_root=cache_root)
            pt += predict_split(model, "test", test_rows, s, flips, args.workers, cache_root=cache_root)
        pv /= len(scales)
        pt /= len(scales)
        rm, tm = ranking_metrics(y, pv), threshold_metrics(y, pv)
        print(f"[{i}] {os.path.basename(os.path.dirname(ck))} use={use} img={img_size} scales={scales} "
              f"cache={os.path.basename(str(cache_root))} "
              f"AUROC={rm['auroc']:.5f} AUPRC={rm['auprc']:.5f} balACC@0.5={tm['balacc']:.5f}", flush=True)
        report["models"].append({"ckpt": ck, "use": use, "img_size": img_size, "scales": scales,
                                 "cache_root": cache_root,
                                 "train_on_all": bool(ckpt.get("train_on_all", False)), **rm, **tm})
        val_probs.append(pv)
        test_probs.append(pt)
        del model
        torch.cuda.empty_cache()

    ens_val = np.zeros_like(val_probs[0])
    ens_test = np.zeros_like(test_probs[0])
    for wi, pv, pt in zip(w, val_probs, test_probs):
        ens_val += wi * pv
        ens_test += wi * pt

    rm, tm = ranking_metrics(y, ens_val), threshold_metrics(y, ens_val)
    blend = 0.5 * (rm["auroc"] + rm["auprc"])
    print(f"\nENSEMBLE weights={np.round(w,3).tolist()}")
    print(f"  AUROC={rm['auroc']:.5f}  AUPRC={rm['auprc']:.5f}  balACC@0.5={tm['balacc']:.5f}")
    print(f"  expected LB if metric=AUROC            : {rm['auroc']*100:.2f}")
    print(f"  expected LB if metric=AUPRC            : {rm['auprc']*100:.2f}")
    print(f"  expected LB if metric=0.5*(AUROC+AUPRC): {blend*100:.2f}")
    print(f"  expected LB if metric=balACC@0.5       : {tm['balacc']*100:.2f}")
    report["ensemble"] = {**rm, **tm, "blend": blend}

    final_val = ens_val
    if args.calibrate:
        shift, t_star = optimal_shift(y, ens_val)
        final_val = apply_shift(ens_val, shift)
        ens_test = apply_shift(ens_test, shift)
        tm2 = threshold_metrics(y, final_val)
        rm2 = ranking_metrics(y, final_val)
        print(f"\nCALIBRATED (shift={shift:+.3f} logit, raw optimum={t_star:.5f})")
        print(f"  ranking unchanged: AUROC={rm2['auroc']:.5f} AUPRC={rm2['auprc']:.5f}")
        print(f"  balACC@0.5: {tm['balacc']:.5f} -> {tm2['balacc']:.5f}   "
              f"(sens {tm['sens']:.3f}->{tm2['sens']:.3f}, spec {tm['spec']:.3f}->{tm2['spec']:.3f})")
        print(f"  expected LB if metric=balACC@0.5 (after calibration): {tm2['balacc']*100:.2f}")
        report["calibration"] = {"shift": float(shift), "raw_t_star": float(t_star), **tm2}

    # 复合口径预估 LB（#29）：线上 ≈ 0.5*balACC@0.5 + 0.25*AUROC + 0.25*AUPRC
    # 三点拟合锁定（见 SUBMISSIONS_LOG.md「口径判定对照」），典型误差 ±0.3
    bal_lb = tm2["balacc"] if args.calibrate else tm["balacc"]
    est_lb = 0.5 * bal_lb + 0.25 * rm["auroc"] + 0.25 * rm["auprc"]
    print(f"\n预估LB (复合口径 0.5*balACC + 0.25*AUROC + 0.25*AUPRC) = {est_lb*100:.2f}  "
          f"(三点拟合，典型误差 ±0.3)")
    report["est_lb"] = {"value": round(est_lb * 100, 2),
                        "formula": "0.5*balACC@0.5 + 0.25*AUROC + 0.25*AUPRC",
                        "balacc_used": bal_lb, "auroc": rm["auroc"], "auprc": rm["auprc"]}

    # ---- write + validate
    names = []
    with open(os.path.join(CSV_ROOT, "test.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            names.append(r["Image_name"])
    assert len(names) == len(ens_test), (len(names), len(ens_test))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, ens_test):
            wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])

    rows = list(csv.DictReader(open(args.out, encoding="utf-8")))
    probs = np.array([float(r["Probability"]) for r in rows])
    labs = np.array([int(r["Pred_Label"]) for r in rows])
    checks = {
        "rows == 5855": len(rows) == 5855,
        "names match test.csv": all(a["Image_name"] == b for a, b in zip(rows, names)),
        "prob in [0,1]": bool(((probs >= 0) & (probs <= 1)).all()),
        "label == (prob>=0.5)": bool(((probs >= 0.5) == (labs == 1)).all()),
    }
    print("\nVALIDATION:", {k: ("OK" if v else "FAIL") for k, v in checks.items()})
    assert all(checks.values()), "submission validation failed"
    print(f"pos_rate in submission = {labs.mean():.4f}")
    report["validation"] = checks
    report["submission_pos_rate"] = float(labs.mean())

    rp = os.path.splitext(args.out)[0] + f"_report{('_' + args.tag) if args.tag else ''}.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}\nwrote {rp}")


if __name__ == "__main__":
    main()