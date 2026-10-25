"""Sanity-check a --train-on-all checkpoint before it is trusted for a submission.

Why this exists
---------------
A train-on-all model has seen every valid row, so its valid AUROC/AUPRC are
contaminated and must never be reported (or used to fit calibration). The only
clean signal left is agreement with a *train-only* reference model on the test
set: architecture, warm start and objective are unchanged, so a healthy
train-on-all run should rank test images almost identically, with a mildly
shifted probability scale.

Usage
-----
  # with a reference model (recommended -- this is the real check)
  python check_train_on_all.py --ckpt runs/all/best.pt --ref runs/base/best.pt --tta 4

  # no reference: still prints the distribution self-check
  python check_train_on_all.py --ckpt runs/all/best.pt --tta 4

Pass criteria (both must hold when --ref is given):
  * Spearman rank correlation on test  >= 0.90
  * |mean_prob - ref_mean_prob| / ref_mean_prob <= 0.25
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import InferDS, predict          # noqa: E402
from train import FundusNet                 # noqa: E402
from paths import load_rows                 # noqa: E402

SPEARMAN_MIN = 0.90
MEAN_REL_TOL = 0.25


def predict_test(ckpt_path, tta, workers, batch_size, cache_root=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    img_size = ckpt.get("img_size", 384)
    use = ckpt.get("use")
    if use not in ("ema", "model"):
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    model = FundusNet(arch, pretrained=False).cuda()
    model.load_state_dict(ckpt[use], strict=True)

    rows = load_rows("test")
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[tta]
    total = np.zeros(len(rows), dtype=np.float64)
    for fl in flips:
        ds = InferDS("test", rows, img_size, flip=fl, cache_root=cache_root)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0, prefetch_factor=4)
        total += predict(model, dl, "cuda")
    del model
    torch.cuda.empty_cache()
    return total / len(flips), ckpt, use, img_size


def describe(tag, p):
    q = np.quantile(p, [0.5, 0.9, 0.99])
    print(f"[{tag}] n={len(p)} mean={p.mean():.5f} pos_rate@0.5={(p >= 0.5).mean():.4f} "
          f"p50={q[0]:.5f} p90={q[1]:.5f} p99={q[2]:.5f} max={p.max():.5f} "
          f"nan={int(np.isnan(p).sum())}", flush=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train-on-all best.pt to check")
    ap.add_argument("--ref", default=None, help="train-only best.pt used as reference")
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=200, help="head-overlap size (AUPRC-relevant)")
    ap.add_argument("--cache-root", default=None,
                    help="test 图像缓存根目录（默认 CACHE_ROOT 常量，云端需显式传）")
    args = ap.parse_args()

    p, ckpt, use, img_size = predict_test(args.ckpt, args.tta, args.workers, args.batch_size,
                                          args.cache_root)
    print(f"ckpt={args.ckpt} arch={ckpt.get('arch')} img={img_size} use={use} "
          f"train_on_all={bool(ckpt.get('train_on_all'))} epoch={ckpt.get('epoch')}")
    if not ckpt.get("train_on_all"):
        print("NOTE: this checkpoint is NOT flagged train_on_all. If it was, the train.py run "
              "predates the flag -- re-run train.py --train-on-all to get a properly tagged one.")
    print("Reminder: valid AUROC/AUPRC for this model are meaningless (valid was trained on). "
          "Never report them or calibrate against them.")

    describe("cand", p)
    if np.isnan(p).sum():
        print("FAIL: NaNs in probabilities -- checkpoint is broken.")
        return 1

    if not args.ref:
        print("No --ref given: distribution self-check only. "
              "Pass a train-only best.pt for the real agreement test.")
        return 0

    r, rck, ruse, rimg = predict_test(args.ref, args.tta, args.workers, args.batch_size,
                                      args.cache_root)
    print(f"ref ={args.ref} arch={rck.get('arch')} img={rimg} use={ruse} "
          f"train_on_all={bool(rck.get('train_on_all'))}")
    if rck.get("train_on_all"):
        print("WARNING: --ref is itself a train-on-all checkpoint; agreement proves little. "
              "Use a train-only run as reference.")
    describe("ref ", r)

    sp = spearman(p, r)
    rel = abs(p.mean() - r.mean()) / max(1e-9, r.mean())
    topk = min(args.topk, len(p))
    a = set(np.argsort(-p)[:topk].tolist())
    b = set(np.argsort(-r)[:topk].tolist())
    overlap = len(a & b) / float(topk)

    print(f"spearman={sp:.4f} (min {SPEARMAN_MIN})   mean_rel_diff={rel:.3f} (max {MEAN_REL_TOL})   "
          f"top{topk}_overlap={overlap:.3f}")

    ok = (sp >= SPEARMAN_MIN) and (rel <= MEAN_REL_TOL)
    verdict = "PASS" if ok else "REVIEW"
    print(f"RESULT: {verdict}")
    if not ok:
        print("  -> do not submit yet. Typical causes: too many epochs (overfit after valid was "
              "folded in), or a warm start that diverged. Compare --epochs and re-run.")
    print(f"SUMMARY: cand={os.path.basename(os.path.dirname(args.ckpt))} "
          f"ref={os.path.basename(os.path.dirname(args.ref))} spearman={sp:.4f} "
          f"mean_rel_diff={rel:.3f} top{topk}_overlap={overlap:.3f} -> {verdict}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
