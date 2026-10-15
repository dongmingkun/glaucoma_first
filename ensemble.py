"""Ensemble predictions from multiple checkpoints.

Modes:
  --eval   : compute valid AUROC/AUPRC for each model and for the averaged ensemble
             (uses cached probs_*.npy written by evaluate.py when present)
  --submit : predict test set with a weighted ensemble and write submission.csv

Usage:
  python ensemble.py --ckpt runA/best.pt runB/best.pt --eval --tta 4
  python ensemble.py --ckpt runA/best.pt runB/best.pt --weights 1 0.8 --submit --out submission.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import predict
from train import FundusNet, load_rows
from sklearn.metrics import roc_auc_score, average_precision_score

CSV_ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"


def load_model(ckpt_path, use="auto"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    if use == "auto":
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    m = FundusNet(arch, pretrained=False).cuda()
    m.load_state_dict(ckpt[use], strict=True)
    m.eval()
    cache_root = (ckpt.get("args") or {}).get("cache_root") or None
    return m, ckpt.get("img_size", 384), use, cache_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--save-probs", default=None, help="npy path to save individual model probs (eval mode)")
    args = ap.parse_args()

    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    w = np.array(args.weights) if args.weights else np.ones(len(args.ckpt))
    w = w / w.sum()

    if args.eval:
        rows = load_rows(args.split)
        labels = np.array([r[2] for r in rows])
        all_probs = []
        for i, ck in enumerate(args.ckpt):
            m, img_size, use, cache_root = load_model(ck)
            p = predict(m, rows, args.split, img_size, flips, args.workers, cache_root=cache_root)
            a = roc_auc_score(labels, p)
            ap_ = average_precision_score(labels, p)
            print(f"[{i}] {os.path.basename(os.path.dirname(ck))} use={use} img={img_size} "
                  f"AUROC={a:.5f} AUPRC={ap_:.5f}", flush=True)
            all_probs.append(p)
            del m
            torch.cuda.empty_cache()
        ens = np.zeros_like(all_probs[0])
        for wi, p in zip(w, all_probs):
            ens += wi * p
        print(f"ENSEMBLE (weights={np.round(w,3).tolist()}): AUROC={roc_auc_score(labels, ens):.5f} "
              f"AUPRC={average_precision_score(labels, ens):.5f}", flush=True)
        # threshold metrics at 0.5 (for the threshold-metric hypothesis)
        pred = ens >= 0.5
        tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum()); tn = int((~pred & (labels == 0)).sum())
        sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
        print(f"ENSEMBLE @0.5: balACC={(sens+spec)/2:.5f} sens={sens:.4f} spec={spec:.4f}", flush=True)
        if args.save_probs:
            np.save(args.save_probs, np.stack(all_probs))
            print(f"saved individual probs -> {args.save_probs}")
            np.save(args.save_probs.replace(".npy", "_ens.npy"), ens)
            print(f"saved ensemble probs   -> {args.save_probs.replace('.npy', '_ens.npy')}")

    if args.submit:
        rows = load_rows("test")
        ens = np.zeros(len(rows))
        for i, ck in enumerate(args.ckpt):
            m, img_size, use, cache_root = load_model(ck)
            p = predict(m, rows, "test", img_size, flips, args.workers, cache_root=cache_root)
            ens += w[i] * p
            print(f"[{i}] {os.path.basename(os.path.dirname(ck))} test mean_prob={p.mean():.5f}", flush=True)
            del m
            torch.cuda.empty_cache()
        names = []
        with open(os.path.join(CSV_ROOT, "test.csv"), encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                names.append(r["Image_name"])
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["Image_name", "Probability", "Pred_Label"])
            for n, p in zip(names, ens):
                wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
        print(f"wrote {args.out} ({len(names)} rows), pos_rate={(ens>=0.5).mean():.4f}")


if __name__ == "__main__":
    main()