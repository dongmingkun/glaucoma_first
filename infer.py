"""Inference: predict probabilities for a split with flip TTA, write submission CSV.

Usage:
  python infer.py --ckpt D:\\glaucoma_work\\runs\\xxx\\best.pt
  python infer.py --ckpt ... --split test --tta 4 --out submission.csv
  python infer.py --ckpt ... --split valid --out valid_pred.csv   (to check AUROC)
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import FundusNet, load_rows, age_tensor, CACHE_ROOT, CSV_ROOT


class InferDS(Dataset):
    def __init__(self, split, rows, img_size, flip=0, cache_root=None):
        self.dir = os.path.join(cache_root or CACHE_ROOT, split)
        self.rows = rows
        self.img_size = img_size
        self.flip = flip  # 0 none, 1 hflip, 2 vflip, 3 both
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age, label = self.rows[i]
        img = Image.open(os.path.join(self.dir, stem + ".jpg")).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.flip & 1:
            img = torchvision.transforms.functional.hflip(img)
        if self.flip & 2:
            img = torchvision.transforms.functional.vflip(img)
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        return t, age_tensor(age), torch.tensor(label, dtype=torch.float32), i


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = np.zeros(len(loader.dataset), dtype=np.float64)
    for x, age, y, idx in loader:
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, age)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        probs[idx.numpy()] = p
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--img-size", type=int, default=0, help="0 = use ckpt value")
    ap.add_argument("--tta", type=int, default=4, help="number of flip variants 1/2/4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use", default="auto", choices=["auto", "ema", "model"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    img_size = args.img_size or ckpt["img_size"]
    use = args.use
    if use == "auto":
        forced = ckpt.get("use")
        if forced in ("ema", "model"):
            use = forced          # train-on-all checkpoints carry an explicit choice
        else:
            a_ema = ckpt.get("auroc_ema", -1)
            a_mod = ckpt.get("auroc", -1)
            use = "ema" if a_ema >= a_mod else "model"
    cache_root = (ckpt.get("args") or {}).get("cache_root") or CACHE_ROOT
    print(f"ckpt={args.ckpt} arch={arch} img={img_size} use={use} cache={cache_root} "
          f"(auroc_ema={ckpt.get('auroc_ema')} auroc={ckpt.get('auroc')})")
    if ckpt.get("train_on_all") and args.split in ("train", "valid"):
        print("!! WARNING: checkpoint was trained with --train-on-all. Rows from this split were "
              "IN the training set, so the AUROC/AUPRC below are contaminated -- do not report them.")

    model = FundusNet(arch, pretrained=False).to(device)
    sd = ckpt["ema"] if use == "ema" else ckpt["model"]
    model.load_state_dict({k: v for k, v in sd.items()}, strict=True)

    rows = load_rows(args.split)
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    total = np.zeros(len(rows), dtype=np.float64)
    for fl in flips:
        ds = InferDS(args.split, rows, img_size, flip=fl, cache_root=cache_root)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4)
        p = predict(model, dl, device)
        total += p
        print(f"  flip={fl} mean_prob={p.mean():.5f}", flush=True)
    probs = total / len(flips)

    if args.split in ("train", "valid"):
        from sklearn.metrics import roc_auc_score, average_precision_score
        labels = np.array([r[2] for r in rows])
        print(f"[{args.split}] AUROC={roc_auc_score(labels, probs):.5f} AUPRC={average_precision_score(labels, probs):.5f}")

    if args.out:
        names = []
        with open(os.path.join(CSV_ROOT, args.split + ".csv"), "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                names.append(r["Image_name"])
        assert len(names) == len(probs)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["Image_name", "Probability", "Pred_Label"])
            for n, p in zip(names, probs):
                wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
        print(f"wrote {args.out} ({len(names)} rows)")


if __name__ == "__main__":
    main()