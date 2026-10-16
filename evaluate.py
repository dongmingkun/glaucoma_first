"""Evaluate a checkpoint on the valid split: AUROC/AUPRC with TTA, plus age ablation.

Usage:
  python evaluate.py --ckpt <best.pt> [--tta 4] [--no-age] [--img-size 384]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import FundusNet, load_rows, age_tensor, CACHE_ROOT
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from sklearn.metrics import roc_auc_score, average_precision_score

AGE_MEDIAN = 58.0


class ValDS(Dataset):
    def __init__(self, split, rows, img_size, flip=0, fixed_age=None, cache_root=None):
        self.dir = os.path.join(cache_root or CACHE_ROOT, split)
        self.rows = rows
        self.img_size = img_size
        self.flip = flip
        self.fixed_age = fixed_age
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age, label = self.rows[i]
        if self.fixed_age is not None:
            age = self.fixed_age
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
def predict(model, rows, split, img_size, flips, workers, fixed_age=None, cache_root=None):
    total = np.zeros(len(rows))
    for fl in flips:
        ds = ValDS(split, rows, img_size, flip=fl, fixed_age=fixed_age, cache_root=cache_root)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers, pin_memory=True)
        for x, age, idx in dl:
            x, age = x.cuda(non_blocking=True), age.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, age)
            total[idx.numpy()] += torch.sigmoid(logits.float()).cpu().numpy()
    return total / len(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--img-sizes", type=int, nargs="+", default=None,
                    help="multi-scale TTA: average over several input scales, e.g. 384 448")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use", default="auto", choices=["auto", "ema", "model"])
    ap.add_argument("--no-age", action="store_true", help="ablation: replace age with median")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    img_size = args.img_size or ckpt.get("img_size", 384)
    use = args.use
    if use == "auto":
        forced = ckpt.get("use")
        if forced in ("ema", "model"):
            use = forced          # train-on-all checkpoints carry an explicit choice
        else:
            use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    model = FundusNet(arch, pretrained=False).cuda()
    model.load_state_dict(ckpt[use], strict=True)
    model.eval()

    rows = load_rows(args.split)
    labels = np.array([r[2] for r in rows])
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    fixed = AGE_MEDIAN if args.no_age else None
    cache_root = (ckpt.get("args") or {}).get("cache_root") or CACHE_ROOT
    scales = args.img_sizes or [img_size]
    probs = np.zeros(len(rows))
    for s in scales:
        p = predict(model, rows, args.split, s, flips, args.workers, fixed, cache_root=cache_root)
        probs += p
        print(f"  scale={s}: mean_prob={p.mean():.5f}", flush=True)
    probs /= len(scales)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    tag = "NO-AGE" if args.no_age else "full"
    scale_tag = "ms" if len(scales) > 1 else f"img{scales[0]}"
    print(f"[{args.split}] ckpt={os.path.basename(os.path.dirname(args.ckpt))} use={use} scales={scales} "
          f"tta={args.tta} ({tag}): AUROC={auroc:.5f} AUPRC={auprc:.5f}")
    np.save(os.path.join(os.path.dirname(args.ckpt), f"probs_{args.split}_{tag}_{scale_tag}_tta{args.tta}.npy"), probs)


if __name__ == "__main__":
    main()