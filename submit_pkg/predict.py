"""Self-contained inference for the glaucoma challenge (competition server deployment).

Runs on: Python 3.12, PyTorch 2.10.0 + torchvision 0.25.0 (CUDA 12.8), NVIDIA T4.
Dependencies: torch, torchvision, pillow, numpy  (all present in the official image).

Input : a directory of fundus images (jpg/png) and a CSV with columns Image_name[,Age]
Output: submission.csv  (Image_name, Probability, Pred_Label)

Usage:
  python predict.py --images /path/to/test_images --csv /path/to/test.csv --weights model_weights.pt --out submission.csv

Design notes:
- Backbone: torchvision efficientnet_b3 features + age MLP fusion head (same as training).
- Preprocessing: JPEG draft-decode -> fundus crop (gray>15 bbox, 0.2/99.8 percentile trim) ->
  centre square -> 512x512 -> model input resize -> ImageNet normalisation.
- TTA: 4 flip variants averaged (probability space).
- Single-image latency on T4: decode(few ms, draft) + crop(~5ms) + forward(~7ms fp16) ~ well under 100ms.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

AGE_MEDIAN = 58.0
CACHE_SIZE = 512  # preprocessing output size (matches training cache)


# ---------------------------------------------------------------- model
class FundusNet(nn.Module):
    def __init__(self, age_dim=16, dropout=0.3):
        super().__init__()
        m = torchvision.models.efficientnet_b3(weights=None)
        self.features = m.features
        self.avgpool = m.avgpool
        self.norm = nn.Identity()
        self.feat_dim = 1536
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_dim), nn.SiLU(), nn.Linear(age_dim, age_dim), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1536 + age_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def forward(self, x, age):
        f = self.features(x)
        f = self.avgpool(f)
        f = self.norm(f)
        f = torch.flatten(f, 1)
        a = self.age_mlp(age)
        return self.head(torch.cat([f, a], dim=1)).squeeze(1)


# ---------------------------------------------------------------- preprocessing
def load_fundus_crop(path, size=CACHE_SIZE):
    im = Image.open(path)
    im.draft("RGB", (768, 768))
    im = im.convert("RGB")
    w, h = im.size
    arr = np.asarray(im)
    gray = arr.mean(axis=2)
    mask = gray > 15
    frac = float(mask.mean())
    if frac < 0.02 or frac > 0.999:
        bbox = (0, 0, w, h)
    else:
        ys, xs = np.nonzero(mask)
        y0, y1 = np.percentile(ys, [0.2, 99.8])
        x0, x1 = np.percentile(xs, [0.2, 99.8])
        bbox = (float(x0), float(y0), float(x1), float(y1))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * 1.02
    nx0, ny0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    nx1, ny1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -nx0), max(0, -ny0)
    pad_r, pad_b = max(0, nx1 - w), max(0, ny1 - h)
    crop = arr[max(0, ny0):min(h, ny1), max(0, nx0):min(w, nx1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


class InferDS(torch.utils.data.Dataset):
    def __init__(self, paths, ages, img_size, flip=0):
        self.paths = paths
        self.ages = ages
        self.img_size = img_size
        self.flip = flip
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_fundus_crop(self.paths[i])
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.flip & 1:
            img = torchvision.transforms.functional.hflip(img)
        if self.flip & 2:
            img = torchvision.transforms.functional.vflip(img)
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        age = self.ages[i]
        return t, torch.tensor([(age - AGE_MEDIAN) / 15.0], dtype=torch.float32), i


def clean_age(v):
    try:
        a = float(v)
    except (ValueError, TypeError):
        return AGE_MEDIAN
    if a < 1 or a > 110:
        return AGE_MEDIAN
    return min(max(a, 10.0), 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory containing the images")
    ap.add_argument("--csv", required=True, help="csv with Image_name[,Age]")
    ap.add_argument("--weights", required=True, help="model weights (.pt from training best.pt)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--img-size", type=int, default=0, help="0 = from checkpoint")
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use", default="auto", choices=["auto", "ema", "model"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    img_size = args.img_size or ckpt.get("img_size", 384)
    use = args.use
    if use == "auto":
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    print(f"weights={args.weights} img_size={img_size} use={use} device={device}", flush=True)

    model = FundusNet().to(device)
    sd = ckpt.get(use) or ckpt.get("model") or ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    names, ages, paths = [], [], []
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            names.append(r["Image_name"])
            ages.append(clean_age(r.get("Age", AGE_MEDIAN)))
            stem = r["Image_name"].rsplit(".", 1)[0]
            p = None
            for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
                cand = os.path.join(args.images, stem + ext)
                if os.path.exists(cand):
                    p = cand
                    break
            if p is None:
                raise FileNotFoundError(f"image not found for {r['Image_name']} under {args.images}")
            paths.append(p)
    print(f"{len(names)} images to predict", flush=True)

    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    total = np.zeros(len(names), dtype=np.float64)
    t_start = time.time()
    for fl in flips:
        ds = InferDS(paths, ages, img_size, flip=fl)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            for x, age, idx in dl:
                x = x.to(device, non_blocking=True)
                age = age.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(x, age)
                total[idx.numpy()] += torch.sigmoid(logits.float()).cpu().numpy()
        print(f"  flip={fl} done ({time.time()-t_start:.0f}s elapsed)", flush=True)
    probs = total / len(flips)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, probs):
            wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
    dt = time.time() - t_start
    print(f"wrote {args.out}: {len(names)} rows in {dt:.0f}s ({dt/len(names)*1000:.1f} ms/img incl. {len(flips)}x TTA)", flush=True)


if __name__ == "__main__":
    main()