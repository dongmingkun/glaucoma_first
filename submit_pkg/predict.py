"""Self-contained inference for the glaucoma challenge (competition server deployment).

Runs on: Python 3.12, PyTorch 2.10.0 + torchvision 0.25.0 (CUDA 12.8), NVIDIA T4.
Dependencies: torch, torchvision, pillow, numpy  (all present in the official image).

Input : a directory of fundus images (jpg/png) and a CSV with columns Image_name[,Age]
Output: submission.csv  (Image_name, Probability, Pred_Label)

Single model:
  python predict.py --images /path/to/test_images --csv /path/to/test.csv \
      --weights model_weights.pt --out submission.csv

Multi-model ensemble (weighted probability average):
  python predict.py --images ... --csv ... \
      --weights a.pt b.pt c.pt --blend 1 0.8 0.5 --out submission.csv

Baked calibration (monotone logit shift so the 0.5 operating point matches online):
  python predict.py ... --calib-shift 2.214

Design notes:
- Backbone: torchvision efficientnet_b0/b3/b4, convnext_tiny/small/base, resnet50
  + age MLP fusion head (same as training; arch read from each checkpoint).
- Preprocessing: JPEG draft-decode -> fundus crop (gray>15 bbox, 0.2/99.8 percentile
  trim) -> centre square -> cache_size -> model input resize -> ImageNet normalisation.
  cache_size is inferred per-checkpoint from the training --cache-root (512 vs 768).
- TTA: 4 flip variants averaged (probability space).
- Latency on T4: decode(few ms, draft) + crop(~5ms) + forward(~7ms fp16 per model).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

AGE_MEDIAN = 58.0
DEFAULT_CACHE_SIZE = 512

# arch -> (constructor, feature dim)  (must match train.py's FundusNet)
ARCH_INFO = {
    "efficientnet_b0": (torchvision.models.efficientnet_b0, 1280),
    "efficientnet_b3": (torchvision.models.efficientnet_b3, 1536),
    "efficientnet_b4": (torchvision.models.efficientnet_b4, 1792),
    "convnext_tiny": (torchvision.models.convnext_tiny, 768),
    "convnext_small": (torchvision.models.convnext_small, 768),
    "convnext_base": (torchvision.models.convnext_base, 1024),
    "resnet50": (torchvision.models.resnet50, 2048),
}


class FundusNet(nn.Module):
    def __init__(self, arch="efficientnet_b3", age_dim=16, dropout=0.3):
        super().__init__()
        if arch not in ARCH_INFO:
            raise ValueError(f"unknown arch: {arch}")
        ctor, feat_dim = ARCH_INFO[arch]
        self.arch = arch
        self.feat_dim = feat_dim
        m = ctor(weights=None)
        if arch.startswith("efficientnet"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        elif arch.startswith("convnext"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = m.classifier[0]  # LayerNorm2d
        elif arch.startswith("resnet"):
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        else:
            raise ValueError(arch)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_dim), nn.SiLU(), nn.Linear(age_dim, age_dim), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + age_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
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
def load_fundus_crop(path, size=DEFAULT_CACHE_SIZE):
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
    def __init__(self, paths, ages, img_size, cache_size, flip=0):
        self.paths = paths
        self.ages = ages
        self.img_size = img_size
        self.cache_size = cache_size
        self.flip = flip
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_fundus_crop(self.paths[i], size=self.cache_size)
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


def infer_cache_size(ckpt):
    """Infer the preprocessing cache size (512 vs 768) from the checkpoint's
    training --cache-root, so each model uses its own training-time crop."""
    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    cr = ""
    if isinstance(args, dict):
        cr = str(args.get("cache_root", ""))
    if "768" in cr:
        return 768
    return DEFAULT_CACHE_SIZE


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    img_size = ckpt.get("img_size", 384)
    cache_size = infer_cache_size(ckpt)
    use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    sd = ckpt.get(use) or ckpt.get("model") or ckpt
    model = FundusNet(arch=arch).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, {"arch": arch, "img_size": img_size, "cache_size": cache_size, "use": use}


def predict_one(model, paths, ages, meta, device, tta, batch_size, workers):
    """Run one model with TTA and return per-image probability array."""
    img_size = meta["img_size"]
    cache_size = meta["cache_size"]
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[tta]
    total = np.zeros(len(paths), dtype=np.float64)
    for fl in flips:
        ds = InferDS(paths, ages, img_size, cache_size, flip=fl)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True)
        with torch.no_grad():
            for x, age, idx in dl:
                x = x.to(device, non_blocking=True)
                age = age.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = model(x, age)
                total[idx.numpy()] += torch.sigmoid(logits.float()).cpu().numpy()
    return total / len(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory containing the images")
    ap.add_argument("--csv", required=True, help="csv with Image_name[,Age]")
    ap.add_argument("--weights", required=True, nargs="+", help="one or more .pt checkpoints")
    ap.add_argument("--blend", type=float, nargs="+", default=None,
                    help="per-model ensemble weights (default: equal)")
    ap.add_argument("--calib-shift", type=float, default=0.0,
                    help="baked logit shift for threshold calibration (from calibrate_submission.py)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tta", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    n_models = len(args.weights)
    blend = args.blend
    if blend is None:
        blend = [1.0 / n_models] * n_models
    else:
        if len(blend) != n_models:
            raise SystemExit(f"--blend 数量 ({len(blend)}) 与 --weights 数量 ({n_models}) 不一致")
        s = sum(blend)
        if s <= 0:
            raise SystemExit("--blend 权重之和必须 > 0")
        blend = [w / s for w in blend]

    models = []
    for p in args.weights:
        m, meta = load_model(p, device)
        print(f"model: arch={meta['arch']} img_size={meta['img_size']} "
              f"cache_size={meta['cache_size']} use={meta['use']} <- {p}", flush=True)
        models.append((m, meta))
    print(f"ensemble: {n_models} models, blend={[round(b, 4) for b in blend]}, "
          f"tta={args.tta}, calib_shift={args.calib_shift}", flush=True)

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

    t_start = time.time()
    ens = np.zeros(len(names), dtype=np.float64)
    for (m, meta), w in zip(models, blend):
        probs = predict_one(m, paths, ages, meta, device, args.tta, args.batch_size, args.workers)
        ens += w * probs
        print(f"  model {meta['arch']} done ({time.time()-t_start:.0f}s elapsed)", flush=True)

    if args.calib_shift != 0.0:
        eps = 1e-7
        p = np.clip(ens, eps, 1 - eps)
        z = np.log(p / (1 - p)) + args.calib_shift
        ens = 1.0 / (1.0 + np.exp(-z))
        print(f"applied calib_shift={args.calib_shift:+.3f} (logit)", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, ens):
            wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
    dt = time.time() - t_start
    pos_rate = float((ens >= 0.5).mean())
    print(f"wrote {args.out}: {len(names)} rows in {dt:.0f}s "
          f"(pos_rate={pos_rate*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
