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

Latency optimisation (issue #20):
- Preprocessing de-dup: each image is decoded+cropped ONCE per cache_size group, then
  shared by all models in that group (instead of once per model x flip = up to 12x).
- Tensor TTA: flips are done with torch.flip on the tensor, never round-tripping to PIL.
- Draft alignment: JPEG draft = max(768, cache_size*1.5), matching the training cache
  (512 cache uses draft 768, 768 cache uses draft 1152), shrinking prob drift to ~1e-4.

Design notes:
- Backbone: torchvision efficientnet_b0/b3/b4, convnext_tiny/small/base, resnet50
  + age MLP fusion head (same as training; arch read from each checkpoint).
- Preprocessing: JPEG draft-decode -> fundus crop (gray>15 bbox, 0.2/99.8 percentile
  trim) -> centre square -> cache_size -> model input resize -> ImageNet normalisation.
  cache_size is inferred per-checkpoint from the training --cache-root (512 vs 768).
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
def load_fundus_crop(path, size=DEFAULT_CACHE_SIZE, draft=None):
    """Decode (draft-aligned) + fundus crop + resize to `size`. Returns PIL RGB."""
    if draft is None:
        draft = max(768, int(size * 1.5))
    im = Image.open(path)
    im.draft("RGB", (draft, draft))
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


def clean_age(v):
    try:
        a = float(v)
    except (ValueError, TypeError):
        return AGE_MEDIAN
    if a < 1 or a > 110:
        return AGE_MEDIAN
    return min(max(a, 10.0), 100.0)


def infer_cache_size(ckpt, ckpt_path=None):
    """Infer the preprocessing cache size (512 vs 768) for this checkpoint.

    多级回退（issue #28）：
      1) ckpt["cache_size"]（瘦身工具显式写入）
      2) ckpt["args"]["cache_root"] 含 "768"
      3) 权重文件名含 "cache768"（覆盖已发布的 Release 权重）
      4) 默认 512
    """
    if isinstance(ckpt, dict):
        cs = ckpt.get("cache_size")
        if cs in (512, 768):
            return int(cs)
        args = ckpt.get("args")
        if isinstance(args, dict) and "768" in str(args.get("cache_root", "")):
            return 768
    if ckpt_path and "cache768" in os.path.basename(str(ckpt_path)):
        return 768
    return DEFAULT_CACHE_SIZE


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b3")
    img_size = ckpt.get("img_size", 384)
    cache_size = infer_cache_size(ckpt, ckpt_path)
    # train-on-all checkpoints carry an explicit `use` (PR #16); respect it, else fall
    # back to the auroc comparison — same rule as make_submission.py / infer.py / ensemble.py.
    use = ckpt.get("use")
    if use not in ("ema", "model"):
        use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
    sd = ckpt.get(use) or ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model = FundusNet(arch=arch).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, {"arch": arch, "img_size": img_size, "cache_size": cache_size, "use": use}


def run_group(paths, ages, members, device, tta, batch_size, prep_workers=8):
    """Run all models sharing one cache_size, preprocessing each image ONCE.

    members: list of (model, meta, weight). Returns a list of per-model prob arrays.
    """
    from concurrent.futures import ThreadPoolExecutor

    cache_size = members[0][1]["cache_size"]
    draft = max(768, int(cache_size * 1.5))
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[tta]
    pool = ThreadPoolExecutor(max_workers=max(1, prep_workers))

    def _prep(path):
        # PIL decode + numpy crop release the GIL, so threads actually scale.
        return torchvision.transforms.functional.to_tensor(
            load_fundus_crop(path, size=cache_size, draft=draft))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    age_all = torch.tensor([[(a - AGE_MEDIAN) / 15.0] for a in ages],
                           dtype=torch.float32, device=device)

    n = len(paths)
    acc = [np.zeros(n, dtype=np.float64) for _ in members]

    # Bounded lookahead: the pool decodes/crops ahead while the GPU runs, so the
    # end-to-end cost is ~max(preprocess, forward) instead of their sum.
    from collections import deque

    src = iter(range(n))
    lookahead = max(batch_size, prep_workers * 2) * 2
    futs = deque()
    for _ in range(min(lookahead, n)):
        i = next(src)
        futs.append(pool.submit(_prep, paths[i]))

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        # ---- preprocessing de-dup: decode+crop ONCE for each image ----
        base = torch.stack([futs.popleft().result() for _ in idx]).to(device)
        for _ in idx:
            j = next(src, None)
            if j is not None:
                futs.append(pool.submit(_prep, paths[j]))
        age_b = age_all[idx]

        for mi, (model, meta, _w) in enumerate(members):
            img_size = meta["img_size"]
            x = F.interpolate(base, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
            x = (x - mean) / std
            with torch.no_grad():
                for fl in flips:
                    xf = x
                    if fl & 1:
                        xf = torch.flip(xf, dims=[3])  # horizontal
                    if fl & 2:
                        xf = torch.flip(xf, dims=[2])  # vertical
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                        logits = model(xf, age_b)
                    acc[mi][idx] += torch.sigmoid(logits.float()).cpu().numpy()
        del base, age_b
    return [a / len(flips) for a in acc]


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
    ap.add_argument("--prep-workers", type=int, default=8,
                    help="threads for decode+crop (PIL/numpy release the GIL)")
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

    # ---- group models by cache_size so preprocessing is shared ----
    groups = {}
    for (m, meta), w in zip(models, blend):
        groups.setdefault(meta["cache_size"], []).append((m, meta, w))
    print(f"cache_size groups: {[(k, len(v)) for k, v in groups.items()]}", flush=True)

    t_start = time.time()
    ens = np.zeros(len(names), dtype=np.float64)
    for cache_size, members in groups.items():
        probs = run_group(paths, ages, members, device, args.tta, args.batch_size,
                          args.prep_workers)
        for (m, meta, w), p in zip(members, probs):
            ens += w * p
            print(f"  model {meta['arch']} (cache{cache_size}) done "
                  f"({time.time()-t_start:.0f}s elapsed)", flush=True)

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
