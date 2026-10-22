"""Per-batch latency profiler for the glaucoma competition submission package.

Reuses `submit_pkg/predict.py`'s preprocessing + model loading so the numbers
reflect *exactly* what real inference does (the forward / preprocess logic is
imported, not re-implemented). It only adds a per-batch wall-clock around the
(bounded-lookahead preprocessing + GPU forward) and reports mean / p50 / p95 /
p99 per-image latency, skipping the first `--warmup` batches.

This is CPU-only to *run the profiler* on any machine; the actual inference
obviously needs a GPU. The captain runs it on the RTX 5060 with the package
weights. No GPU training, no cloud.

Copy-paste on the target machine (RTX 5060), from inside the package dir
(where `predict.py` + the `.pt` files live):

  # 1-model package
  python measure_latency.py --images <test_images> --csv <test.csv> \
      --weights cnxb_384_all_pseudo.pt --tta 1 --batch-size 32 --prep-workers 8

  # 2-model package
  python measure_latency.py --images <test_images> --csv <test.csv> \
      --weights cb384.pt cnxb_384_all_pseudo.pt --blend 0.65 0.35 \
      --tta 1 --batch-size 32 --prep-workers 8

Safety-margin targets (issue #110): single-model p95 <= 40 ms (2.5x margin to
the 100 ms T4 red line); 2-model p95 <= 50 ms (2x margin). T4 itself is NOT
measured here — the 5060 number is a *lower bound*; the 100 ms red line can only
be certified on a real T4 (issue #81).
"""
import argparse
import csv
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# Single source of truth: import the real inference pieces from predict.py.
# Works both when this file sits INSIDE a submission package (predict.py alongside)
# and when it lives at repo tools/ (predict.py is in ../submit_pkg).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "submit_pkg"))
from predict import (  # noqa: E402  (identical to the deployed predict.py)
    AGE_MEDIAN,
    clean_age,
    load_fundus_crop,
    load_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory of fundus images")
    ap.add_argument("--csv", required=True, help="csv with Image_name[,Age]")
    ap.add_argument("--weights", required=True, nargs="+", help="one or more .pt")
    ap.add_argument("--blend", type=float, nargs="+", default=None,
                    help="per-model weights (default: equal)")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prep-workers", type=int, default=8,
                    help="decode+crop threads (PIL/numpy release the GIL)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="skip the first N batches from timing")
    ap.add_argument("--device", default=None, help="force cpu/cuda")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    n_models = len(args.weights)
    blend = args.blend or [1.0 / n_models] * n_models
    if len(blend) != n_models:
        raise SystemExit(f"--blend ({len(blend)}) != --weights ({n_models})")
    s = sum(blend)
    if s <= 0:
        raise SystemExit("--blend sum must be > 0")
    blend = [w / s for w in blend]

    models = []
    for p in args.weights:
        m, meta = load_model(p, device)
        print(f"model: arch={meta['arch']} img_size={meta['img_size']} "
              f"cache_size={meta['cache_size']} use={meta['use']} <- {p}", flush=True)
        models.append((m, meta))
    print(f"profiling: {n_models} models, blend={[round(b, 4) for b in blend]}, "
          f"tta={args.tta}, device={device}", flush=True)

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
                raise FileNotFoundError(f"image not found: {r['Image_name']}")
            paths.append(p)
    print(f"{len(names)} images to profile", flush=True)

    # group by cache_size so preprocessing is shared (same as predict.py)
    groups = {}
    for (m, meta), w in zip(models, blend):
        groups.setdefault(meta["cache_size"], []).append((m, meta, w))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    age_all = torch.tensor([[(a - AGE_MEDIAN) / 15.0] for a in ages],
                           dtype=torch.float32, device=device)

    per_image_ms = []
    n = len(paths)
    batch_idx = 0

    for cache_size, members in groups.items():
        draft = max(768, int(cache_size * 1.5))
        flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
        pool = ThreadPoolExecutor(max_workers=max(1, args.prep_workers))

        def _prep(path):
            return torchvision.transforms.functional.to_tensor(
                load_fundus_crop(path, size=cache_size, draft=draft))

        src = iter(range(n))
        lookahead = max(args.batch_size, args.prep_workers * 2) * 2
        futs = deque()
        for _ in range(min(lookahead, n)):
            i = next(src)
            futs.append(pool.submit(_prep, paths[i]))

        for start in range(0, n, args.batch_size):
            idx = list(range(start, min(start + args.batch_size, n)))
            t0 = time.perf_counter()
            # ---- preprocessing de-dup: decode+crop ONCE for each image ----
            base = torch.stack([futs.popleft().result() for _ in idx]).to(device)
            for _ in idx:
                j = next(src, None)
                if j is not None:
                    futs.append(pool.submit(_prep, paths[j]))
            age_b = age_all[idx]
            for mi, (model, meta, _w) in enumerate(members):
                img_size = meta["img_size"]
                x = F.interpolate(base, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
                x = (x - mean) / std
                with torch.no_grad():
                    for fl in flips:
                        xf = x
                        if fl & 1:
                            xf = torch.flip(xf, dims=[3])  # horizontal
                        if fl & 2:
                            xf = torch.flip(xf, dims=[2])  # vertical
                        with torch.amp.autocast("cuda", dtype=torch.float16,
                                                enabled=(device == "cuda")):
                            model(xf, age_b)
            t1 = time.perf_counter()
            if batch_idx >= args.warmup:
                per_image_ms.append((t1 - t0) * 1000.0 / len(idx))
            batch_idx += 1
        pool.shutdown()

    per_image_ms = np.array(per_image_ms, dtype=np.float64)
    if per_image_ms.size == 0:
        raise SystemExit("no batches timed (increase --warmup relative to data?)")

    def pct(q):
        return float(np.percentile(per_image_ms, q))

    mean_ms = float(per_image_ms.mean())
    p50 = pct(50)
    p95 = pct(95)
    p99 = pct(99)

    print("\n=== LATENCY (per-image ms; first %d warmup batches excluded) ===" % args.warmup)
    print(f"device={device}  models={n_models}  tta={args.tta}  "
          f"batch={args.batch_size}  prep_workers={args.prep_workers}")
    print(f"batches timed : {per_image_ms.size}")
    print(f"mean          : {mean_ms:.2f} ms")
    print(f"p50           : {p50:.2f} ms")
    print(f"p95           : {p95:.2f} ms")
    print(f"p99           : {p99:.2f} ms")

    # safety-margin verdict vs issue #110 targets
    if n_models == 1:
        target = 40.0
        tag = "single-model (target p95 <= 40 ms, 2.5x margin to 100ms T4 red line)"
    else:
        target = 50.0
        tag = "2-model (target p95 <= 50 ms, 2x margin to 100ms T4 red line)"
    verdict = "PASS (safety margin)" if p95 <= target else "CHECK (>= target; lower bound only)"
    print(f"\nsafety margin : {tag}")
    print(f"verdict       : {verdict}")
    print("NOTE: 5060 number is a LOWER BOUND; T4 p95 must be measured on a real T4 "
          "before claiming the 100 ms red line is met (issue #81).")

    # copy-paste command for the captain (re-run without profiling overhead is just predict.py)
    cmd = (f"python predict.py --images <test_images> --csv <test.csv> "
           f"--weights {' '.join(args.weights)}")
    if args.blend:
        cmd += f" --blend {' '.join(str(b) for b in blend)}"
    cmd += f" --tta {args.tta} --batch-size {args.batch_size} --prep-workers {args.prep_workers} --out submission.csv"
    print(f"\nreproduce (deployed predict.py):\n  {cmd}")


if __name__ == "__main__":
    main()
