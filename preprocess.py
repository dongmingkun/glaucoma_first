"""Preprocess fundus images: crop black border around fundus, resize to 512x512, cache as JPEG on D:.

Also writes a metadata CSV per split (original size, crop bbox, mask fraction, brightness).
"""
import csv
import os
import glob
import json
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from PIL import Image

DATA_ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"
OUT_ROOT = r"D:\glaucoma_work\cache512"
SPLITS = ["train", "valid", "test"]
SIZE = 512
MARGIN = 1.0  # extra margin around detected fundus bbox (1.0 = none)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_ROOT, help="output cache dir")
    ap.add_argument("--size", type=int, default=SIZE, help="output image size (square)")
    ap.add_argument("--draft", type=int, default=768, help="JPEG draft decode size (>= size)")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


ARGS = None  # set in main()


def center_square(bbox, w, h, scale=1.02):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * scale
    half = side / 2
    nx0, ny0 = int(round(cx - half)), int(round(cy - half))
    nx1, ny1 = int(round(cx + half)), int(round(cy + half))
    return nx0, ny0, nx1, ny1


def process(args, size=512, draft=768):
    src, dst, rel = args
    try:
        if os.path.exists(dst) and os.path.getsize(dst) > 5000:
            return (rel, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 2)  # 2 = skipped (already cached)
        im = Image.open(src)
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
        x0, y0, x1, y1 = center_square(bbox, w, h)
        # pad if out of bounds
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        crop = arr[y0c:y1c, x0c:x1c]
        if pad_l or pad_t or pad_r or pad_b:
            crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="constant")
        out = Image.fromarray(crop).resize((size, size), Image.BICUBIC)
        out.save(dst, quality=92, subsampling=1)
        return (rel, w, h, round(x0c), round(y0c), round(x1c), round(y1c), round(frac, 4),
                round(float(gray.mean()), 1), 1)
    except Exception as e:
        return (rel, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0)


LOG = None


def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()


def main():
    global LOG
    args = parse_args()
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    LOG = open(os.path.join(out_root, "preprocess_run.log"), "a", encoding="utf-8")
    tasks = []
    for s in SPLITS:
        os.makedirs(os.path.join(out_root, s), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(DATA_ROOT, s, "*.[jJ][pP][gG]"))):
            base = os.path.basename(p)
            stem = base.rsplit(".", 1)[0]
            dst = os.path.join(out_root, s, stem + ".jpg")
            tasks.append((p, dst, s + "/" + base))
    print(f"total {len(tasks)} images to process -> {out_root} ({args.size}px, draft {args.draft})", flush=True)
    log(f"\n=== start {time.strftime('%H:%M:%S')} total={len(tasks)} size={args.size} ===")
    from functools import partial
    fn = partial(process, size=args.size, draft=max(args.draft, args.size))
    rows = []
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=3000) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=16)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                log(f"  {i+1}/{len(tasks)}  {el:.0f}s  {(i+1)/el:.1f} img/s")
    rows.sort()
    meta = os.path.join(out_root, "meta_all.csv")
    with open(meta, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rel", "w", "h", "x0", "y0", "x1", "y1", "mask_frac", "mean_gray", "ok"])
        for r in rows:
            wr.writerow(r)
    nfail = sum(1 for r in rows if r[-1] == 0)
    nskip = sum(1 for r in rows if r[-1] == 2)
    print(f"done. failed={nfail} skipped={nskip}. meta -> {meta}")


if __name__ == "__main__":
    main()