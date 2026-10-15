"""Verify cache512 integrity: every CSV row has a readable cached image; report bad files."""
import csv
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE_ROOT, CSV_ROOT  # noqa: E402  自动探测（env → 仓库布局 → 旧路径）

CACHE = CACHE_ROOT
SPLITS = ["train", "valid", "test"]


def check(args):
    path, stem = args
    try:
        sz = os.path.getsize(path)
        if sz < 5000:
            return (stem, f"small:{sz}")
        from PIL import Image
        with Image.open(path) as im:
            if im.size != (512, 512):
                return (stem, f"size:{im.size}")
            im.load()
        return (stem, None)
    except Exception as e:
        return (stem, f"err:{type(e).__name__}")


def main():
    bad = []
    for s in SPLITS:
        names = []
        with open(os.path.join(CSV_ROOT, s + ".csv"), encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                names.append(r["Image_name"].rsplit(".", 1)[0])
        tasks = [(os.path.join(CACHE, s, n + ".jpg"), n) for n in names]
        missing = [n for n, p in [(n, os.path.join(CACHE, s, n + ".jpg")) for n in names] if not os.path.exists(p)]
        print(f"{s}: csv={len(names)} missing_from_cache={len(missing)} {missing[:5]}", flush=True)
        with Pool(8) as pool:
            res = pool.map(check, tasks, chunksize=128)
        b = [(n, e) for n, e in res if e]
        print(f"{s}: bad={len(b)} {b[:8]}", flush=True)
        bad += [(s, n, e) for n, e in b]
    print("TOTAL BAD:", len(bad))


if __name__ == "__main__":
    main()