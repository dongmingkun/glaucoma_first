"""Dataset audit: label balance, age stats, native sizes, near-duplicate detection across splits."""
import csv
import os
import glob
import json

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

DATA_ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"
SPLITS = ["train", "valid", "test"]
OUT_DIR = r"D:\glaucoma_work"


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def stats(split, rows):
    print(f"\n=== {split}: {len(rows)} rows ===")
    if "True_Label" in rows[0]:
        cnt = Counter(r["True_Label"] for r in rows)
        n1 = cnt.get("1", 0)
        print(f"  labels: {dict(cnt)}  positive_rate={n1/len(rows):.4f}")
        ages = defaultdict(list)
        for r in rows:
            try:
                ages[r["True_Label"]].append(float(r["Age"]))
            except (ValueError, TypeError):
                ages[r["True_Label"]].append(float("nan"))
        for lab in sorted(ages):
            a = np.array(ages[lab])
            a = a[~np.isnan(a)]
            if len(a) == 0:
                continue
            print(f"  age[label={lab}]: n={len(a)} min={a.min():.0f} p25={np.percentile(a,25):.0f} "
                  f"med={np.median(a):.0f} p75={np.percentile(a,75):.0f} max={a.max():.0f} mean={a.mean():.1f}")
    else:
        a = np.array([float(r["Age"]) for r in rows if r.get("Age") not in (None, "")])
        print(f"  age: n={len(a)} min={a.min():.0f} med={np.median(a):.0f} max={a.max():.0f}")


def phash_worker(args):
    fpath, rel = args
    try:
        from PIL import Image
        im = Image.open(fpath)
        w, h = im.size
        im.draft("RGB", (128, 128))
        small = im.convert("RGB")
        g = small.convert("L").resize((9, 8), Image.BILINEAR)
        px = np.asarray(g, dtype=np.int16)
        diff = px[:, 1:] > px[:, :-1]  # 8x8 bool
        bits = 0
        for b in diff.flatten():
            bits = (bits << 1) | int(b)
        t = np.asarray(small.resize((1, 1), Image.BILINEAR)).reshape(-1)
        return (rel, w, h, int(bits), int(t[0]), int(t[1]), int(t[2]))
    except Exception:
        return (rel, -1, -1, None, 0, 0, 0)


def main():
    all_rows = {s: load_csv(os.path.join(DATA_ROOT, s + ".csv")) for s in SPLITS}
    for s in SPLITS:
        stats(s, all_rows[s])
    for s in SPLITS:
        csv_names = {r["Image_name"].rsplit(".", 1)[0].lower() for r in all_rows[s]}
        files = {os.path.basename(p).rsplit(".", 1)[0].lower()
                 for p in glob.glob(os.path.join(DATA_ROOT, s, "*"))}
        print(f"{s}: csv={len(csv_names)} files={len(files)} only_csv={len(csv_names-files)} only_fs={len(files-csv_names)}")
        if csv_names - files:
            print("  only_csv sample:", sorted(csv_names - files)[:3])
        if files - csv_names:
            print("  only_fs sample:", sorted(files - csv_names)[:3])

    tasks = []
    for s in SPLITS:
        for p in sorted(glob.glob(os.path.join(DATA_ROOT, s, "*.[jJ][pP][gG]"))):
            tasks.append((p, s + "/" + os.path.basename(p)))
    print(f"\nHashing {len(tasks)} images ...")
    with Pool(10, maxtasksperchild=500) as pool:
        results = pool.map(phash_worker, tasks, chunksize=64)

    sizes = Counter()
    per_split = {s: ([], [], np.zeros((0, 3))) for s in SPLITS}
    failed = []
    for rel, w, h, bits, r, g, b in results:
        if w < 0:
            failed.append(rel)
            continue
        sizes[(w, h)] += 1
        s = rel.split("/")[0]
        per_split[s][0].append(rel)
        per_split[s][1].append(bits)
    print(f"failed={len(failed)} {failed[:5]}")
    print("top native sizes:")
    for (w, h), c in sizes.most_common(20):
        print(f"  {w}x{h}: {c}")

    # within-split exact dup count
    for s in SPLITS:
        c = Counter(per_split[s][1])
        dups = sum(v - 1 for v in c.values() if v > 1)
        print(f"{s}: within-split exact dHash dups = {dups} ({sum(1 for v in c.values() if v>1)} hash groups)")

    def bitcount(x):
        return np.bitwise_count(x) if hasattr(np, "bitwise_count") else np.unpackbits(
            x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.uint16)

    print("numpy", np.__version__, "has bitwise_count:", hasattr(np, "bitwise_count"))

    def cross_compare(a_names, a_bits, b_names, b_bits, thr, chunk=4096):
        A = np.array(a_bits, dtype=np.uint64)
        B = np.array(b_bits, dtype=np.uint64)
        found = []
        for i in range(0, len(A), chunk):
            sub = A[i:i + chunk]
            X = np.bitwise_xor(sub[:, None], B[None, :])
            d = bitcount(X.ravel()).reshape(X.shape)
            idx = np.argwhere(d <= thr)
            for ai, bi in idx[:50]:
                found.append((a_names[i + ai], b_names[bi], int(d[ai, bi])))
            if len(found) >= 200:
                break
        return found

    for thr in (0, 3, 6):
        print(f"\n--- cross-split dHash distance <= {thr} ---")
        for s1, s2 in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            pairs = cross_compare(per_split[s1][0], per_split[s1][1],
                                  per_split[s2][0], per_split[s2][1], thr)
            print(f"{s1} vs {s2}: {len(pairs) if len(pairs)<200 else '>=200'} pairs (showing up to 10) {pairs[:10]}")

    with open(os.path.join(OUT_DIR, "audit_hashes.json"), "w") as f:
        json.dump({"splits": {s: per_split[s][0] for s in SPLITS},
                   "bits": {s: per_split[s][1] for s in SPLITS},
                   "sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()}}, f)
    print("\nsaved", os.path.join(OUT_DIR, "audit_hashes.json"))


if __name__ == "__main__":
    main()