"""Exact duplicate detection via md5 of file bytes (definitive, unlike dHash)."""
import hashlib
import os
import glob
import json
from collections import defaultdict
from multiprocessing import Pool

DATA_ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"
SPLITS = ["train", "valid", "test"]


def md5_file(p):
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return (p, h.hexdigest())
    except Exception as e:
        return (p, None)


def main():
    tasks = []
    for s in SPLITS:
        for p in sorted(glob.glob(os.path.join(DATA_ROOT, s, "*.[jJ][pP][gG]"))):
            tasks.append(p)
    print(f"hashing {len(tasks)} files")
    with Pool(8) as pool:
        res = pool.map(md5_file, tasks, chunksize=32)

    by_hash = defaultdict(list)
    for p, h in res:
        if h:
            by_hash[h].append(p)
    groups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}
    print(f"total files={len(res)} unique_md5={len(by_hash)} dup_groups={len(groups)}")
    cross = defaultdict(int)
    examples = []
    for h, ps in groups.items():
        splits = sorted({os.path.normpath(p).replace("\\", "/").split("/")[-2] for p in ps})
        key = "+".join(splits)
        cross[key] += 1
        if len(examples) < 30:
            examples.append((key, [os.path.basename(x) for x in ps]))
    print("dup groups by split-combo:", dict(cross))
    for k, ex in examples:
        print(" ", k, ex)
    with open(r"D:\glaucoma_work\md5_dupes.json", "w") as f:
        json.dump({h: ps for h, ps in groups.items()}, f, indent=1, ensure_ascii=False)
    print("saved D:\\glaucoma_work\\md5_dupes.json")


if __name__ == "__main__":
    main()