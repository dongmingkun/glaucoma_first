"""Two-stage duplicate detection: coarse top-K by cosine, then precise MAE at 512.

Outputs counts of test images that are literal duplicates (MAE<=3) or same-eye
near-duplicates (MAE<=15) of a train image, plus the mapping for later use.
"""
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

DUP = r"D:\glaucoma_work\dup_search"
CACHE = r"D:\glaucoma_work\cache512"
TOP_K = 25


def load512(split, name):
    return np.asarray(Image.open(os.path.join(CACHE, split, name)).convert("RGB"), dtype=np.float32)


def worker(args):
    tname, cand_names = args
    try:
        a = load512("test", tname)
    except Exception:
        return (ti, [])
    out = []
    for tr in cand_names:
        try:
            b = load512("train", tr)
        except Exception:
            continue
        mae = float(np.abs(a - b).mean())
        if mae <= 30.0:
            out.append((tname, tr, round(mae, 2)))
    return (tname, out)


if __name__ == "__main__":
    train = np.load(os.path.join(DUP, "train_desc.npy"))
    test = np.load(os.path.join(DUP, "test_desc.npy"))
    TRAIN_NAMES = open(os.path.join(DUP, "train_names.txt"), encoding="utf-8").read().split("\n")
    TEST_NAMES = open(os.path.join(DUP, "test_names.txt"), encoding="utf-8").read().split("\n")
    print(f"train={train.shape} test={test.shape}")

    print(f"stage1: 每张 test 找 top-{TOP_K} 候选 …", flush=True)
    t0 = time.time()
    cand_idx = np.zeros((len(test), TOP_K), dtype=np.int64)
    for i in range(0, len(test), 256):
        S = test[i:i + 256] @ train.T
        part = np.argpartition(-S, TOP_K, axis=1)[:, :TOP_K]
        cand_idx[i:i + 256] = part
    print(f"  完成 ({time.time()-t0:.0f}s)")

    tasks = [(TEST_NAMES[i], [TRAIN_NAMES[j] for j in cand_idx[i]]) for i in range(len(test))]
    print("stage2: 512 分辨率像素级复核 …", flush=True)
    t0 = time.time()
    dup, near = [], []
    with Pool(10) as pool:
        for n, (tname, res) in enumerate(pool.imap_unordered(worker, tasks, chunksize=16)):
            for tn, tr, mae in res:
                tname = tn
                if mae <= 3.0:
                    dup.append((tname, tr, mae))
                elif mae <= 12.0:
                    near.append((tname, tr, mae))
            if (n + 1) % 1000 == 0:
                print(f"  {n+1}/{len(tasks)}  dup={len(dup)} near={len(near)} ({time.time()-t0:.0f}s)", flush=True)

    print("\n==== 结果 ====")
    print(f"  MAE<=3  （同图重编码，几乎确定是重复）: {len(dup)} 张 / {len(test)} = {len(dup)/len(test)*100:.2f}%")
    print(f"  3<MAE<=12（同一只眼、另一次拍摄）    : {len(near)} 张 = {len(near)/len(test)*100:.2f}%")
    with open(os.path.join(DUP, "duplicates.csv"), "w", encoding="utf-8", newline="") as f:
        import csv as _c
        w = _c.writer(f)
        w.writerow(["test_image", "train_image", "mae"])
        for r in sorted(dup, key=lambda x: x[2]):
            w.writerow(r)
    print("  明细已存 D:\\glaucoma_work\\dup_search\\duplicates.csv")
    print("\n  前 10 对（MAE 升序）:")
    for r in sorted(dup, key=lambda x: x[2])[:10]:
        print("   ", r)
    print("\n  near 样例（前 8）:")
    for r in sorted(near, key=lambda x: x[2])[:8]:
        print("   ", r)