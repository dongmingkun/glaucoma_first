"""Near-duplicate search: test/valid vs train, using low-res normalized descriptors.

1) build 64x64 grayscale descriptors from cache512 for each split (multiprocess)
2) cosine search: each test (and a control sample of train) image vs all train images
3) compare distributions -> if test has a heavy tail above the train-train control,
   the splits share near-duplicate images (same eye, another capture).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

CACHE = r"D:\glaucoma_work\cache512"
OUT = r"D:\glaucoma_work\dup_search"
SIZE = 64


def desc_one(path):
    from PIL import Image
    try:
        im = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32).ravel()
        a -= a.mean()
        n = np.linalg.norm(a)
        return a / n if n > 1e-6 else a
    except Exception:
        return np.zeros(SIZE * SIZE, dtype=np.float32)


def build_split(split, workers=10):
    out = os.path.join(OUT, f"{split}_desc.npy")
    if os.path.exists(out):
        print(f"  {split}: 已存在，跳过")
        return
    files = []
    for root, _, names in os.walk(os.path.join(CACHE, split)):
        for n in names:
            if n.lower().endswith(".jpg"):
                files.append(os.path.join(root, n))
    files.sort()
    t0 = time.time()
    with Pool(workers) as pool:
        arrs = pool.map(desc_one, files, chunksize=64)
    m = np.stack(arrs).astype(np.float32)
    np.save(out, m)
    with open(os.path.join(OUT, f"{split}_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.basename(p) for p in files))
    print(f"  {split}: {m.shape} 构建完成 ({time.time()-t0:.0f}s)")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[1] 构建描述符")
    for s in ("train", "valid", "test"):
        build_split(s)

    train = np.load(os.path.join(OUT, "train_desc.npy"))
    test = np.load(os.path.join(OUT, "test_desc.npy"))
    valid = np.load(os.path.join(OUT, "valid_desc.npy"))
    tn = open(os.path.join(OUT, "train_names.txt"), encoding="utf-8").read().split("\n")
    te_n = open(os.path.join(OUT, "test_names.txt"), encoding="utf-8").read().split("\n")
    print(f"  train={train.shape} test={test.shape} valid={valid.shape}")

    def max_cos(query, ref, chunk=256, exclude_self=False):
        best = np.zeros(len(query), dtype=np.float32)
        arg = np.zeros(len(query), dtype=np.int64)
        for i in range(0, len(query), chunk):
            q = query[i:i + chunk]
            S = q @ ref.T                      # (chunk, N)
            if exclude_self:
                for r in range(S.shape[0]):
                    S[r, i + r] = -1.0
            j = S.argmax(axis=1)
            best[i:i + chunk] = S[np.arange(S.shape[0]), j]
            arg[i:i + chunk] = j
        return best, arg

    print("\n[2] test -> train 最近邻")
    t0 = time.time()
    cos_te, j_te = max_cos(test, train)
    print(f"  完成 ({time.time()-t0:.0f}s)")

    print("[3] train -> train 控制组（随机抽 3000 张，排除自身）")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), size=3000, replace=False)
    cos_tr, _ = max_cos(train[idx], train, exclude_self=False)  # 近似控制（含自身）
    print(f"  完成")

    def show(tag, c):
        qs = np.percentile(c, [50, 90, 99, 99.9, 99.99])
        print(f"  {tag}: p50={qs[0]:.4f} p90={qs[1]:.4f} p99={qs[2]:.4f} p99.9={qs[3]:.4f} p99.99={qs[4]:.4f}")
        for t in (0.99, 0.98, 0.95, 0.90):
            print(f"      > {t}: {(c > t).sum()} 张 ({(c > t).mean()*100:.2f}%)")

    print("\n[4] 结果对比")
    show("test->train  (被测)", cos_te)
    show("train->train (对照)", cos_tr)
    np.save(os.path.join(OUT, "cos_test_to_train.npy"), cos_te)
    np.save(os.path.join(OUT, "j_test_to_train.npy"), j_te)

    print("\n[5] 高分样例（test 图 -> 最相似的 train 图）")
    top = np.argsort(-cos_te)[:10]
    for i in top:
        print(f"  {te_n[i]:22s} cos={cos_te[i]:.4f}  <->  {tn[j_te[i]]:22s}")


if __name__ == "__main__":
    main()