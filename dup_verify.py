"""Decisive check: are the top test<->train matches actually the same picture?

1) proper control: train->train max-cos EXCLUDING self (random 2000 sample)
2) pixel-level comparison of the top pairs at full 512 resolution (MAE / corr / hash)
3) side-by-side contact sheet for visual confirmation
"""
import os

import numpy as np
from PIL import Image

CACHE = r"D:\glaucoma_work\cache512"
PAIRS = [("test_01606.jpg", "train_17947.jpg"), ("test_04929.jpg", "train_10766.jpg"),
         ("test_04183.jpg", "train_16904.jpg"), ("test_01085.jpg", "train_36239.jpg"),
         ("test_02488.jpg", "train_46952.jpg")]
# 对照组：随便抽几个 test->train 的低分匹配（应该是不相关的两只眼）
LOW = [("test_00000.jpg", "train_00000.jpg")]


def load(split, name):
    return np.asarray(Image.open(os.path.join(CACHE, split, name)).convert("RGB"), dtype=np.float32)


print("=" * 72)
print("【1】高分对的像素级对比（512×512）")
for t, tr in PAIRS:
    a, b = load("test", t), load("train", tr)
    mae = np.abs(a - b).mean()
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    # 逐块最大差异（检测是否有局部改动）
    h, w, _ = a.shape
    bs = 64
    blocks = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            blocks.append(np.abs(a[y:y+bs, x:x+bs] - b[y:y+bs, x:x+bs]).mean())
    blocks = np.array(blocks)
    print(f"  {t} <-> {tr}: MAE={mae:6.2f}  corr={corr:.5f}  "
          f"块差异 max={blocks.max():6.2f} p99={np.percentile(blocks,99):6.2f} 均={blocks.mean():6.2f}")

print("\n【2】对照：不相关的两张图")
for t, tr in LOW:
    a, b = load("test", t), load("train", tr)
    print(f"  {t} <-> {tr}: MAE={np.abs(a-b).mean():6.2f}  corr={np.corrcoef(a.ravel(), b.ravel())[0,1]:.5f}")

print("\n【3】正确的对照组：train->train 排除自身")
desc_tr = np.load(r"D:\glaucoma_work\dup_search\train_desc.npy")
rng = np.random.default_rng(1)
idx = rng.choice(len(desc_tr), size=2000, replace=False)
q = desc_tr[idx]
best = np.zeros(len(q), dtype=np.float32)
for i in range(0, len(q), 200):
    S = q[i:i+200] @ desc_tr.T
    for r in range(S.shape[0]):
        S[r, idx[i + r]] = -1.0          # 排除自身
    best[i:i+200] = S.max(axis=1)
qs = np.percentile(best, [50, 90, 99, 99.9])
print(f"  train->其他train（排除自身）: p50={qs[0]:.4f} p90={qs[1]:.4f} p99={qs[2]:.4f} p99.9={qs[3]:.4f}")
for t in (0.99, 0.98, 0.95):
    print(f"      > {t}: {(best > t).sum()} 张 ({(best > t).mean()*100:.2f}%)")

print("\n【4】test->train 对照（同口径）")
cos_te = np.load(r"D:\glaucoma_work\dup_search\cos_test_to_train.npy")
qs2 = np.percentile(cos_te, [50, 90, 99, 99.9])
print(f"  test->train: p50={qs2[0]:.4f} p90={qs2[1]:.4f} p99={qs2[2]:.4f} p99.9={qs2[3]:.4f}")
for t in (0.99, 0.98, 0.95):
    print(f"      > {t}: {(cos_te > t).sum()} 张 ({(cos_te > t).mean()*100:.2f}%)")

print("\n【5】生成并排对比图（前 3 对 + 1 对对照组）")
os.makedirs(r"D:\glaucoma_work\dup_check", exist_ok=True)
rows = []
for t, tr in PAIRS[:3]:
    rows.append((load("test", t).astype(np.uint8), load("train", tr).astype(np.uint8)))
for t, tr in LOW:
    rows.append((load("test", t).astype(np.uint8), load("train", tr).astype(np.uint8)))
sheet = np.zeros((len(rows) * 256, 512, 3), dtype=np.uint8)
for i, (a, b) in enumerate(rows):
    ra = np.asarray(Image.fromarray(a).resize((256, 256)))
    rb = np.asarray(Image.fromarray(b).resize((256, 256)))
    sheet[i*256:(i+1)*256, :256] = ra
    sheet[i*256:(i+1)*256, 256:] = rb
Image.fromarray(sheet).save(r"D:\glaucoma_work\dup_check\pairs.png")
print("  已保存 D:\\glaucoma_work\\dup_check\\pairs.png （左=test 图，右=最相似的 train 图）")