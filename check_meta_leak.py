"""Check EXIF / metadata and ordering artifacts as possible label leaks.

1) EXIF fields present in train/valid/test samples, and per-label distributions
2) Does the label correlate with the file index in train? (ordering artifact)
3) Any filename-derived signal (e.g. consecutive duplicates, index parity)
"""
import csv
import glob
import os
import random
from collections import Counter, defaultdict

from PIL import Image
from PIL.ExifTags import TAGS

ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"


def exif_of(path):
    try:
        im = Image.open(path)
        ex = im.getexif()
        out = {}
        if ex:
            for k, v in ex.items():
                out[TAGS.get(k, str(k))] = str(v)[:60]
        return out
    except Exception as e:
        return {"__err__": type(e).__name__}


def main():
    print("=" * 72)
    print("【1】EXIF 字段（每类抽 200 张）")
    labels = {}
    with open(os.path.join(ROOT, "train.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labels[r["Image_name"].rsplit(".", 1)[0]] = r["True_Label"]
    pos = [k for k, v in labels.items() if v == "1"]
    neg = [k for k, v in labels.items() if v == "0"]
    random.Random(0).shuffle(pos)
    random.Random(0).shuffle(neg)
    for tag, keys in (("label=1", pos[:200]), ("label=0", neg[:200])):
        cnt = Counter()
        for k in keys:
            for kk in exif_of(os.path.join(ROOT, "train", k + ".JPG")):
                cnt[kk] += 1
        print(f"  train {tag}: 有 EXIF 的字段 -> {dict(cnt.most_common(12))}")

    print("\n  样例 EXIF（3 张 label=1 + 3 张 label=0）:")
    for k in pos[:3] + neg[:3]:
        e = exif_of(os.path.join(ROOT, "train", k + ".JPG"))
        print(f"    {k} ({labels[k]}): {e}")

    print("\n【2】test 的 EXIF（抽 50 张）")
    tcnt = Counter()
    tfiles = sorted(glob.glob(os.path.join(ROOT, "test", "*")))[:50]
    for p in tfiles:
        for kk in exif_of(p):
            tcnt[kk] += 1
    print(f"  字段 -> {dict(tcnt.most_common(12))}")
    for p in tfiles[:3]:
        print(f"    {os.path.basename(p)}: {exif_of(p)}")

    print("\n" + "=" * 72)
    print("【3】训练集里 标签 vs 文件序号 的相关性（每 500 张一块）")
    names = sorted(labels.keys(), key=lambda x: int(x.split("_")[1]))
    blocks = [names[i:i + 500] for i in range(0, len(names), 500)]
    rates = []
    for b in blocks:
        r = sum(1 for k in b if labels[k] == "1") / len(b)
        rates.append(r)
    print(f"  整体阳性率 {sum(1 for v in labels.values() if v=='1')/len(labels):.4f}")
    print("  各块阳性率（前 20 块）:", " ".join(f"{r:.2f}" for r in rates[:20]))
    print(f"  块阳性率 min={min(rates):.3f} max={max(rates):.3f} 方差={sum((r-sum(rates)/len(rates))**2 for r in rates)/len(rates):.5f}")

    print("\n【4】相邻文件同标签率（流水线排序痕迹）")
    same = sum(1 for a, b in zip(names, names[1:]) if labels[a] == labels[b])
    print(f"  相邻同标签: {same}/{len(names)-1} = {same/(len(names)-1)*100:.1f}% （随机基线约 93.7%）")


if __name__ == "__main__":
    main()