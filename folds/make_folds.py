# -*- coding: utf-8 -*-
"""issue #109 — Task 1：train+valid 按标签分层 5 折切分（纯 CPU，零依赖）。

用法：
    python make_folds.py            # 默认从 ../glaucoma_first/ 或 ./data/ 读取
    python make_folds.py --data-dir data --out-dir folds --seed 42 --n-folds 5

产出（写入 --out-dir）：
    fold0_train.csv ... fold4_train.csv   # 每折训练集 = 其余 4 折（4/5）
    fold0_valid.csv ... fold4_valid.csv   # 每折验证集 = held-out 1/5（OOF 评估用）
    split_manifest.json                   # 每折行数/正例数/比例 + 切分参数（可复现）

约定：
    - 输入列 Image_name,Age,True_Label；输出保持相同列结构，训练脚本可直接消费。
    - 仅在 train+valid 上分层切分；test.csv 完全不触碰（顺序/内容不变）。
    - 切分确定性：固定 seed，pos/neg 各自洗牌后按 index%n 轮转分配，保证每折正例比例≈全局。
    - 训练时按 #109 用 `--extra-rows` 把本折 4/5 的 valid 行并入训练；本脚本只产出 fold CSV。
"""
import argparse
import csv
import json
import os
import random
import sys

COLS = ["Image_name", "Age", "True_Label"]
SEED = 42
N_FOLDS = 5


def load(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        rows = [{"Image_name": x["Image_name"],
                 "Age": x["Age"],
                 "True_Label": x["True_Label"]} for x in r]
    return rows


def stratified_folds(rows, n_folds, seed):
    pos = [r for r in rows if r["True_Label"] == "1"]
    neg = [r for r in rows if r["True_Label"] == "0"]
    rp = random.Random(seed)
    rn = random.Random(seed)
    rp.shuffle(pos)
    rn.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for i, x in enumerate(pos):
        folds[i % n_folds].append(x)
    for i, x in enumerate(neg):
        folds[i % n_folds].append(x)
    return folds


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="含 train.csv/valid.csv 的目录（默认自动探测 data/ 或 ../glaucoma_first/）")
    ap.add_argument("--out-dir", default="folds")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    # 探测数据目录
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if args.data_dir:
        candidates.append(args.data_dir)
    candidates += [os.path.join(here, "data"),
                   os.path.join(here, "..", "glaucoma_first"),
                   os.path.join(here, "glaucoma_first")]
    data_dir = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, "train.csv")) and \
           os.path.isfile(os.path.join(c, "valid.csv")):
            data_dir = c
            break
    if data_dir is None:
        print("找不到 train.csv/valid.csv，请用 --data-dir 指定", file=sys.stderr)
        return 1

    train = load(os.path.join(data_dir, "train.csv"))
    valid = load(os.path.join(data_dir, "valid.csv"))
    combined = train + valid
    n_total = len(combined)
    n_pos = sum(1 for r in combined if r["True_Label"] == "1")
    print("合并 train+valid: %d 行, 正例 %d (%.4f)" % (n_total, n_pos, n_pos / n_total))

    folds = stratified_folds(combined, args.n_folds, args.seed)
    out = os.path.join(here, args.out_dir)
    os.makedirs(out, exist_ok=True)

    manifest = {"seed": args.seed, "n_folds": args.n_folds,
                "n_total": n_total, "n_pos_total": n_pos,
                "folds": []}
    all_valid_imgs = set()
    for k, fk in enumerate(folds):
        fv = fk                                   # held-out 1/5
        ft = [r for i, f in enumerate(folds) if i != k for r in f]  # 其余 4/5
        # 防泄漏：train 与 valid 不得有重复 Image_name
        sv, st = {r["Image_name"] for r in fv}, {r["Image_name"] for r in ft}
        assert sv.isdisjoint(st), "fold %d 出现 train/valid 重叠!" % k
        assert sv.isdisjoint(all_valid_imgs), "fold %d 与前面折的 valid 重叠!" % k
        all_valid_imgs |= sv

        pv = sum(1 for r in fv if r["True_Label"] == "1")
        pt = sum(1 for r in ft if r["True_Label"] == "1")
        write_csv(os.path.join(out, "fold%d_valid.csv" % k), fv)
        write_csv(os.path.join(out, "fold%d_train.csv" % k), ft)
        manifest["folds"].append({
            "fold": k,
            "train_rows": len(ft), "train_pos": pt,
            "train_pos_ratio": round(pt / len(ft), 6),
            "valid_rows": len(fv), "valid_pos": pv,
            "valid_pos_ratio": round(pv / len(fv), 6),
        })
        print("fold%d: train %d (pos %d, %.4f) | valid %d (pos %d, %.4f)"
              % (k, len(ft), pt, pt / len(ft), len(fv), pv, pv / len(fv)))

    # 完整性校验：5 折 valid 并集 == 全部 combined，且无重复
    assert len(all_valid_imgs) == n_total, "valid 并集行数 != 总数"
    assert sum(len(f) for f in folds) == n_total
    manifest["verified"] = True
    with open(os.path.join(out, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("切分完成，已写入 %s/ （含 split_manifest.json）" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
