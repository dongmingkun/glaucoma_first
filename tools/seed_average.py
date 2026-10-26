# -*- coding: utf-8 -*-
"""#145 seed 平均 + ens10 融合候选（CPU-only，numpy 依赖，无需 GPU）。

背景
----
#138 的 train+all Base@512 RA（seed 7）线上实测 98.84（+0.02 vs 98.82），w10 融合首次真正提分，
但单 seed 的 test-only 证据仍薄。本工具把 #138（seed7）与 #145（seed42）在 **logit 空间等权平均**，
降低单 seed 方差；再与 ens10（98.82 基线）做 logit/rank 融合，按 #145 指定的权重
（seed 平均路径 w=0.10/0.15/0.20；可改）生成候选。

说明：这是 **test-only 探索**（train+all 成员无干净 valid，不声称 clean 增益）；
候选不自动提交，最终是否提分由一次线上提交回答，失败不降低 98.82。

用法
----
  # 纯 seed 平均（logit 等权；#138 seed7 的 test CSV 在仓库根，不是 oof/）
  python tools/seed_average.py \
    --members cnxb_512_e16_ra_all_test_tta4.csv oof/cnxb_512_e16_ra_all_s42_test_tta4.csv \
    --weights 0.5 0.5 --out oof/seed7_s42_test_tta4.csv

  # seed 平均 + 与 ens10 logit 融合（w=0.10/0.15/0.20，按 #145）
  python tools/seed_average.py \
    --members cnxb_512_e16_ra_all_test_tta4.csv oof/cnxb_512_e16_ra_all_s42_test_tta4.csv \
    --weights 0.5 0.5 --fuse-ens10 submission_20251018_ens10_cal.csv \
    --fuse-weights 0.10 0.15 0.20 --out-dir oof --tag seed7_s42

  # 自测（不需要数据文件，CPU 秒级）
  python tools/seed_average.py --self-test
"""
import argparse
import csv
import os
import sys

import numpy as np

TEST_ORDER_DEFAULT = "glaucoma_first/test.csv"
ENS10_DEFAULT = "submission_20251018_ens10_cal.csv"
EXPECTED_ROWS = 5855


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -40, 40)))


def rank01(a):
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2.0       # 并列取平均秩，避免 rank 融合人为拆散 ties
        i = j
    return r / max(n - 1, 1)


def spearman(a, b):
    return float(np.corrcoef(rank01(a), rank01(b))[0, 1])


def read_names(path):
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(r["Image_name"])
    return out


def read_probs(path):
    d = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if "Probability" not in (rd.fieldnames or []):
            sys.exit(f"ERROR: {path} 缺少 Probability 列")
        for r in rd:
            d[r["Image_name"]] = float(r["Probability"])
    return d


def write_csv(path, names, probs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, probs):
            w.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])


def validate(name, probs, expected=EXPECTED_ROWS):
    if len(probs) != expected:
        sys.exit(f"ERROR: {name} 行数 {len(probs)} != {expected}")
    if not np.isfinite(probs).all():
        sys.exit(f"ERROR: {name} 含 NaN/Inf")
    if probs.min() < 0 or probs.max() > 1:
        sys.exit(f"ERROR: {name} 概率越界 [{probs.min():.4f},{probs.max():.4f}]")


def _align(d, names, path):
    missing = [n for n in names if n not in d]
    if missing:
        sys.exit(f"ERROR: {path} 缺 {len(missing)} 个 Image_name（示例 {missing[:3]}）")
    return np.asarray([d[n] for n in names], dtype=np.float64)


def run(member_paths, weights, test_csv, out, fuse_ens10, fuse_weights, fuse_mode,
        out_dir, tag):
    names = read_names(test_csv)
    if len(names) != EXPECTED_ROWS:
        sys.exit(f"ERROR: {test_csv} 行数 {len(names)} != {EXPECTED_ROWS}")
    mats = [_align(read_probs(p), names, p) for p in member_paths]
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(mats):
        sys.exit(f"ERROR: --weights 数量 {len(w)} != 成员数 {len(mats)}")
    if (w <= 0).any() or w.sum() <= 0:
        sys.exit("ERROR: --weights 必须为正")
    w = w / w.sum()
    print("成员:", [os.path.basename(p) for p in member_paths], "权重:", np.round(w, 4).tolist())

    P = np.vstack(mats)                       # (M, N)
    L = np.sum([wi * logit(row) for wi, row in zip(w, P)], axis=0)
    avg = sigmoid(L)
    validate("seed_avg", avg, len(names))
    for i, p in enumerate(member_paths):
        print(f"  {os.path.basename(p):48s} mean={P[i].mean():.5f} "
              f"Spearman(avg,{i})={spearman(avg, P[i]):.5f}")
    write_csv(out, names, avg)
    print(f"wrote {out}  mean={avg.mean():.5f}  n={len(avg)}")

    if fuse_ens10:
        pe = _align(read_probs(fuse_ens10), names, fuse_ens10)
        validate("ens10", pe, len(names))
        s = spearman(pe, avg)
        drift = abs(avg.mean() - pe.mean()) / max(pe.mean(), 1e-9)
        k = 200
        ov = len(set(np.argsort(-pe)[:k].tolist()) & set(np.argsort(-avg)[:k].tolist())) / k
        print(f"\nens10={os.path.basename(fuse_ens10)}  Spearman(ens10,seed_avg)={s:.5f}  "
              f"mean: ens10={pe.mean():.5f} avg={avg.mean():.5f} 漂移={drift:.1%} top200重叠={ov:.3f}")
        if s < 0.80:
            print("WARN: 相关性 <0.80（#138 判负参考线），谨慎使用")
        os.makedirs(out_dir, exist_ok=True)
        stem = tag or "seed_avg"
        for fw in fuse_weights:
            if fuse_mode == "logit":
                mix = sigmoid((1 - fw) * logit(pe) + fw * logit(avg))
                name = f"{stem}_ens10_logit_w{int(round(fw * 100))}.csv"
            else:
                mix = (1 - fw) * rank01(pe) + fw * rank01(avg)
                name = f"{stem}_ens10_rank_w{int(round(fw * 100))}.csv"
            p = os.path.join(out_dir, name)
            validate(name, mix, len(names))
            write_csv(p, names, mix)
            flip = int(((mix >= 0.5) != (pe >= 0.5)).sum())
            print(f"wrote {p}  mean={mix.mean():.5f} 翻正数(vs ens10@0.5)={flip}")
        print("NOTE: test-only 探索候选，不自动提交；是否提分由一次线上提交回答（失败不降低 98.82）。")
    return 0


def self_test():
    rng = np.random.default_rng(0)
    n = EXPECTED_ROWS
    names = [f"test_{i:05d}.jpg" for i in range(n)]
    base = np.clip(rng.beta(0.5, 9, n), 1e-6, 1 - 1e-6)
    m1 = np.clip(base * rng.uniform(0.8, 1.2, n), 1e-6, 1 - 1e-6)
    m2 = np.clip(base * rng.uniform(0.8, 1.2, n), 1e-6, 1 - 1e-6)
    ens10 = np.clip(base * rng.uniform(0.7, 1.3, n), 1e-6, 1 - 1e-6)

    def _w(path, names, p, with_label=False):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image_name", "Probability", "Pred_Label"] if with_label
                       else ["Image_name", "Probability"])
            for nm, v in zip(names, p):
                w.writerow([nm, f"{v:.6f}"] + ([1 if v >= 0.5 else 0] if with_label else []))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_csv = os.path.join(td, "test.csv")
        with open(test_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Image_name", "Age"])
            for nm in names:
                w.writerow([nm, 50])
        p1 = os.path.join(td, "m1.csv"); p2 = os.path.join(td, "m2.csv"); pe = os.path.join(td, "ens10.csv")
        _w(p1, names, m1, True); _w(p2, names, m2, True); _w(pe, names, ens10, True)
        # 纯平均
        avg = sigmoid(0.5 * logit(m1) + 0.5 * logit(m2))
        # 手算一致性
        a2 = sigmoid(0.5 * np.log(m1 / (1 - m1)) + 0.5 * np.log(m2 / (1 - m2)))
        assert np.allclose(avg, a2, atol=1e-9), "logit 平均不一致"
        # 并列秩：rank 融合不能人为拆散 ties
        rt = rank01(np.array([0.1, 0.1, 0.5]))
        assert np.allclose(rt, [0.25, 0.25, 1.0]), f"并列秩异常: {rt}"
        run([p1, p2], [0.5, 0.5], test_csv, os.path.join(td, "seed_avg.csv"),
            pe, [0.10, 0.15, 0.20], "logit", os.path.join(td, "oof"), "seed7_s42")
        outs = sorted(os.listdir(os.path.join(td, "oof")))
        assert len([x for x in outs if x.endswith(".csv")]) == 3, f"候选数不对: {outs}"
        # 权重归一：2 倍权重等价
        assert np.allclose(sigmoid(logit(m1)), sigmoid(0.5 * 2 * logit(m1))), "权重归一异常"
    print("self-test PASS ✔")
    return 0


def main():
    ap = argparse.ArgumentParser(description="#145 seed 平均 + ens10 融合（CPU）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--members", nargs="+", default=None, help="待平均的 test TTA4 CSV（按 Image_name 对齐）")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="各成员权重（默认等权）")
    ap.add_argument("--test-csv", default=TEST_ORDER_DEFAULT, help="官方 test 顺序（5855 行）")
    ap.add_argument("--out", default="oof/seed_avg_test_tta4.csv", help="纯平均输出 CSV")
    ap.add_argument("--fuse-ens10", default=None, help="ens10 基线 CSV（给了才做融合）")
    ap.add_argument("--fuse-weights", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    ap.add_argument("--fuse-mode", default="logit", choices=["logit", "rank"])
    ap.add_argument("--out-dir", default="oof")
    ap.add_argument("--tag", default="seed_avg", help="融合候选文件名前缀")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.members:
        sys.exit("需要 --members（或用 --self-test）")
    weights = args.weights if args.weights else [1.0] * len(args.members)
    return run(args.members, weights, args.test_csv, args.out,
               args.fuse_ens10, args.fuse_weights, args.fuse_mode, args.out_dir, args.tag)


if __name__ == "__main__":
    sys.exit(main())
