# -*- coding: utf-8 -*-
"""#138：ens10（98.82 线上基线）× train+all 新成员的 test 融合候选生成。

test-only 探索（无干净 valid，不声称 clean 增益）：rank / logit 两种融合，w=0.1/0.2/0.3。
产出 6 个候选 CSV（Image_name,Probability,Pred_Label）+ 一致性报告
（Spearman、mean 漂移、top200 重叠）。候选**不自动提交**，由队长决定是否上线验证。

用法（仓库根目录）：
  python tools/fusion_ens10.py                     # 默认路径，见 --help
  python tools/fusion_ens10.py --self-test         # 合成数据自测（秒级）
"""
import argparse
import csv
import os
import sys

import numpy as np

ENS10_DEFAULT = "submission_20251018_ens10_cal.csv"
NEW_DEFAULT = "oof/cnxb_512_e16_ra_all_test_tta4.csv"
TEST_ORDER = "glaucoma_first/test.csv"
DEFAULT_WEIGHTS = (0.1, 0.2, 0.3)   # 向后兼容默认；可用 --weights 覆盖（#152 需 0.10/0.15/0.20/0.25）


def read_probs(path):
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            n = r["Image_name"]
            if n in d:
                sys.exit(f"ERROR: {path} Image_name 重复: {n}")
            p = float(r["Probability"])
            if not (0.0 <= p <= 1.0) or p != p:
                sys.exit(f"ERROR: {path} 概率越界/NaN")
            d[n] = p
    return d


def read_names(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["Image_name"] for r in csv.DictReader(f)]


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind="mergesort")
        sx = x[order]
        r = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def topk_overlap(a, b, k=200):
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / k


def run(ens10_path, new_path, out_dir, weights=DEFAULT_WEIGHTS):
    ens10 = read_probs(ens10_path)
    new = read_probs(new_path)
    if set(ens10) != set(new):
        sys.exit(f"ERROR: 名字集合不一致（ens10={len(ens10)} new={len(new)}）")
    if len(ens10) != 5855:
        sys.exit(f"ERROR: 行数 {len(ens10)} != 5855")
    names = list(ens10)   # preserve ens10 input order (official 5855) by default
    if os.path.exists(TEST_ORDER):
        order = read_names(TEST_ORDER)
        if set(order) != set(ens10):
            sys.exit(f"ERROR: 与官方 {TEST_ORDER} 名字集合不一致")
        names = order       # always write candidates in official test.csv order
    pe = np.asarray([ens10[n] for n in names])
    pn = np.asarray([new[n] for n in names])

    s = spearman(pe, pn)
    drift = abs(pn.mean() - pe.mean()) / max(pe.mean(), 1e-9)
    ov = topk_overlap(pe, pn)
    print(f"ens10={os.path.basename(ens10_path)} new={os.path.basename(new_path)} n={len(names)}")
    print(f"Spearman(ens10,new)={s:.5f}  mean: ens10={pe.mean():.5f} new={pn.mean():.5f} "
          f"漂移={drift:.1%}  top200 重叠={ov:.3f}")
    if s < 0.80:
        print("WARN: 相关性 <0.80（#138 判负参考线：漂移型成员，谨慎使用）")

    le = logit(pe)
    ln = logit(pn)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(new_path))[0]
    for mode, (a, b) in (("rank", (pe, pn)), ("logit", (le, ln))):
        for w in weights:
            if mode == "rank":
                ra = np.argsort(np.argsort(a)).astype(float) / len(a)
                rb = np.argsort(np.argsort(b)).astype(float) / len(b)
                mix = (1 - w) * ra + w * rb
            else:
                mix = 1.0 / (1.0 + np.exp(-((1 - w) * a + w * b)))
            out = os.path.join(out_dir, f"{stem}_ens10_{mode}_w{int(w * 100)}.csv")
            with open(out, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["Image_name", "Probability", "Pred_Label"])
                for n, v in zip(names, mix):
                    wr.writerow([n, f"{v:.6f}", 1 if v >= 0.5 else 0])
            print(f"wrote {out}  mean={mix.mean():.5f} 翻正数(vs ens10@0.5)="
                  f"{int(((mix >= 0.5) != (pe >= 0.5)).sum())}")
    print("NOTE: test-only 探索候选，不自动提交；最终是否提分由一次线上提交回答（失败不降低 98.82）。")


def self_test():
    rng = np.random.default_rng(0)
    n = 5855
    names = [f"test_{i:05d}.jpg" for i in range(n)]
    base = np.clip(rng.beta(0.4, 8, n), 1e-6, 1 - 1e-6)
    new = np.clip(base * rng.uniform(0.8, 1.2, n), 1e-6, 1 - 1e-6)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "ens10.csv")
        f2 = os.path.join(td, "new.csv")
        for path, p in ((f1, base), (f2, new)):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Image_name", "Probability"])
                for nm, v in zip(names, p):
                    w.writerow([nm, f"{v:.6f}"])
        run(f1, f2, os.path.join(td, "oof"))
    print("self-test PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens10", default=ENS10_DEFAULT, help="ens10 校准提交 CSV（98.82 基线）")
    ap.add_argument("--new", default=NEW_DEFAULT, help="#138 新成员 test TTA4 CSV")
    ap.add_argument("--out-dir", default="oof")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="logit/rank 融合权重（默认 0.1 0.2 0.3；#152 用 0.10 0.15 0.20 0.25）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not os.path.exists(args.ens10):
        sys.exit(f"ERROR: {args.ens10} 不存在（ens10 校准提交应在仓库根目录）")
    weights = tuple(args.weights) if args.weights else DEFAULT_WEIGHTS
    if any(w <= 0 or w >= 1 for w in weights):
        sys.exit(f"ERROR: 权重需在 (0,1) 内，收到 {weights}")
    run(args.ens10, args.new, args.out_dir, weights)


if __name__ == "__main__":
    main()
