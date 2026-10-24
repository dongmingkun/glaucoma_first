# -*- coding: utf-8 -*-
"""RESULTS 报告用 CPU 指标工具：AUROC/AUPRC、两两 Spearman、blend 权重扫描 + 配对 bootstrap。

服务 #125/#126/#128 的结果回填（docs/ISSUE_109_RESULTS.md、ISSUE_117_RERUN_RESULTS.md、
ISSUE_117_RA_RESULTS.md），也用于 #127 前的成员初筛。只依赖 numpy/sklearn（仓库已有），
不需要 GPU / 图像缓存。

输入约定：
- 标签：一个或多个含 ``Image_name,True_Label`` 的 CSV（如 ``glaucoma_first/train.csv
  glaucoma_first/valid.csv`` 拼成 57245 官方顺序；或单折 ``folds3/fold0_valid.csv``）；
- 概率：``.csv`` 需含 ``Image_name,Probability``（按名字对齐，顺序无关）；
  ``.npy`` 一维数组，**按 labels 顺序**（如 ``oof/oof_probs.npy`` 对 57245 官方顺序）。

用法：
  # 自测（合成数据，秒级，不需要任何真数据）
  python tools/oof_report.py --self-test

  # 每折 held-out valid 指标（#126 每折 AUROC/AUPRC）
  python tools/oof_report.py metrics --labels folds3/fold0_valid.csv --probs oof/cnxs_fold0_valid_tta4.csv

  # OOF 57245 整体指标（#126）
  python tools/oof_report.py metrics \
      --labels glaucoma_first/train.csv glaucoma_first/valid.csv --probs oof/oof_probs.npy

  # 3 折两两 Spearman（#126 验收 <0.98；用 test 概率；test.csv 无 True_Label）
  python tools/oof_report.py metrics --names-only --labels glaucoma_first/test.csv \
      --probs oof/cnxs_fold0_test_tta4.csv oof/cnxs_fold1_test_tta4.csv oof/cnxs_fold2_test_tta4.csv

  # 新成员 vs 基线（如 ens9）：blend w 扫描 + 配对 bootstrap（#125/#128/#126）
  python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
      --base ens9_valid_probs_tta4.csv --new oof/cnxb_512_e16_cloud_valid_last_tta4.csv
"""
import argparse
import csv
import os
import sys
import tempfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_labels(paths, names_only=False):
    """读标签 CSV，返回 (names, y)；names_only=True 时 y=None（test.csv 无标签）。"""
    names, y = [], []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if "Image_name" not in r:
                    sys.exit(f"ERROR: {path} 缺 Image_name 列")
                if not names_only and "True_Label" not in r:
                    sys.exit(f"ERROR: {path} 缺 True_Label 列（test 用 --names-only）")
                names.append(r["Image_name"])
                if not names_only:
                    y.append(int(float(r["True_Label"])))
    if len(set(names)) != len(names):
        sys.exit("ERROR: 标签 CSV 里 Image_name 有重复")
    if names_only:
        return names, None
    y = np.asarray(y, dtype=np.int64)
    if not np.isin(y, (0, 1)).all():
        sys.exit("ERROR: True_Label 不是 0/1")
    return names, y


def read_probs(path, names):
    """读概率文件并对齐到 names 顺序。csv 按名字对齐；npy 按 labels 顺序。"""
    tag = os.path.basename(path)
    if path.endswith(".npy"):
        p = np.load(path).astype(np.float64).ravel()
        if len(p) != len(names):
            sys.exit(f"ERROR: {tag} 长度 {len(p)} != labels {len(names)}（npy 必须按官方顺序）")
    else:
        d = {}
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if "Image_name" not in r or "Probability" not in r:
                    sys.exit(f"ERROR: {path} 缺 Image_name/Probability 列")
                d[r["Image_name"]] = float(r["Probability"])
        miss = [n for n in names if n not in d]
        if miss:
            sys.exit(f"ERROR: {tag} 缺 {len(miss)} 个名字（如 {miss[:3]}）")
        extra = set(d) - set(names)
        if extra:
            sys.exit(f"ERROR: {tag} 多出 {len(extra)} 个不在 labels 里的名字（如 {sorted(extra)[:3]}）")
        p = np.asarray([d[n] for n in names], dtype=np.float64)
    if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        sys.exit(f"ERROR: {tag} 概率含 NaN/inf 或超出 [0,1]")
    return p


def spearman(a, b):
    """Spearman 秩相关（平均秩，tie 安全；不依赖 scipy）。"""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(1, len(x) + 1, dtype=np.float64)
        # 同值取平均秩
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")   # 常数向量（如退化预测）：秩相关无定义，如实返回 NaN
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_ap(y, p):
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def blend_at(base, new, w, mode="prob"):
    if mode == "prob":
        return (1 - w) * base + w * new
    lo_b, lo_n = np.log(np.clip(base, 1e-6, 1)), np.log(np.clip(new, 1e-6, 1))
    return 1.0 / (1.0 + np.exp(-(lo_b + w * (lo_n - lo_b))))


def cmd_metrics(args):
    names, y = read_labels(args.labels, names_only=args.names_only)
    probs = {os.path.basename(p): read_probs(p, names) for p in args.probs}
    if y is None:
        print(f"n={len(names)} (names-only; AUROC/AUPRC skipped)")
    else:
        print(f"n={len(y)} pos={int(y.sum())} ({y.mean():.4%})")
    for tag, p in probs.items():
        if y is not None:
            a, ap = auc_ap(y, p)
            print(f"  {tag:44s} AUROC={a:.5f} AUPRC={ap:.5f} "
                  f"mean={p.mean():.5f} std={p.std():.5f} min={p.min():.5f} max={p.max():.5f}")
        else:
            print(f"  {tag:44s} mean={p.mean():.5f} std={p.std():.5f} "
                  f"min={p.min():.5f} max={p.max():.5f}")
    tags = list(probs)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            s = spearman(probs[tags[i]], probs[tags[j]])
            print(f"  Spearman({tags[i]}, {tags[j]}) = {s:.5f}")


def cmd_blend(args):
    names, y = read_labels(args.labels)
    base = read_probs(args.base, names)
    new = read_probs(args.new, names)
    a_b, ap_b = auc_ap(y, base)
    a_n, ap_n = auc_ap(y, new)
    print(f"n={len(y)} pos={int(y.sum())}")
    print(f"  base {os.path.basename(args.base):40s} AUROC={a_b:.5f} AUPRC={ap_b:.5f}")
    print(f"  new  {os.path.basename(args.new):40s} AUROC={a_n:.5f} AUPRC={ap_n:.5f}")
    print(f"  Spearman(base,new) = {spearman(base, new):.5f}")

    ws = np.round(np.arange(0.0, args.w_max + 1e-9, args.w_step), 4)
    rng = np.random.default_rng(args.seed)
    for mode in args.modes.split(","):
        aucs = [auc_ap(y, blend_at(base, new, w, mode))[0] for w in ws]
        k = int(np.argmax(aucs))
        best_w, best_auc = float(ws[k]), aucs[k]
        line = "  " + " ".join(f"{w:.2f}:{a:.5f}" for w, a in zip(ws, aucs))
        print(f"  [{mode}] w 扫描: {line}")
        print(f"  [{mode}] best w={best_w:.2f} AUROC={best_auc:.5f} Δ={best_auc - a_b:+.5f}")
        # 配对 bootstrap：best w 的 blend vs base
        pb = blend_at(base, new, best_w, mode)
        deltas = np.empty(args.bootstrap)
        n = len(y)
        for b in range(args.bootstrap):
            idx = rng.integers(0, n, n)
            deltas[b] = roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], base[idx])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_gt = float((deltas > 0).mean())
        print(f"  [{mode}] paired bootstrap({args.bootstrap}) Δ 95%CI=[{lo:+.5f},{hi:+.5f}] "
              f"P(Δ>0)={p_gt:.3f}")
    print("  NOTE: best w 是在同一批 labels 上选的，bootstrap 是条件推断、偏乐观；"
          "#127 决策请用 nested CV 或预指定 w。")


def self_test():
    rng = np.random.default_rng(0)
    n, prev = 4000, 0.05
    y = (rng.random(n) < prev).astype(np.int64)
    names = [f"img{i:05d}.jpg" for i in range(n)]
    z = rng.standard_normal(n)
    base = np.clip(0.5 + 0.9 * y + 0.35 * z, 1e-4, 1 - 1e-4)
    new = np.clip(base + 0.10 * y * rng.random(n) + 0.05 * rng.standard_normal(n), 1e-4, 1 - 1e-4)
    with tempfile.TemporaryDirectory() as td:
        lab = os.path.join(td, "valid.csv")
        with open(lab, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image_name", "Age", "True_Label"])
            for nm, yy in zip(names, y):
                w.writerow([nm, 55, yy])
        # 打乱行序的 CSV：验证按名字对齐
        perm = rng.permutation(n)
        fb, fn = os.path.join(td, "base.csv"), os.path.join(td, "new.csv")
        for path, p in ((fb, base), (fn, new)):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Image_name", "Probability", "Pred_Label"])
                for i in perm:
                    w.writerow([names[i], f"{p[i]:.6f}", int(p[i] >= 0.5)])
        npy = os.path.join(td, "probs.npy")
        np.save(npy, base)
        # metrics：csv 与 npy 应给出一致的 base AUROC
        ns, yy = read_labels([lab])
        assert ns == names and (yy == y).all()
        p_csv, p_npy = read_probs(fb, ns), read_probs(npy, ns)
        # CSV 以 6 位小数落盘，对齐后允许 5e-7 舍入误差；npy 精确
        assert np.allclose(p_csv, base, atol=1e-6) and np.array_equal(p_npy, base)
        a_b, ap_b = auc_ap(y, p_csv)
        a_n, ap_n = auc_ap(y, read_probs(fn, ns))
        assert 0.6 < a_b < 1.0 and 0.6 < a_n < 1.0 and 0.2 < ap_b, ap_n
        s_bn = spearman(base, new)
        assert 0.5 < s_bn < 1.0, s_bn
        assert abs(spearman(base, base) - 1.0) < 1e-9
        # 常数向量（全并列、零方差）：秩相关无定义，返回 NaN 而不是崩溃/假装有值
        assert np.isnan(spearman(base, np.full(n, 0.5)))
        # blend：best 不少于 base，bootstrap P(Δ>0) 在 [0,1]
        ws = np.round(np.arange(0, 0.52, 0.02), 4)
        aucs = [auc_ap(y, blend_at(base, new, w))[0] for w in ws]
        assert max(aucs) >= aucs[0] - 1e-12
        pb = blend_at(base, new, float(ws[int(np.argmax(aucs))]))
        dl = []
        for _ in range(300):
            idx = rng.integers(0, n, n)
            dl.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], base[idx]))
        pgt = float((np.asarray(dl) > 0).mean())
        assert 0.0 <= pgt <= 1.0
        # 错误路径：缺列 / 名字不匹配 / NaN
        bad = os.path.join(td, "bad.csv")
        with open(bad, "w", newline="", encoding="utf-8") as f:
            f.write("Image_name,Probability\nfoo.jpg,0.5\n")
        try:
            read_probs(bad, ns)
            raise AssertionError("应当因名字不匹配退出")
        except SystemExit:
            pass
        # names-only：test.csv 没有 True_Label，只对齐/算 Spearman
        names_csv = os.path.join(td, "test_names.csv")
        with open(names_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image_name", "Age"])
            for nm in names:
                w.writerow([nm, 55])
        ns2, y2 = read_labels([names_csv], names_only=True)
        assert ns2 == names and y2 is None
        try:
            read_labels([names_csv])
            raise AssertionError("无 True_Label 且未 names_only 应退出")
        except SystemExit:
            pass
    print("self-test PASS: 标签/对齐/npy顺序/AUROC/AUPRC/Spearman(tie)/blend/bootstrap/names-only/错误路径")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="合成数据自测（不需要真数据）")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("metrics", help="AUROC/AUPRC + 两两 Spearman")
    m.add_argument("--labels", nargs="+", required=True, help="标签 CSV（Image_name,True_Label）")
    m.add_argument("--names-only", action="store_true",
                   help="只对齐 Image_name/算 Spearman（test.csv 无 True_Label）")
    m.add_argument("--probs", nargs="+", required=True,
                   help="概率 .csv（按名对齐）或 .npy（按 labels 顺序）")
    m.set_defaults(func=cmd_metrics)
    b = sub.add_parser("blend", help="base vs new：w 扫描 + 配对 bootstrap")
    b.add_argument("--labels", nargs="+", required=True)
    b.add_argument("--base", required=True)
    b.add_argument("--new", required=True)
    b.add_argument("--w-step", type=float, default=0.02)
    b.add_argument("--w-max", type=float, default=0.5)
    b.add_argument("--modes", default="prob,logit")
    b.add_argument("--bootstrap", type=int, default=2000)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=cmd_blend)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.cmd:
        ap.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
