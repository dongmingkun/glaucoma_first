"""Stacking ensemble: learn how to combine member predictions (instead of plain averaging).

为什么：同族成员的简单平均已饱和（边际 +0.0003/成员）；stacking 能利用成员间的互补性
（例如 hi512 强 AUPRC、Base 强 AUROC、B3 提供多样性），且元学习器只用 7 个参数。

诚实评估：元学习器在 valid 上用 **5 折交叉验证**给出 out-of-fold 概率后再算指标；
对照的简单平均没有任何拟合参数，故直接在全 valid 上算，两者可比。

用法：
  python stack_ensemble.py --eval-only                 # 只用已缓存的概率做评估（秒级）
  python stack_ensemble.py --rebuild-probs             # 重新跑一遍成员推理（GPU ~35min）
  python stack_ensemble.py --submit --out submit_ready/submission.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_submission import (load_model, predict_split, ranking_metrics, threshold_metrics,
                             optimal_shift, apply_shift)
from train import load_rows, CSV_ROOT
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

PROB_DIR = r"D:\glaucoma_work\stack_probs"
MEMBERS = [
    ("hi512_v2", r"D:\glaucoma_work\runs\convnext_small_512_1015_191923_cs_hi512_v2\best.pt", 2.0),
    ("cb512", r"D:\glaucoma_work\runs\convnext_base_512_1016_005020_cb_hi512b\best.pt", 1.0),
    ("cb384", r"D:\glaucoma_work\runs\convnext_base_384_1015_113411_cnxb_384\best.pt", 0.5),
    ("b384", r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont\best.pt", 0.5 / 3),
    ("b448", r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448\best.pt", 0.5 / 3),
    ("b512", r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512\best.pt", 0.5 / 3),
]


def composite(auroc, auprc, maxbalacc):
    """线上复合口径：0.5*balACC@0.5(校准后) + 0.25*AUROC + 0.25*AUPRC（×100）"""
    return (0.5 * maxbalacc + 0.25 * (auroc + auprc)) * 100


def max_balacc(y, p):
    o = np.argsort(-p); ys, ss = y[o], p[o]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx]; fps = np.cumsum(1 - ys)[idx]
    P, N = ys.sum(), len(ys) - ys.sum()
    return float(np.max(tps / P + (1 - fps / N)) / 2)


def build_probs(workers=2, tta=4, scales=None):
    os.makedirs(PROB_DIR, exist_ok=True)
    val_rows = load_rows("valid"); test_rows = load_rows("test")
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[tta]
    for name, ck, _ in MEMBERS:
        vp, tp = os.path.join(PROB_DIR, f"{name}_val.npy"), os.path.join(PROB_DIR, f"{name}_test.npy")
        if os.path.exists(vp) and os.path.exists(tp):
            print(f"  {name}: 已缓存，跳过"); continue
        model, img_size, use, ckpt, cache_root = load_model(ck)
        sc = scales or [img_size]
        pv = np.zeros(len(val_rows)); pt = np.zeros(len(test_rows))
        for one in sc:
            pv += predict_split(model, "valid", val_rows, one, flips, workers, cache_root=cache_root)
            pt += predict_split(model, "test", test_rows, one, flips, workers, cache_root=cache_root)
        pv /= len(sc); pt /= len(sc)
        np.save(vp, pv)
        np.save(tp, pt)
        print(f"  {name}: val/test 概率已保存（cache={os.path.basename(str(cache_root))} img={sc}）", flush=True)
        del model; torch.cuda.empty_cache()


def load_members():
    names, V, T, W = [], [], [], []
    for name, ck, w in MEMBERS:
        vp = os.path.join(PROB_DIR, f"{name}_val.npy")
        tp = os.path.join(PROB_DIR, f"{name}_test.npy")
        if not (os.path.exists(vp) and os.path.exists(tp)):
            print(f"  !! {name} 概率缺失，先跑 --rebuild-probs"); continue
        names.append(name); V.append(np.load(vp)); T.append(np.load(tp)); W.append(w)
    W = np.array(W) / np.sum(W)
    return names, np.stack(V), np.stack(T), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-probs", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--out", default="submit_ready/submission.csv")
    ap.add_argument("--C", type=float, default=0.3, help="LogisticRegression 正则强度")
    args = ap.parse_args()

    if args.rebuild_probs:
        print("[1] 重新构建成员概率 …")
        build_probs(args.workers, args.tta)

    names, V, T, W = load_members()
    rows = load_rows("valid"); y = np.array([r[2] for r in rows])
    age = np.array([r[1] for r in rows], dtype=np.float64)
    print(f"\n成员: {names}\n权重(简单平均用): {np.round(W,3).tolist()}")

    # --- 基线：简单加权平均 ---
    avg = np.tensordot(W, V, axes=1)
    a, ap_ = roc_auc_score(y, avg), average_precision_score(y, avg)
    mb = max_balacc(y, avg)
    print(f"\n[基线] 简单加权平均: AUROC={a:.5f} AUPRC={ap_:.5f} maxbalACC={mb:.5f} → 复合分 {composite(a, ap_, mb):.3f}")

    # --- logit 特征（含 age） ---
    def feats(p, ag):
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))   # (n_members, n)
        return np.vstack([z, (ag - 58.0) / 15.0]).T

    X = feats(V, age)

    # --- 5 折 OOF 评估 ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(C=args.C, max_iter=2000, class_weight=None)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    a2, ap2 = roc_auc_score(y, oof), average_precision_score(y, oof)
    mb2 = max_balacc(y, oof)
    print(f"[Stacking] 5折OOF:      AUROC={a2:.5f} AUPRC={ap2:.5f} maxbalACC={mb2:.5f} → 复合分 {composite(a2, ap2, mb2):.3f}")
    print(f"[对比] 提升 = {composite(a2, ap2, mb2) - composite(a, ap_, mb):+.3f} 分（同期简单平均 {composite(a, ap_, mb):.3f}）")

    # --- 用全 valid 重拟合（提交用） ---
    lr = LogisticRegression(C=args.C, max_iter=2000).fit(X, y)
    coefs = dict(zip(names + ["age"], np.round(lr.coef_[0], 3).tolist()))
    print(f"[全量拟合] 系数: {coefs}")

    if args.submit:
        test_rows = load_rows("test")
        test_age = np.array([r[1] for r in test_rows], dtype=np.float64)
        Xt = feats(T, test_age)
        p_test = lr.predict_proba(Xt)[:, 1]
        # 校准：用 OOF 的 valid 概率求工作点，使 0.5 落在最优 balACC 处
        shift, t_star = optimal_shift(y, oof)
        p_test_cal = apply_shift(p_test, shift)
        names_out = []
        with open(os.path.join(CSV_ROOT, "test.csv"), encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                names_out.append(r["Image_name"])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["Image_name", "Probability", "Pred_Label"])
            for n, p in zip(names_out, p_test_cal):
                wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
        rep = {"members": names, "weights_simple": W.tolist(), "coefs": coefs,
               "simple": {"auroc": a, "auprc": ap_, "maxbalacc": mb, "composite": composite(a, ap_, mb)},
               "stacking_oof": {"auroc": a2, "auprc": ap2, "maxbalacc": mb2, "composite": composite(a2, ap2, mb2)},
               "shift": float(shift), "raw_t_star": float(t_star),
               "pos_rate": float((p_test_cal >= 0.5).mean())}
        with open(os.path.splitext(args.out)[0] + "_report.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"\n已写出 {args.out}（阳性率 {(p_test_cal>=0.5).mean():.4f}）+")


if __name__ == "__main__":
    main()