"""指标口径探针（issue #1 任务 1）：找出官方评分 ≈ 线上 89.2 的候选口径。

背景：线上 89.2 与本地 AUROC 98.5 / AUPRC 81.6 都对不上，却非常接近
valid 平衡准确率 @0.5（88.83）。本脚本在 valid 上把所有候选评分口径
（排序类 + 阈值类，各在默认 0.5 与各自最优工作点下）都算一遍，
并打印每个口径 ×100 后与线上分的差距，帮助锁定评分口径。

用法：
  python analysis_metric_probe.py --probs-valid probs_valid_full_tta4.npy \
      --valid-csv glaucoma_first/valid.csv --lb 89.2

依赖：numpy、scikit-learn（仅 AUROC/AUPRC）。
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_gt(valid_csv):
    with open(valid_csv, encoding="utf-8-sig") as f:
        return np.array([int(r["True_Label"]) for r in csv.DictReader(f)])


def thr_metrics(y, s, thr):
    """在固定阈值 thr 下计算一组分类指标。"""
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    P, N = tp + fn, fp + tn
    sens = tp / P if P else 0.0
    spec = tn / N if N else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (P + N) if (P + N) else 0.0
    bal = 0.5 * (sens + spec)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    gmean = float(np.sqrt(max(sens * spec, 0.0)))
    youden = sens + spec - 1.0
    return {
        "thr": thr,
        "pos_rate": float((pred == 1).mean()),
        "accuracy": acc,
        "balanced_acc": bal,
        "f1": f1,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "gmean": gmean,
        "youden": youden,
    }


def best_by(y, s, key, candidates):
    """在 ROC 断点阈值上，按 key 指标取最优阈值，返回该阈值下的全套指标。"""
    order = np.argsort(-s)
    ys, ss = y[order], s[order]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    best_thr, best_val, best_m = None, -np.inf, None
    for k in idx:
        t = float(ss[k])
        m = thr_metrics(ys, ss, t)
        if m[key] > best_val:
            best_val, best_thr, best_m = m[key], t, m
    best_m["thr"] = best_thr
    return best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-valid", required=True)
    ap.add_argument("--valid-csv", default=os.path.join(ROOT, "glaucoma_first", "valid.csv"))
    ap.add_argument("--lb", type=float, default=89.2, help="线上分数，用于对比差距")
    args = ap.parse_args()

    s = np.load(args.probs_valid)
    y = load_gt(args.valid_csv)
    assert len(s) == len(y), f"长度不一致: probs {len(s)} vs label {len(y)}"

    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))

    # 阈值类：默认 0.5 与各自最优工作点
    m_05 = thr_metrics(y, s, 0.5)
    m_bal = best_by(y, s, "balanced_acc", None)   # Youden 最优点（sens+spec 最大）
    m_f1 = best_by(y, s, "f1", None)
    m_gm = best_by(y, s, "gmean", None)

    print("=" * 72)
    print(f"valid 集：N={len(y)}  阳性={int(y.sum())} ({y.mean()*100:.2f}%)")
    print(f"排序类：AUROC={auroc*100:.2f}  AUPRC={auprc*100:.2f}  "
          f"混合(0.5*AUROC+0.5*AUPRC)={0.5*(auroc+auprc)*100:.2f}")
    print("=" * 72)

    rows = [
        ("AUROC (排序)", auroc * 100),
        ("AUPRC (排序)", auprc * 100),
        ("混合 0.5*(AUROC+AUPRC)", 0.5 * (auroc + auprc) * 100),
        ("accuracy @0.5", m_05["accuracy"] * 100),
        ("balanced_acc @0.5", m_05["balanced_acc"] * 100),
        ("F1 @0.5", m_05["f1"] * 100),
        ("gmean @0.5", m_05["gmean"] * 100),
        ("balanced_acc @Youden最优点", m_bal["balanced_acc"] * 100),
        ("F1 @F1最优点", m_f1["f1"] * 100),
        ("gmean @gmean最优点", m_gm["gmean"] * 100),
    ]
    print(f"{'候选口径':<28}{'valid×100':>12}{'与线上差距':>12}")
    print("-" * 56)
    for name, v in rows:
        gap = v - args.lb
        flag = "  <== 最接近" if abs(gap) <= 0.5 else ""
        print(f"{name:<28}{v:>12.2f}{gap:>+12.2f}{flag}")

    print("-" * 56)
    print(f"注：线上分参考值 {args.lb}。差距 ≤0.5 的候选最可能是官方口径。")
    print("     排序类（AUROC/AUPRC/混合）对校准不变；阈值类对校准敏感。")
    print()
    print("阈值细节：")
    print(f"  @0.5:            pos_rate={m_05['pos_rate']*100:.2f}%  "
          f"sens={m_05['sensitivity']:.3f} spec={m_05['specificity']:.3f}")
    print(f"  @Youden最优点:   thr={m_bal['thr']:.5f}  pos_rate={m_bal['pos_rate']*100:.2f}%  "
          f"sens={m_bal['sensitivity']:.3f} spec={m_bal['specificity']:.3f}")
    print(f"  @F1最优点:       thr={m_f1['thr']:.5f}  pos_rate={m_f1['pos_rate']*100:.2f}%")
    print(f"  @gmean最优点:    thr={m_gm['thr']:.5f}  pos_rate={m_gm['pos_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
