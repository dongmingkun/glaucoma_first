# -*- coding: utf-8 -*-
"""#127 OOF 干净 stacking / 权重搜索（CPU-only，numpy 依赖，无需 GPU）。

背景
----
现有 ens10（98.82）的权重是在 **valid** 上调的，而 train+all 成员看过 valid，valid 不是干净的选择集。
本脚本改用 **#109 的 3 折 held-out OOF（57245 行 / 1848 正例）** 做成员组合：
每个样本的 OOF 概率都由「没见过它」的模型产出，因此可在 OOF 上做 nested 3-fold CV 选权重/正则，
而不会泄漏到 valid。最终把选定权重/模型应用到成员的 test TTA4 概率，产出 `stack_test_tta4.csv`。

方法（三种，CV 选优）
----
1. `rank`   ：各成员 OOF 转经验秩 [0,1] 后等权平均（无拟合参数，最稳的基线）；
2. `weight` ：非负权重单纯形上坐标上升，最大化 OOF AUROC（只用训练折标签）；
3. `logreg` ：对 logit(成员概率) 做 L2 逻辑回归（二层 stacker），正则 λ 由内层 CV 选。

评估
----
- nested 3-fold CV：外层 held-out 折评估，内层（weight/logreg）只在其余折上拟合/选参；
- 报告 OOF AUROC/AUPRC、与 ens9 的配对 bootstrap（B 次重采样，P(Δ>0)）；
- 门槛（#127）：ΔAUROC vs ens9 ≥ +0.0003 且 P(Δ>0) ≥ 0.90 才算有希望；否则如实记录负结果。

用法
----
  # 自测（不需要任何数据文件，验证数学/流程；CPU 秒级）
  python tools/stack_oof.py --self-test

  # 真跑（#126 产出 3 折 OOF 后；成员 OOF/test 由 folds3 生成）
  python tools/stack_oof.py \
    --labels glaucoma_first/train.csv glaucoma_first/valid.csv \
    --folds folds3 --test-names glaucoma_first/test.csv \
    --base oof/oof_probs.npy::oof/test_probs_tta4.csv \
    --member cb384=oof/cb384_oof.npy::oof/cb384_test_tta4.csv \
    --member base384_8ep=oof/base384_8ep_oof.npy::oof/base384_8ep_test_tta4.csv \
    --member base512_4ep=oof/base512_4ep_oof.npy::oof/base512_4ep_test_tta4.csv \
    --member swin_s7=oof/swin_s7_oof.npy::oof/swin_s7_test_tta4.csv \
    --member swin_s123=oof/swin_s123_oof.npy::oof/swin_s123_test_tta4.csv \
    --ens9 ens9=oof/ens9_oof.npy::oof/ens9_test_tta4.csv \
    --out-prefix oof/stack --bootstrap 2000

  # 自动发现成员（oof/*_oof.npy + 同名 *_test_tta4.csv）
  python tools/stack_oof.py --labels ... --folds folds3 --test-names ... --autodiscover
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------- 指标
def _rankdata(a):
    """tie-aware 平均秩（1-based）。"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def rank01(a):
    """经验秩归一化到 (0,1)。"""
    r = _rankdata(a)
    return (r - 0.5) / len(r)


def auc_roc(y, s):
    """ROC-AUC（Mann-Whitney U，tie-aware）。"""
    y = np.asarray(y).astype(np.int8)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_pr(y, s):
    """AUPRC（average precision，按分数降序的阶梯积分）。"""
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0)
    total_pos = max(y.sum(), 1.0)
    recall = tp / total_pos
    ap = 0.0
    prev_r = 0.0
    for i in range(len(y)):
        ap += precision[i] * (recall[i] - prev_r)
        prev_r = recall[i]
    return float(ap)


def paired_bootstrap(y, s_hi, s_lo, B=1000, seed=42):
    """配对 bootstrap：ΔAUROC = AUC(s_hi) - AUC(s_lo)。"""
    y = np.asarray(y).astype(np.int8)
    n = len(y)
    base = auc_roc(y, s_hi) - auc_roc(y, s_lo)
    rng = np.random.default_rng(seed)
    d = np.empty(B, dtype=np.float64)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    for b in range(B):
        ip = rng.integers(0, n1, n1)
        ineg = rng.integers(0, n0, n0)
        j = np.concatenate([pos_idx[ip], neg_idx[ineg]])
        yy = y[j]
        d[b] = auc_roc(yy, s_hi[j]) - auc_roc(yy, s_lo[j])
    return {
        "delta": float(base),
        "boot_mean": float(d.mean()),
        "p_gt0": float((d > 0).mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "B": int(B),
    }


# ----------------------------------------------------------------------------- 特征
def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _standardize(fit_X, X):
    mu = fit_X.mean(axis=0)
    sd = fit_X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


# ----------------------------------------------------------------------------- 方法
def fit_weights(X_rank, y, iters=60, seed=0):
    """非负权重单纯形上坐标上升，最大化 AUROC。X_rank: (n,m) 秩特征。"""
    m = X_rank.shape[1]
    w = np.full(m, 1.0 / m)
    best = auc_roc(y, X_rank @ w)
    step = 0.25
    for _ in range(iters):
        improved = False
        for j in range(m):
            for delta in (step, -step):
                w2 = w.copy()
                w2[j] = max(0.0, w2[j] + delta)
                s = w2.sum()
                if s <= 0:
                    continue
                w2 /= s
                a = auc_roc(y, X_rank @ w2)
                if a > best + 1e-9:
                    best, w, improved = a, w2, True
        if not improved:
            step *= 0.5
        if step < 1e-3:
            break
    return w, best


def fit_logreg(X, y, lam=1.0, iters=400, lr=0.5):
    """L2 逻辑回归（梯度下降，类别加权抗不平衡）。X 已标准化。"""
    n, m = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(m + 1)
    pos = max(int(y.sum()), 1)
    neg = max(n - pos, 1)
    sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (sw * (p - y)) / n
        g[:-1] += lam * w[:-1] / n
        w -= lr * g
    return w


def pred_logreg(X, w):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    z = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- 嵌套 CV
def nested_cv(P_oof, R_oof, y, folds, method, lambdas=(0.03, 0.1, 0.3, 1.0, 3.0)):
    """外层 3 折；返回交叉验证后的 stack 概率（诚实 OOF）。"""
    n = len(y)
    pred = np.zeros(n, dtype=np.float64)
    chosen = []
    for k in sorted(set(folds.tolist())):
        tr = folds != k
        va = folds == k
        if tr.sum() < 50 or va.sum() < 10:
            continue
        if method == "rank":
            pred[va] = R_oof[va].mean(axis=1)
        elif method == "weight":
            w, _ = fit_weights(R_oof[tr], y[tr])
            pred[va] = R_oof[va] @ w
            chosen.append({"fold": int(k), "weights": w.tolist()})
        elif method == "logreg":
            # 内层：在 tr 内按 fold 再切一刀选 λ
            inner_ids = sorted(set(folds[tr].tolist()))
            best_lam, best_auc = lambdas[0], -1.0
            if len(inner_ids) >= 2:
                ik = inner_ids[0]
                itr = tr & (folds != ik)
                iva = tr & (folds == ik)
                Xtr, mu, sd = _standardize(logit(P_oof[itr]), logit(P_oof[itr]))
                for lam in lambdas:
                    w = fit_logreg(Xtr, y[itr], lam=lam)
                    a = auc_roc(y[iva], pred_logreg(_standardize(logit(P_oof[itr]), logit(P_oof[iva]))[0], w))
                    if a > best_auc:
                        best_auc, best_lam = a, lam
            Xtr, mu, sd = _standardize(logit(P_oof[tr]), logit(P_oof[tr]))
            w = fit_logreg(Xtr, y[tr], lam=best_lam)
            Xva, _, _ = _standardize(logit(P_oof[tr]), logit(P_oof[va]))
            pred[va] = pred_logreg(Xva, w)
            chosen.append({"fold": int(k), "lambda": float(best_lam)})
    return pred, chosen


def final_fit(P_oof, R_oof, y, method, lambdas=(0.03, 0.1, 0.3, 1.0, 3.0)):
    """全量 OOF 上拟合最终模型（返回应用到 test 的 callable + 参数）。"""
    if method == "rank":
        return (lambda P_test: R_test_mean(P_test)), {"method": "rank"}
    if method == "weight":
        w, a = fit_weights(R_oof, y)
        return (lambda P_test: np.column_stack([rank01(c) for c in P_test.T]) @ w), {
            "method": "weight", "weights": w.tolist(), "train_auc": a}
    # logreg：全量上以 3 折 CV 选 λ
    best_lam, best_auc = lambdas[0], -1.0
    for lam in lambdas:
        # 简单 3 折（按索引分块）评 λ
        idx = np.arange(len(y))
        aucs = []
        for k in range(3):
            va = idx % 3 == k
            tr = ~va
            Xtr, _, _ = _standardize(logit(P_oof[tr]), logit(P_oof[tr]))
            w = fit_logreg(Xtr, y[tr], lam=lam)
            Xva, _, _ = _standardize(logit(P_oof[tr]), logit(P_oof[va]))
            aucs.append(auc_roc(y[va], pred_logreg(Xva, w)))
        a = float(np.nanmean(aucs))
        if a > best_auc:
            best_auc, best_lam = a, lam
    Xall, mu, sd = _standardize(logit(P_oof), logit(P_oof))
    w = fit_logreg(Xall, y, lam=best_lam)

    def apply_logreg(P_test):
        return pred_logreg((logit(P_test) - mu) / sd, w)
    return apply_logreg, {"method": "logreg", "lambda": float(best_lam),
                          "coef": w[:-1].tolist(), "intercept": float(w[-1]),
                          "cv_auc": best_auc}


def R_test_mean(P_test):
    return np.column_stack([rank01(c) for c in P_test.T]).mean(axis=1)


# ----------------------------------------------------------------------------- IO
def read_label_csv(paths):
    names, y = [], []
    for p in paths:
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                names.append(r["Image_name"])
                y.append(int(float(r["True_Label"])))
    return names, np.asarray(y, dtype=np.int8)


def read_names(path):
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(r["Image_name"])
    return out


def read_prob_csv(path, names=None):
    d = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d[r["Image_name"]] = float(r["Probability"])
    if names is None:
        return np.asarray(list(d.values()), dtype=np.float64)
    missing = [n for n in names if n not in d]
    if missing:
        raise SystemExit(f"{path} 缺少 {len(missing)} 个 Image_name（示例 {missing[:3]}）")
    return np.asarray([d[n] for n in names], dtype=np.float64)


def load_vec(spec, names):
    """spec 可为 .npy（已按官方顺序）或 .csv（按 Image_name 对齐）。"""
    if spec.endswith(".npy"):
        v = np.load(spec).astype(np.float64).ravel()
        if names is not None and len(v) != len(names):
            raise SystemExit(f"{spec} 长度 {len(v)} != 期望 {len(names)}")
        return v
    return read_prob_csv(spec, names)


def parse_member(tok, names, test_names):
    """NAME=OOF::TEST（test 可省略；用 ``::`` 分隔以兼容 Windows 盘符 ``C:``）。也支持裸路径（自动取名）。"""
    if "=" in tok:
        name, rest = tok.split("=", 1)
    else:
        name, rest = None, tok
    if "::" in rest:
        oof, test = rest.split("::", 1)
    else:
        oof, test = rest, None
    if name is None:
        name = os.path.basename(oof).replace("_oof.npy", "").replace(".npy", "")
    return dict(name=name, oof=load_vec(oof, names),
                test=(load_vec(test, test_names) if test else None), oof_path=oof, test_path=test)


def load_folds(folds_dir, names):
    idx = {n: i for i, n in enumerate(names)}
    fold = np.full(len(names), -1, dtype=np.int16)
    k = 0
    while True:
        p = os.path.join(folds_dir, f"fold{k}_valid.csv")
        if not os.path.exists(p):
            break
        for n in read_names(p):
            if n in idx:
                fold[idx[n]] = k
        k += 1
    if k == 0:
        raise SystemExit(f"{folds_dir} 下未找到 fold*_valid.csv")
    bad = int((fold < 0).sum())
    if bad:
        raise SystemExit(f"有 {bad} 个样本未分配到任何 fold（folds 未覆盖全量）")
    return fold


def write_test_csv(path, names, probs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, probs):
            w.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])


# ----------------------------------------------------------------------------- 自测
def self_test():
    rng = np.random.default_rng(0)
    n, npos = 57245, 1848
    y = np.zeros(n, dtype=np.int8)
    y[:npos] = 1
    rng.shuffle(y)
    folds = np.zeros(n, dtype=np.int16)
    # 分层分折
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    for i, p in enumerate(pos):
        folds[p] = i % 3
    for i, p in enumerate(neg):
        folds[p] = i % 3

    def make_member(signal, noise):
        z = signal * y + rng.normal(0, noise, n)
        p = 1.0 / (1.0 + np.exp(-(z - np.log(3.0))))  # base rate ~ 1/4
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return p

    P = np.column_stack([
        make_member(1.2, 2.0),
        make_member(1.0, 2.0),
        make_member(1.4, 2.2),
        make_member(0.9, 2.5),
    ])
    R = np.column_stack([rank01(c) for c in P.T])
    ens9 = P[:, 0].copy()  # 用成员0当 ens9 基线

    print("[self-test] 单成员 AUROC:", [round(auc_roc(y, P[:, j]), 4) for j in range(P.shape[1])])
    results = {}
    for m in ("rank", "weight", "logreg"):
        cv_pred, chosen = nested_cv(P, R, y, folds, m)
        a = auc_roc(y, cv_pred)
        results[m] = a
        print(f"[self-test] {m:7s} CV-OOF AUROC={a:.4f} AUPRC={auc_pr(y, cv_pred):.4f}")
    best_m = max(results, key=results.get)
    apply_fn, params = final_fit(P, R, y, best_m)
    P_test = P[:500].copy()
    test_pred = apply_fn(P_test)
    boot = paired_bootstrap(y, cv_pred, ens9, B=200)
    print(f"[self-test] best={best_m} test_pred_len={len(test_pred)} "
          f"bootstrap Δ={boot['delta']:+.5f} P(Δ>0)={boot['p_gt0']:.3f}")
    # 断言
    assert abs(auc_roc(y, y.astype(float)) - 1.0) < 1e-12, "AUC perfect != 1"
    assert auc_roc(y, rng.normal(size=n)) < 0.55, "AUC random too high"
    best_single = max(auc_roc(y, P[:, j]) for j in range(P.shape[1]))
    assert results[best_m] >= best_single - 0.002, "stack 明显差于最佳单成员"
    assert len(test_pred) == 500 and np.isfinite(test_pred).all(), "test 预测异常"
    assert 0.0 <= boot["p_gt0"] <= 1.0
    assert params["method"] in ("rank", "weight", "logreg")
    print("[self-test] OK ✔ 所有断言通过")
    return 0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="#127 OOF 干净 stacking / 权重搜索（CPU）")
    ap.add_argument("--self-test", action="store_true", help="用合成数据验证数学/流程（不需要数据文件）")
    ap.add_argument("--labels", nargs="+", default=None, help="train.csv valid.csv（官方顺序）")
    ap.add_argument("--folds", default="folds3", help="含 fold{k}_valid.csv 的目录")
    ap.add_argument("--test-names", default="glaucoma_first/test.csv", help="test 名称顺序")
    ap.add_argument("--base", default=None, help="#126 基础：OOF.npy:test.csv")
    ap.add_argument("--member", action="append", default=[], help="NAME=OOF::TEST（可重复）")
    ap.add_argument("--ens9", default=None, help="基线 NAME=OOF::TEST（配对 bootstrap 用）")
    ap.add_argument("--autodiscover", action="store_true", help="自动发现 oof/*_oof.npy")
    ap.add_argument("--out-prefix", default="oof/stack")
    ap.add_argument("--methods", default="rank,weight,logreg")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None, help="可选：写 docs/ENSEMBLE_OOF_RESULTS.md")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.labels:
        raise SystemExit("需要 --labels（或用 --self-test）")
    names, y = read_label_csv(args.labels)
    test_names = read_names(args.test_names)
    folds = load_folds(args.folds, names)
    print(f"labels: n={len(names)} pos={int(y.sum())} ({y.mean():.5f}); test={len(test_names)}; "
          f"folds={sorted(set(folds.tolist()))}")

    members = []
    if args.base:
        members.append(parse_member("base=" + args.base, names, test_names))
    for tok in args.member:
        members.append(parse_member(tok, names, test_names))
    if args.autodiscover:
        seen = {m["name"] for m in members}
        for fn in sorted(os.listdir("oof")):
            if fn.endswith("_oof.npy"):
                nm = fn[:-8]
                if nm in seen:
                    continue
                tp = os.path.join("oof", nm + "_test_tta4.csv")
                spec = f"{nm}=oof/{fn}" + (f"::{tp}" if os.path.exists(tp) else "")
                members.append(parse_member(spec, names, test_names))
    if not members:
        raise SystemExit("无成员：用 --base/--member/--autodiscover 提供至少 1 个")

    params = [m["name"] for m in members]
    P = np.column_stack([m["oof"] for m in members])
    R = np.column_stack([rank01(c) for c in P.T])
    print(f"成员({len(params)}): {params}")
    for j, m in enumerate(members):
        print(f"  {m['name']:16s} OOF-AUROC={auc_roc(y, P[:, j]):.4f} AUPRC={auc_pr(y, P[:, j]):.4f}")

    ens9_vec = None
    if args.ens9:
        e = parse_member(args.ens9, names, test_names)
        ens9_vec = e["oof"]
        print(f"ens9 基线: {e['name']} OOF-AUROC={auc_roc(y, ens9_vec):.4f}")

    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    cv_results, cv_preds, fit_params = {}, {}, {}
    for m in methods:
        if m not in ("rank", "weight", "logreg"):
            continue
        cv_pred, chosen = nested_cv(P, R, y, folds, m)
        cv_results[m] = {"auroc": auc_roc(y, cv_pred), "auprc": auc_pr(y, cv_pred), "chosen": chosen}
        cv_preds[m] = cv_pred
        _, fp = final_fit(P, R, y, m)
        fit_params[m] = fp
        print(f"[{m:7s}] CV-OOF AUROC={cv_results[m]['auroc']:.5f} AUPRC={cv_results[m]['auprc']:.5f}")

    best_m = max(cv_results, key=lambda k: cv_results[k]["auroc"])
    print(f"\n=== 选中方法: {best_m} (CV-OOF AUROC={cv_results[best_m]['auroc']:.5f}) ===")

    # 应用最终模型到 test
    apply_fn, fp = final_fit(P, R, y, best_m)
    need_test = [m for m in members if m["test"] is not None]
    if len(need_test) == len(members):
        P_test = np.column_stack([m["test"] for m in members])
        stack_test = apply_fn(P_test)
        out_csv = args.out_prefix + "_test_tta4.csv"
        write_test_csv(out_csv, test_names, stack_test)
        print(f"写出 {out_csv} ({len(stack_test)} 行, mean={stack_test.mean():.5f})")
    else:
        stack_test = None
        print("WARN: 有成员缺 test 概率，跳过 stack_test_tta4.csv")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    np.save(args.out_prefix + "_oof_probs.npy", cv_preds[best_m])
    print(f"写出 {args.out_prefix}_oof_probs.npy ({len(cv_preds[best_m])})")

    boot = None
    if ens9_vec is not None:
        boot = paired_bootstrap(y, cv_preds[best_m], ens9_vec, B=args.bootstrap, seed=args.seed)
        gate = "PASS" if (boot["delta"] >= 0.0003 and boot["p_gt0"] >= 0.90) else "FAIL(负结果)"
        print(f"vs ens9: ΔAUROC={boot['delta']:+.5f} P(Δ>0)={boot['p_gt0']:.3f} "
              f"CI95=[{boot['ci95'][0]:+.5f},{boot['ci95'][1]:+.5f}] -> 门槛 {gate}")

    payload = {
        "members": params, "selected": best_m, "cv_results": cv_results,
        "fit_params": fit_params, "bootstrap_vs_ens9": boot,
        "gate": ("pass" if boot and boot["delta"] >= 0.0003 and boot["p_gt0"] >= 0.90
                 else "fail" if boot else "n/a"),
        "n": int(len(y)), "pos": int(y.sum()),
    }
    with open(args.out_prefix + "_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"写出 {args.out_prefix}_weights.json")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        lines = ["# ENSEMBLE OOF 结果（#127）", "",
                 f"- 样本：{len(y)}（正例 {int(y.sum())}，比例 {y.mean():.5f}）",
                 f"- 成员：{', '.join(params)}",
                 f"- 选中方法：**{best_m}**", "",
                 "## CV-OOF 指标（nested 3-fold CV）", "",
                 "| 方法 | OOF AUROC | OOF AUPRC |", "|---|---|---|"]
        for m, r in cv_results.items():
            lines.append(f"| {m} | {r['auroc']:.5f} | {r['auprc']:.5f} |")
        if boot:
            lines += ["", "## 与 ens9 的配对 bootstrap", "",
                      f"- ΔAUROC = {boot['delta']:+.5f}，P(Δ>0) = {boot['p_gt0']:.3f}",
                      f"- 95% CI = [{boot['ci95'][0]:+.5f}, {boot['ci95'][1]:+.5f}]，B = {boot['B']}",
                      f"- 门槛（Δ≥+0.0003 且 P≥0.90）：**{payload['gate'].upper()}**"]
        lines += ["", "## 权重/参数", "", "```json",
                  json.dumps(fit_params[best_m], ensure_ascii=False, indent=2), "```"]
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"写出 {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
