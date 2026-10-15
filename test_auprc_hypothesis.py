"""Test the hypothesis: leaderboard score = AUPRC on the test set (higher prevalence than valid).

Method: reweight positives to simulate a prevalence change p -> p', keeping the model's
per-class score distributions fixed (valid assumes no covariate shift; splits look identical).

  AP(p') = sum_k  [ w*TP_k / (w*TP_k + FP_k) ] * dR_k ,  w = [p'/(1-p')] / [p/(1-p)]

Find p' that reproduces the observed scores, then check whether the SAME p' explains both
submissions (89.20 for model A, 94.07 for model B).
"""
import csv
import json
import os

import numpy as np

ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first"
y = np.array([int(r["True_Label"]) for r in csv.DictReader(open(
    os.path.join(ROOT, "glaucoma_first", "valid.csv"), encoding="utf-8-sig"))])
P, N = y.sum(), len(y) - y.sum()
p_valid = P / len(y)

# model A: first submission = single B3@384, TTA4, uncalibrated
A = np.load(r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont\probs_valid_full_tta4.npy")
# model B: second submission = ens4 (4-model ensemble), raw (pre-calibration).
# We have ens4's *calibrated* valid probs? No - we have the raw components; rebuild the same weights:
b384 = A
b448 = np.load(r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448\probs_valid_full_tta4.npy")
b512 = np.load(r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512\probs_valid_full_tta4.npy")
cx = np.load(r"D:\glaucoma_work\runs\convnext_tiny_384_1015_011832_cnxt_384\probs_valid_full_tta4.npy")
B = (cx + 0.5 * (b384 + b448 + b512) / 3) / 1.5   # ens4 raw = cx 0.667 + b3trio 0.333


def ap_at_prevalence(scores, y, p_target):
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    w = (p_target / (1 - p_target)) / (p_valid / (1 - p_valid))
    tp_w = w * tp
    prec = tp_w / (tp_w + fp)
    recall = tp / P                      # unchanged by reweighting
    dR = np.diff(np.r_[0.0, recall])
    return float((prec * dR).sum())


obs = {"A (single b3, 89.20)": (A, 0.8920), "B (ens4, 94.07)": (B, 0.9407)}

print(f"valid prevalence = {p_valid:.4f}")
print("\nAUPRC as a function of hypothetical test prevalence:")
print(f"{'p_test':>8} " + " ".join(f"{k:>22}" for k in obs))
for p_t in (0.0327, 0.05, 0.06, 0.07, 0.08, 0.10):
    row = [f"{p_t:8.4f}"]
    for k, (s, o) in obs.items():
        row.append(f"{ap_at_prevalence(s, y, p_t):22.4f}")
    print(" ".join(row))

print("\nSolve for the prevalence that reproduces each observed score (bisection):")
solved = {}
for k, (s, o) in obs.items():
    lo, hi = 0.0327, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if ap_at_prevalence(s, y, mid) < o:
            lo = mid
        else:
            hi = mid
    solved[k] = (lo + hi) / 2
    print(f"  {k}: p_test = {solved[k]:.4f}  (AUPRC@{solved[k]:.4f} = {ap_at_prevalence(s, y, solved[k]):.4f})")

pA, pB = solved["A (single b3, 89.20)"], solved["B (ens4, 94.07)"]
print(f"\n一致性检验: p_test(A)={pA:.4f} vs p_test(B)={pB:.4f}  差 {abs(pA-pB):.4f}")
print("若两值接近 => 『榜单 = AUPRC，测试集阳性率更高』假设成立")

# Cross-check: use p from A to predict B's score (and vice versa)
predB = ap_at_prevalence(B, y, pA)
predA = ap_at_prevalence(A, y, pB)
print(f"\n用 A 解出的 p={pA:.4f} 预测 B 的分数: {predB*100:.2f} (实测 94.07, 差 {abs(predB*100-94.07):.2f})")
print(f"用 B 解出的 p={pB:.4f} 预测 A 的分数: {predA*100:.2f} (实测 89.20, 差 {abs(predA*100-89.20):.2f})")