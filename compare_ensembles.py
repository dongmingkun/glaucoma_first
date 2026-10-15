"""Compare ensemble variants on the valid split using saved TTA4 probabilities."""
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

RUNS = {
    "384": r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont",
    "448": r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448",
    "512": r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512",
    "cx": r"D:\glaucoma_work\runs\convnext_tiny_384_1015_011832_cnxt_384",
    "cx448": r"D:\glaucoma_work\runs\convnext_tiny_448_1015_045435_cnxt_448",
    "cs": r"D:\glaucoma_work\runs\convnext_small_384_1015_060006_cnxs_384",
    "cb": r"D:/glaucoma_work/runs/convnext_base_384_1015_113411_cnxb_384",
    "hi": r"D:/glaucoma_work/runs/convnext_small_512_1015_170341_cs_hi512",
    "hi2": r"D:/glaucoma_work/runs/convnext_small_512_1015_191923_cs_hi512_v2",
    "cb512": r"D:/glaucoma_work/runs/convnext_base_512_1016_005020_cb_hi512b",
}
VALID_CSV = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first\valid.csv"

y = np.array([int(r["True_Label"]) for r in csv.DictReader(open(VALID_CSV, encoding="utf-8-sig"))])
probs = {}
for k, d in RUNS.items():
    p = os.path.join(d, "probs_valid_full_tta4.npy")
    probs[k] = np.load(p)
    print(f"{k}: n={len(probs[k])} AUROC={roc_auc_score(y, probs[k]):.5f} AUPRC={average_precision_score(y, probs[k]):.5f}")


def balacc(p, thr=0.5):
    pred = p >= thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return 0.5 * (sens + spec), sens, spec


def max_balacc(p):
    o = np.argsort(-p); ys, ss = y[o], p[o]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx]; fps = np.cumsum(1 - ys)[idx]
    P, N = ys.sum(), len(ys) - ys.sum()
    score = tps / P + (1 - fps / N)
    k = int(np.argmax(score))
    return score[k] / 2, float(ss[idx][k])


def report(name, p):
    a, ap = roc_auc_score(y, p), average_precision_score(y, p)
    b0, s0, sp0 = balacc(p)
    bmax, tmax = max_balacc(p)
    print(f"{name:26s} AUROC={a:.5f} AUPRC={ap:.5f} blend={0.5*(a+ap):.5f} "
          f"balACC@0.5={b0:.5f} maxbalACC={bmax:.5f}(@raw {tmax:.4f})")


def rank_avg(ps):
    out = np.zeros_like(ps[0])
    for p in ps:
        out += np.argsort(np.argsort(p)) / len(p)
    return out / len(ps)


def logit_mean(ps):
    z = [np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))) for p in ps]
    z = sum(z) / len(z)
    return 1 / (1 + np.exp(-z))


print()
for name, ps in [("single 384", [probs["384"]]), ("single 448", [probs["448"]]), ("single 512", [probs["512"]]),
                 ("single convnext", [probs["cx"]])]:
    report(name, ps[0])

print()
report("ens 384+448 (prob)", (probs["384"] + probs["448"]) / 2)
report("ens 384+448+512 (prob)", (probs["384"] + probs["448"] + probs["512"]) / 3)
report("ens 384+448+512 (rank)", rank_avg([probs["384"], probs["448"], probs["512"]]))
report("ens 384+448 (rank)", rank_avg([probs["384"], probs["448"]]))
report("ens 384+512 (prob)", (probs["384"] + probs["512"]) / 2)
report("ens 1.5*384+448+512", (1.5 * probs["384"] + probs["448"] + probs["512"]) / 3.5)
report("ens 1.5*384+448", (1.5 * probs["384"] + probs["448"]) / 2.5)
report("ens 384+1.5*448", (probs["384"] + 1.5 * probs["448"]) / 2.5)
report("ens 384+448+0.5*512", (probs["384"] + probs["448"] + 0.5 * probs["512"]) / 2.5)
report("ens logit 384+448", logit_mean([probs["384"], probs["448"]]))
report("ens logit 384+448+512", logit_mean([probs["384"], probs["448"], probs["512"]]))

print("\n--- 加入 ConvNeXt ---")
report("ens cx alone", probs["cx"])
report("ens cx+384", (probs["cx"] + probs["384"]) / 2)
report("ens cx+448", (probs["cx"] + probs["448"]) / 2)
report("ens cx+b3ens3", (probs["cx"] + (probs["384"] + probs["448"] + probs["512"]) / 3) / 2)
report("ens cx+b3ens3 (1:0.5)", (probs["cx"] + 0.5 * (probs["384"] + probs["448"] + probs["512"]) / 3) / 1.5)
report("ens cx+b3ens3 (0.6:0.4)", 0.6 * probs["cx"] + 0.4 * (probs["384"] + probs["448"] + probs["512"]) / 3)
report("ens cx+384+448", (probs["cx"] + probs["384"] + probs["448"]) / 3)
report("ens cx*2+384+448+512", (2 * probs["cx"] + probs["384"] + probs["448"] + probs["512"]) / 5)
report("ens logit cx+b3ens3", logit_mean([probs["cx"], probs["384"], probs["448"], probs["512"]]))
report("ens rank cx+384+448+512", rank_avg([probs["cx"], probs["384"], probs["448"], probs["512"]]))

print("\n--- 五模型（加入 ConvNeXt@448）---")
report("cx448 alone", probs["cx448"])
report("cx+cx448", (probs["cx"] + probs["cx448"]) / 2)
b3 = (probs["384"] + probs["448"] + probs["512"]) / 3
report("cx+cx448+b3ens3 (1:1:0.5)", (probs["cx"] + probs["cx448"] + 0.5 * b3) / 2.5)
report("cx+cx448+b3ens3 (1:0.6:0.4)", (probs["cx"] + 0.6 * probs["cx448"] + 0.4 * b3) / 2.0)
report("cx+cx448+b3ens3 (1:1:1)", (probs["cx"] + probs["cx448"] + b3) / 3)
report("cx*1.5+cx448+b3ens3(1:0.5)", (1.5 * probs["cx"] + probs["cx448"] + 0.5 * b3) / 3)
report("cx+cx448+384+448+512 (等权)", (probs["cx"] + probs["cx448"] + probs["384"] + probs["448"] + probs["512"]) / 5)
report("cx2+cx4481+b3 (2:1:1)", (2 * probs["cx"] + probs["cx448"] + b3) / 4)
report("logit cx+cx448+b3ens3", logit_mean([probs["cx"], probs["cx448"], probs["384"], probs["448"], probs["512"]]))

print("\n--- 六模型（加入 ConvNeXt-Small）---")
report("cs alone", probs["cs"])
report("cs+cx", (probs["cs"] + probs["cx"]) / 2)
report("cs+cx+b3ens3 (1:1:0.5)", (probs["cs"] + probs["cx"] + 0.5 * b3) / 2.5)
report("cs+cx+b3ens3 (1:0.7:0.5)", (probs["cs"] + 0.7 * probs["cx"] + 0.5 * b3) / 2.2)
report("cs*1.5+cx+b3ens3", (1.5 * probs["cs"] + probs["cx"] + 0.5 * b3) / 3)
report("cs+cx+b3ens3 (1:1:1)", (probs["cs"] + probs["cx"] + b3) / 3)
report("cs+cx+cx448+b3ens3 (1:1:1:0.5)", (probs["cs"] + probs["cx"] + probs["cx448"] + 0.5 * b3) / 3.5)
report("cs+cx+cx448+b3ens3 (1:1:1:1)", (probs["cs"] + probs["cx"] + probs["cx448"] + b3) / 4)
report("cs+ens4 (1:1)", (probs["cs"] + (probs["cx"] + 0.5 * b3) / 1.5) / 2)
report("cs+ens4 (1:0.8)", (probs["cs"] + 0.8 * (probs["cx"] + 0.5 * b3) / 1.5) / 1.8)
report("cs+ens4 (1:0.6)", (probs["cs"] + 0.6 * (probs["cx"] + 0.5 * b3) / 1.5) / 1.6)
report("cs+cx+b3 best3 only(384/448)", (probs["cs"] + probs["cx"] + probs["384"] + probs["448"]) / 4)
report("logit cs+cx+b3ens3", logit_mean([probs["cs"], probs["cx"], probs["384"], probs["448"], probs["512"]]))
report("rank cs+cx+b3ens3", rank_avg([probs["cs"], probs["cx"], probs["384"], probs["448"], probs["512"]]))

print(chr(10) + "--- 加入 ConvNeXt-Base ---")
report("cb alone", probs["cb"])
report("cb+cs", (probs["cb"] + probs["cs"]) / 2)
report("cb+cs+cx", (probs["cb"] + probs["cs"] + probs["cx"]) / 3)
cnx = (probs["cb"] + probs["cs"] + probs["cx"]) / 3
report("cnx3+b3ens3 (1:0.5)", (cnx + 0.5 * b3) / 1.5)
report("cnx3+b3ens3 (1:0.4)", (cnx + 0.4 * b3) / 1.4)
report("cnx3+b3ens3 (1:0.3)", (cnx + 0.3 * b3) / 1.3)
report("cb+cs+b3ens3 (1:1:0.5)", (probs["cb"] + probs["cs"] + 0.5 * b3) / 2.5)
report("cb*1.2+cs+cx+0.4*b3", (1.2*probs["cb"] + probs["cs"] + probs["cx"] + 0.4*b3) / 3.6)
report("ens6(cs1+cx0.4+b30.6) 对比", (probs["cs"] + 0.4*probs["cx"] + 0.6*b3) / 2.0)


print(chr(10) + "--- 高分辨率成员 hi512 ---")
report("hi alone", probs["hi"])
report("cb+hi+b3 (1:1:0.5)", (probs["cb"] + probs["hi"] + 0.5*b3) / 2.5)
report("cb+cs+hi+b3 (1:1:1:0.5)", (probs["cb"] + probs["cs"] + probs["hi"] + 0.5*b3) / 3.5)
report("cb+hi+0.5cs+b3 (1:1:0.5:0.5)", (probs["cb"] + probs["hi"] + 0.5*probs["cs"] + 0.5*b3) / 3.0)
report("hi+cs+b3 (1:1:0.5)", (probs["hi"] + probs["cs"] + 0.5*b3) / 2.5)
report("hi*1.5+cb+b3 (1.5:1:0.5)", (1.5*probs["hi"] + probs["cb"] + 0.5*b3) / 3.0)
report("对比: cb+cs+b3 (1:1:0.5)", (probs["cb"] + probs["cs"] + 0.5*b3) / 2.5)


print(chr(10) + "--- hi512_v2（续训版）---")
report("hi2 alone", probs["hi2"])
report("hi2+cb+b3 (1:1:0.5)", (probs["hi2"] + probs["cb"] + 0.5*b3) / 2.5)
report("hi2*1.5+cb+b3 (1.5:1:0.5)", (1.5*probs["hi2"] + probs["cb"] + 0.5*b3) / 3.0)
report("hi2+hi+cb+b3 (1:1:1:0.5)", (probs["hi2"] + probs["hi"] + probs["cb"] + 0.5*b3) / 3.5)
report("hi2*1.5+hi+cb+b3 (1.5:1:1:0.5)", (1.5*probs["hi2"] + probs["hi"] + probs["cb"] + 0.5*b3) / 4.0)
report("hi2*1.5+hi*0.5+cb+b3", (1.5*probs["hi2"] + 0.5*probs["hi"] + probs["cb"] + 0.5*b3) / 3.5)
report("hi2+cs+cb+b3 (1:1:1:0.5)", (probs["hi2"] + probs["cs"] + probs["cb"] + 0.5*b3) / 3.5)
report("hi2*2+cb+b3 (2:1:0.5)", (2*probs["hi2"] + probs["cb"] + 0.5*b3) / 3.5)
report("对比 ens6 配方(cs1+cx0.4+b30.6)", (probs["cs"] + 0.4*probs["cx"] + 0.6*b3) / 2.0)
report("对比 当前最优(hi1.5+cb+b3)", (1.5*probs["hi"] + probs["cb"] + 0.5*b3) / 3.0)


print(chr(10) + "--- Base@512 加入（当前 ens7 = hi2*2+cb+b3*0.5）---")
report("cb512 alone", probs["cb512"])
report("hi2+cb512+b3 (1:1:0.5)", (probs["hi2"] + probs["cb512"] + 0.5*b3) / 2.5)
report("hi2*2+cb512+b3 (2:1:0.5)", (2*probs["hi2"] + probs["cb512"] + 0.5*b3) / 3.5)
report("hi2*2+cb512*2+b3 (2:2:0.5)", (2*probs["hi2"] + 2*probs["cb512"] + 0.5*b3) / 4.5)
report("hi2*1.5+cb512*1.5+b3 (1.5:1.5:0.5)", (1.5*probs["hi2"] + 1.5*probs["cb512"] + 0.5*b3) / 3.5)
report("hi2*2+cb+cb512+b3 (2:1:1:0.5)", (2*probs["hi2"] + probs["cb"] + probs["cb512"] + 0.5*b3) / 4.5)
report("对照 ens7 = hi2*2+cb+b3 (2:1:0.5)", (2*probs["hi2"] + probs["cb"] + 0.5*b3) / 3.5)
report("cb512*2+cb+b3 (2:1:0.5)", (2*probs["cb512"] + probs["cb"] + 0.5*b3) / 3.5)
