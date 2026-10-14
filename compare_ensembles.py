"""Compare ensemble variants on the valid split using saved TTA4 probabilities."""
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

RUNS = {
    "384": r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont",
    "448": r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448",
    "512": r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512",
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
for name, ps in [("single 384", [probs["384"]]), ("single 448", [probs["448"]]), ("single 512", [probs["512"]])]:
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