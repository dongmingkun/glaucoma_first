"""ens8：多尺度 TTA 集成 + 校准（每个成员用自己的尺度集合，概率缓存可续跑）

相对 ens7 的改动：只提升推理侧（多尺度 TTA），不训练任何新权重。
- 512 输入的成员用 448/512/576；640 的用 576/640/704；其余用自身尺寸
- 每个成员的概率落盘（D:\\glaucoma_work\\ens8_probs），崩溃可续跑
- 输出：submit_ready/submission.csv（+ 报告 json）
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_submission import (load_model, predict_split, ranking_metrics, threshold_metrics,
                             optimal_shift, apply_shift)
from train import load_rows, CSV_ROOT

PROB_DIR = r"D:\glaucoma_work\ens8_probs"
OUT = r"C:\Users\Kevin\Desktop\glaucoma_first\submit_ready\submission.csv"

# (名称, 权重, ckpt, 尺度集合)
MEMBERS = [
    ("hi512_v2", 2.0, r"D:\glaucoma_work\runs\convnext_small_512_1015_191923_cs_hi512_v2\best.pt", [448, 512, 576]),
    ("cb512",    1.0, r"D:\glaucoma_work\runs\convnext_base_512_1016_005020_cb_hi512b\best.pt", [448, 512, 576]),
    ("cb384",    0.5, r"D:\glaucoma_work\runs\convnext_base_384_1015_113411_cnxb_384\best.pt", [352, 384, 448]),
    ("seed2",    1.0, r"D:\glaucoma_work\weights\convnext_small_512_cache768_1016_seed2_auprc0.8750.pt", [512]),  # 概率由队友提供（云端 4090D 训的多样性成员）
    ("hi640",    0.5, r"D:\glaucoma_work\runs\convnext_small_640_1016_130942_cs_hi640d\best.pt", [576, 640, 704]),
    ("b384", 0.5 / 3, r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont\best.pt", [384]),
    ("b448", 0.5 / 3, r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448\best.pt", [448]),
    ("b512", 0.5 / 3, r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512\best.pt", [512]),
]
FLIPS = [0, 1, 2, 3]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(PROB_DIR, exist_ok=True)
    val_rows = load_rows("valid"); test_rows = load_rows("test")
    y = np.array([r[2] for r in val_rows])
    V, T, W, names = [], [], [], []
    for name, w, ck, scales in MEMBERS:
        vp, tp = os.path.join(PROB_DIR, f"{name}_val.npy"), os.path.join(PROB_DIR, f"{name}_test.npy")
        if os.path.exists(vp) and os.path.exists(tp) and os.path.getsize(vp) > 1000:
            log(f"{name}: 概率已缓存，跳过")
        else:
            try:
                model, img_size, use, ckpt, cache_root = load_model(ck)
                pv = np.zeros(len(val_rows)); pt = np.zeros(len(test_rows))
                for s in scales:
                    pv += predict_split(model, "valid", val_rows, s, FLIPS, 2, bs=16, cache_root=cache_root)
                    pt += predict_split(model, "test", test_rows, s, FLIPS, 2, bs=16, cache_root=cache_root)
                pv /= len(scales); pt /= len(scales)
                np.save(vp, pv); np.save(tp, pt)
                log(f"{name}: 完成（scales={scales}, cache={os.path.basename(str(cache_root))}）")
                del model; torch.cuda.empty_cache()
            except Exception as e:
                log(f"!! {name} 失败: {type(e).__name__}: {str(e)[:120]}")
                continue
        V.append(np.load(vp)); T.append(np.load(tp)); W.append(w); names.append(name)

    W = np.array(W) / np.sum(W)
    ens_v = np.tensordot(W, np.stack(V), axes=1)
    ens_t = np.tensordot(W, np.stack(T), axes=1)
    rm, tm = ranking_metrics(y, ens_v), threshold_metrics(y, ens_v)
    blend = 0.5 * (rm["auroc"] + rm["auprc"])
    shift, t_star = optimal_shift(y, ens_v)
    ens_t_cal = apply_shift(ens_t, shift)
    tm_cal = threshold_metrics(y, apply_shift(ens_v, shift))
    comp = (0.5 * tm_cal["balacc"] + 0.25 * (rm["auroc"] + rm["auprc"])) * 100
    log(f"集成（多尺度）: AUROC={rm['auroc']:.5f} AUPRC={rm['auprc']:.5f} blend={blend*100:.2f} "
        f"校准后 balACC={tm_cal['balacc']:.5f} → 复合分 {comp:.3f}")

    names_out = []
    with open(os.path.join(CSV_ROOT, "test.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            names_out.append(r["Image_name"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names_out, ens_t_cal):
            wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
    rep = {"members": names, "weights": W.tolist(), "scales": {m[0]: m[3] for m in MEMBERS},
           "auroc": rm["auroc"], "auprc": rm["auprc"], "blend": blend,
           "balacc_cal": tm_cal["balacc"], "composite": comp,
           "shift": float(shift), "raw_t_star": float(t_star),
           "pos_rate": float((ens_t_cal >= 0.5).mean())}
    with open(os.path.join(os.path.dirname(OUT), "submission_ens8_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    log(f"已写出 {OUT}（阳性率 {rep['pos_rate']:.4f}）+ 报告 json")
    # 与 ens7 对比
    log("对照 ens7（单尺度）复合分 = 94.852（本地）／线上 94.81")


if __name__ == "__main__":
    main()