# Issue #126 结果记录 — 3-fold ConvNeXt-Small@384 16ep 真机执行（OOF 57245 + test 5855）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/126 （#109 方案 A 的云端实跑）。
> 本文件为**预交付模板**：真机跑完 `tools/pipeline_109.sh` 后按各节命令回填数字，
> 未填处标 `待填`。指标工具：`tools/oof_report.py`（CPU，numpy/sklearn）。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `CACHE_SRC=cache512 BS=8 ACCUM=6 bash tools/pipeline_109.sh` |
| 机器 | 待填（GPU 型号 / 显存 / CUDA） |
| 耗时 | 待填（每折起止与总时长） |
| 缓存 | `cache512 train=51396 valid=5849 test=5855`；`cache_all` 57245（脚本自动拼） |
| 口径 | 每折 ImageNet 初始化；OOF/test 一律 `last.pt` + 固定 `--use ema`（禁止 best/auto） |

## 每折 held-out valid TTA4（last.pt + EMA）

```bash
for k in 0 1 2; do
  python tools/oof_report.py metrics --labels folds3/fold${k}_valid.csv --probs oof/cnxs_fold${k}_valid_tta4.csv
done
```

| fold | valid 行数 | 正例 | AUROC | AUPRC | 耗时 |
|---|---|---|---|---|---|
| 0 | 19082 | 616 | 待填 | 待填 | 待填 |
| 1 | 19082 | 待填 | 待填 | 待填 | 待填 |
| 2 | 19082 | 待填 | 待填 | 待填 | 待填 |

## OOF 57245 整体（#127 stacking 的基础成员）

```bash
python tools/oof_report.py metrics \
  --labels glaucoma_first/train.csv glaucoma_first/valid.csv --probs oof/oof_probs.npy
```

| 指标 | 值 |
|---|---|
| OOF AUROC | 待填 |
| OOF AUPRC | 待填 |
| 覆盖 | 57245 / 57245（脚本已硬校验，缺一即非零退出） |

test 3 折平均：`oof/test_probs_tta4.csv`，5855 行，test_mean = 待填。

## 3 折两两 Spearman（验收 <0.98）

```bash
python tools/oof_report.py metrics --names-only --labels glaucoma_first/test.csv \
  --probs oof/cnxs_fold0_test_tta4.csv oof/cnxs_fold1_test_tta4.csv oof/cnxs_fold2_test_tta4.csv
```

| 对 | Spearman |
|---|---|
| fold0 × fold1 | 待填 |
| fold0 × fold2 | 待填 |
| fold1 × fold2 | 待填 |

## 与 ens9 的配对 bootstrap（OOF 在官方 valid 5849 子集上）

先把 OOF 的 valid 段切成 CSV（`oof_probs.npy` 按官方 train+valid 顺序，valid 为后 5849 行）：

```bash
python - <<'EOF'
import csv, numpy as np
names = [r['Image_name'] for r in csv.DictReader(open('glaucoma_first/valid.csv', encoding='utf-8-sig'))]
oof = np.load('oof/oof_probs.npy'); assert len(oof) == 57245
p = oof[-len(names):]
with open('oof/oof_valid_subset_tta4.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['Image_name', 'Probability', 'Pred_Label'])
    for n, v in zip(names, p): w.writerow([n, f'{v:.6f}', int(v >= 0.5)])
print('wrote oof/oof_valid_subset_tta4.csv', len(names))
EOF
```

```bash
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/oof_valid_subset_tta4.csv
```
（`ens9_valid_probs_tta4.csv` 从 Release/群文件下载后放仓库根目录；若拿不到，本节如实标「ens9 valid 概率未拿到」。）

| 项 | 值 |
|---|---|
| OOF(valid 段) AUROC / AUPRC | 待填 |
| ens9 AUROC / AUPRC | 待填 |
| Spearman(OOF, ens9) | 待填 |
| blend best w / Δ / P(Δ>0) | 待填 |

> best w 是在同批 valid 上选的，bootstrap P 偏乐观；#127 决策用 nested CV 或预指定 w。

## 验收（对照 #126）

- [ ] OOF 覆盖 57245、无 train/valid 重叠（脚本硬校验）；
- [ ] test 5855 行、顺序与官方 CSV 一致、无 NaN（脚本硬校验）；
- [ ] 3 折两两 Spearman <0.98；
- [ ] OOF 用 `last.pt` + 固定 `--use ema`（本 run 的 pipeline 已固定，历史 auto 产物作废）；
- [ ] 某折失败时脚本非零退出、未零填 OOF。

## 交付物

- `runs/cnxs_fold{0,1,2}/last.pt` — 位置：待填（Release 链接 / 群文件）；
- `oof/cnxs_fold{k}_valid_tta4.csv` × 3、`oof/cnxs_fold{k}_test_tta4.csv` × 3；
- `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855，3 折平均）；
- 本文件回填后的完整数字。

## 下游

- #127（干净 stacking）：`oof/oof_probs.npy` + `oof/test_probs_tta4.csv` 作为 base 成员，见 `docs/ISSUE_127_RUNBOOK.md`；
- #111（蒸馏）：3 个 `last.pt` 作为教师；建议 `tools/gen_soft_labels.py` 也补固定 EMA 口径（当前 auto 同样按该折 AUROC 选权重，另行小改动，不在本单范围）。
