# Issue #134 结果记录 — RA Base@512 16ep × 3-fold clean OOF（#128 最强新成员纳入 #127 99 门槛）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/134 。本文件为**预交付模板**：
> 云端 24GB+ 跑完 `CACHE_SRC=cache768 bash tools/pipeline_134.sh` 后回填，未填处标 `待填`。
> 指标工具：`tools/oof_report.py`（CPU，含 `--names-only`）。
> **红线**：OOF/test 一律 `last.pt` + 固定 `--use ema`；禁止 best.pt / auto / train-on-all。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `CACHE_SRC=cache768 bash tools/pipeline_134.sh` |
| 机器 / 耗时 | 待填（工单估 4090 24GB 约 8–9h：3×~2.5h + OOF） |
| 缓存 | `cache768 51396/5849/5855` → `cache_all768 57245/57245/5855`（符号链接，脚本硬校验） |
| 配方 | ConvNeXt-Base@512 16ep bs8×accum6 + RandAugment 2 9，seed 7，每折 ImageNet 初始化 |

## 每折 held-out valid TTA4（last.pt + 固定 EMA）

```bash
for k in 0 1 2; do
  python tools/oof_report.py metrics --labels folds3/fold${k}_valid.csv --probs oof/ra_fold${k}_valid_tta4.csv
done
```

| fold | valid 行数 | 正例 | AUROC | AUPRC | 耗时 |
|---|---|---|---|---|---|
| 0 | 19082 | 616 | 待填 | 待填 | 待填 |
| 1 | 19082 | 待填 | 待填 | 待填 | 待填 |
| 2 | 19082 | 待填 | 待填 | 待填 | 待填 |

## OOF 57245 整体（#127 clean stack 的 RA 成员）

```bash
python tools/oof_report.py metrics \
  --labels glaucoma_first/train.csv glaucoma_first/valid.csv --probs oof/ra_oof.npy
```

| 指标 | 值 |
|---|---|
| OOF AUROC | 待填 |
| OOF AUPRC | 待填 |
| 覆盖 | 57245/57245（脚本硬校验，缺一即非零退出） |

test 3 折平均：`oof/ra_test_tta4.csv`（5855），test_mean = 待填。

## 3 折两两 Spearman（验收 <0.98）

```bash
python tools/oof_report.py metrics --names-only --labels glaucoma_first/test.csv \
  --probs oof/ra_fold0_test_tta4.csv oof/ra_fold1_test_tta4.csv oof/ra_fold2_test_tta4.csv
```

| 对 | Spearman |
|---|---|
| fold0 × fold1 | 待填 |
| fold0 × fold2 | 待填 |
| fold1 × fold2 | 待填 |

## vs ens9（official valid 5849 子集）blend + 配对 bootstrap

OOF 的 valid 段（官方顺序后 5849 行）切出后对 ens9（0.99260）做 blend：

```bash
python - <<'EOF'
import csv, numpy as np
names = [r['Image_name'] for r in csv.DictReader(open('glaucoma_first/valid.csv', encoding='utf-8-sig'))]
oof = np.load('oof/ra_oof.npy'); assert len(oof) == 57245
p = oof[-len(names):]
with open('oof/ra_oof_valid_subset_tta4.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['Image_name', 'Probability', 'Pred_Label'])
    for n, v in zip(names, p): w.writerow([n, f'{v:.6f}', int(v >= 0.5)])
print('wrote oof/ra_oof_valid_subset_tta4.csv', len(names))
EOF
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/ra_oof_valid_subset_tta4.csv
```

| 项 | 值 |
|---|---|
| RA OOF(valid 段) AUROC / AUPRC | 待填 |
| Spearman(vs ens9) | 待填 |
| blend best w / Δ / P(Δ>0)（prob & logit） | 待填 |

## 验收（对照 #134）

- [ ] OOF 覆盖 57245、3 折 valid 无重叠；OOF/test 均 fixed EMA；
- [ ] test 5855 行、顺序一致、无 NaN（pipeline 硬校验）；
- [ ] 3 折两两 Spearman <0.98；
- [ ] **本单不直接宣布 99**：RA OOF 作为 #127 成员进 clean stack，
      只有 nested CV target99（Δ≥+0.0015 且 P≥0.95）通过才建议提交。

## 交付物

- `runs/ra_fold{0,1,2}/last.pt` — Release/群文件（待挂）；
- `oof/ra_fold{k}_valid_tta4.csv` × 3、`oof/ra_fold{k}_test_tta4.csv` × 3；
- `oof/ra_oof.npy`（57245）+ `oof/ra_test_tta4.csv`（5855，3 折平均）；
- 本文件回填后的完整数字。
