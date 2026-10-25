# Issue #134 结果记录 — RA Base@512 16ep × 3-fold clean OOF（#128 最强新成员纳入 #127 99 门槛）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/134 。真机执行完成（2026-10-25
> 05:56 → 14:13，rc=0），本文件为回填后的完整结果。指标工具：`tools/oof_report.py`（CPU），
> 本地 + 云机双重复算一致。
> **本单不直接宣布 99**：RA OOF 作为 #127 clean stack 成员，只有 nested CV target99
> （Δ≥+0.0015 且 P≥0.95）通过才建议提交；能提分门槛 Δ≥+0.0005 且 P≥0.90 同样由 #127 判定。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `CACHE_SRC=<data>/cache768 bash tools/pipeline_134.sh`（feat/134 分支 = main PR #135） |
| 机器 | AutoDL RTX 4090 24GB |
| 耗时 | 05:56 → 14:13 ≈ 8h17m（3 折训练 ~2.5h/折 + 6 次 TTA4 推理 + 组装） |
| 缓存 | `cache768 51396/5849/5855` → `cache_all768 57245/57245/5855`（符号链接，硬校验通过） |
| 配方 | ConvNeXt-Base@512 16ep bs8×accum6 + RandAugment 2 9，seed 7，每折 ImageNet 初始化 |
| 口径 | OOF/test 一律 `last.pt` + 固定 `--use ema`（含 PR #136 修复版 oof_predict，Age 同口径清洗） |

## 每折 held-out valid TTA4（last.pt + 固定 EMA）

| fold | valid 行数 | 正例 | AUROC | AUPRC |
|---|---|---|---|---|
| 0 | 19082 | 616 | 0.98750 | 0.85167 |
| 1 | 19082 | 616 | 0.98995 | 0.84822 |
| 2 | 19081 | 616 | 0.98995 | 0.86673 |

（fold2 少 1 行为 folds3 切分固有，三折合计 = 57245 整。）

## OOF 57245 整体（#127 clean stack 的 RA 成员）

| 指标 | 值 |
|---|---|
| OOF AUROC | **0.98887** |
| OOF AUPRC | **0.85420** |
| 覆盖 | 57245/57245（组装脚本硬校验，缺一即非零退出；npy 与 per-fold CSV 逐名对齐 max diff = 0） |
| vs cnxs3 OOF（#126） | RA 0.98887/0.85420 **>** Small 0.98685/0.83398——RA fold 版同样强于 Small fold 版，与单模结论一致 |

test 3 折平均：`oof/ra_test_tta4.csv`（本 PR 交付名 `ra3_test_probs_tta4.csv`），
5855 行，test_mean 0.05250。

## 3 折两两 Spearman（test 5855；验收 <0.98 ✅）

| 对 | Spearman |
|---|---|
| fold0 × fold1 | 0.74597 |
| fold0 × fold2 | 0.76937 |
| fold1 × fold2 | 0.73530 |

## 与 cnxs3 OOF 的互补性（#127 成员间关键指标）

**Spearman(RA OOF, cnxs OOF) = 0.66772（57245 全量）**——两大家族（同为 ConvNeXt 但
分辨率/增强/数据子集不同）排序相关仅 0.67，是 stack 里互补性最强的组合之一；
叠加 #139 Swin（架构级异构）后，三成员 clean stack 值得期待。

## vs ens9 blend + 配对 bootstrap

ens9 valid 概率文件（团队 zip）仍缺，**待补**；`oof/ra_oof_valid_subset_tta4.csv`
（OOF 的 official valid 5849 段，pipeline 已产出，AUROC 0.98886 / AUPRC 0.86410）已就位：

```bash
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/ra_oof_valid_subset_tta4.csv
```

## 验收（对照 #134）

- [x] OOF 覆盖 57245、3 折 valid 无重叠；OOF/test 均 fixed EMA；
- [x] test 5855 行、顺序一致、无 NaN（pipeline 硬校验）；
- [x] 3 折两两 Spearman <0.98（0.73–0.77）；
- [x] 不直接宣布 99——留给 #127 nested CV gate（Δ≥+0.0015/P≥0.95；提分 Δ≥+0.0005/P≥0.90）。

## 交付物

- 权重：`runs/ra_fold{0,1,2}/last.pt`（各 673MB），Release
  [`ra3-1025`](https://github.com/dongmingkun/glaucoma_first/releases/tag/ra3-1025)（含 `ra_oof.npy`）；
- 概率 CSV × 6（本 PR，仓库根目录）：`ra_fold{k}_{valid,test}_tta4.csv`；
- `ra3_test_probs_tta4.csv`（5855，3 折平均）；
- 训练曲线 × 3：`ra_fold{k}_log.csv`；
- 本文件回填后的完整数字。

## 下游

- **#127**：ra3 OOF 即刻可用；等 #139 swin3 OOF（明天）后三成员 nested CV 一次跑完；
- **#138**：train+all 版 RA 成员已在跑（复用本单建好的 cache_all768）。
