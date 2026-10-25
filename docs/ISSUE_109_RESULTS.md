# Issue #126 结果记录 — 3-fold ConvNeXt-Small@384 16ep 真机执行（OOF 57245 + test 5855）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/126 （#109 方案 A 的云端实跑）。
> 真机执行完成（2026-10-24 22:39 → 10-25 05:52 训练段；10-25 09:15–09:26 OOF 补跑），
> 本文件为回填后的完整结果。指标工具：`tools/oof_report.py`（CPU），本地 + 云机双重复算一致。
> **执行事故与修复**见文末「事故记录」——valid OOF 曾因空 Age 崩溃后零填，已用 PR #136 修复并补跑，
> 本报告数字全部来自补跑后的干净产物。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `CACHE_SRC=<data>/cache512 BS=8 ACCUM=6 bash tools/pipeline_109.sh`（fix/126-oof-fixed-ema 分支，OOF 固定 EMA） |
| 机器 | AutoDL RTX 4090 24GB（不占队长本地 5060） |
| 耗时 | 训练 3 折各 ~2h23m（22:39 → 05:49），全流水线 7h13m；OOF 补跑 11 分钟 |
| 缓存 | `cache512 51396/5849/5855` → `cache_all` 57245（符号链接，脚本硬校验通过） |
| 口径 | 每折 ImageNet 初始化、seed 42；OOF/test 一律 `last.pt` + 固定 `--use ema`（禁止 best/auto） |

## 每折 held-out valid TTA4（last.pt + 固定 EMA）

| fold | valid 行数 | 正例 | AUROC | AUPRC | balACC@0.5 |
|---|---|---|---|---|---|
| 0 | 19082 | 616 | 0.98662 | 0.82623 | 0.88417 |
| 1 | 19082 | 616 | 0.98635 | 0.83456 | 0.89215 |
| 2 | 19081 | 616 | 0.98841 | 0.84553 | 0.89334 |

（fold2 比另两折少 1 行是 folds3 切分固有：三折合计 = 57245 整。）

## OOF 57245 整体（#127 clean stacking 的 base 成员）

| 指标 | 值 |
|---|---|
| OOF AUROC | **0.98685** |
| OOF AUPRC | **0.83398** |
| balACC@0.5 | 0.88989 |
| 覆盖 | **57245/57245**（组装脚本硬校验，缺一即非零退出；npy 与 per-fold CSV 逐名对齐 max diff = 0） |
| 概率范围 | [0.000013, 0.999693]，mean 0.04032，全部有限值 |

OOF 的 official valid 5849 子集：AUROC 0.98958 / AUPRC 0.84969
（vs ens9 0.99110——OOF 是「模型没见过的图」上的诚实估计，直接对比单模 valid 略低是预期行为；
blend/bootstrap 需 ens9 valid 概率文件，**待补**，命令已就绪）。

test 3 折平均：`oof/test_probs_tta4.csv`（本 PR 交付名 `cnxs3_test_probs_tta4.csv`），
5855 行，test_mean 0.04671。

## 3 折两两 Spearman（test 5855；验收 <0.98 ✅）

| 对 | Spearman |
|---|---|
| fold0 × fold1 | 0.77214 |
| fold0 × fold2 | 0.73231 |
| fold1 × fold2 | 0.78759 |

三折均为 ImageNet 初始化 + 不同数据子集，相关度远低于 0.98 门槛——异构性良好，适合 stacking。

## 与 ens9 的配对 bootstrap

需 ens9 valid 概率文件（队长本地/Release），**待补**：

```bash
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/oof_valid_subset_tta4.csv
```
（`oof_valid_subset_tta4.csv` 的切出命令见模板历史；OOF npy 按官方 train+valid 顺序，valid 为后 5849 行。）

## 验收（对照 #126）

- [x] OOF 覆盖 57245、无 train/valid 重叠（fold 训练用 `--csv-root folds3/fold{k}_csvroot`，PR #120 修复后实测 fold0 38163/19082）；
- [x] test 5855 行、顺序与官方 CSV 一致、无 NaN（脚本硬校验）；
- [x] 3 折两两 Spearman <0.98（0.73–0.79）；
- [x] OOF 用 `last.pt` + 固定 `--use ema`（PR #131 修复分支；运行日志 `use=ema` 可查）；
- [x] 某折失败时脚本非零退出、不零填 OOF——**初版脚本违反此条（见下），已修复并补跑**。

## 事故记录（透明披露）

- 训练段（22:39–05:52）rc=0，但复查发现 per-fold **valid OOF 全部缺失**、`oof_probs.npy` 为零填：
  官方 `train.csv` 有 **10 行 Age 为空**（folds3 fold valid 继承），`oof_predict.load_csv` 裸
  `float(Age)` 崩溃；初版 pipeline 预测步骤无失败保护、组装「先落盘后校验」、外层未查 rc，
  导致零填产物以 rc=0 收场（违反验收）；
- 修复：PR #136——`oof_predict` Age 清洗与训练侧 `paths.load_rows` 完全同口径
  （AGE_MEDIAN 兜底 + 钳制 [10,100]，不改变非空行行为）+ pipeline_109/118 失败即停/先校验后落盘；
- 补跑：3 折 ckpt 与 test 概率完好无需重训；09:15–09:26 用修复版补齐 3 折 valid OOF + 组装
  （rc=0，覆盖 57245/57245）。**本报告全部数字来自补跑后产物**，test 段为原跑产物（未受影响）。

## 交付物

- 权重：`runs/cnxs_fold{0,1,2}/last.pt`（各 381MB），Release
  [`cnxs3-1025`](https://github.com/dongmingkun/glaucoma_first/releases/tag/cnxs3-1025)；
- `oof/cnxs_fold{k}_valid_tta4.csv` × 3 + `oof/cnxs_fold{k}_test_tta4.csv` × 3（本 PR，仓库根目录）；
- `oof/oof_probs.npy`（57245）→ Release 资产（*.npy 不入库）；
- `oof/test_probs_tta4.csv`（5855，3 折平均）→ 本 PR `cnxs3_test_probs_tta4.csv`；
- 训练曲线 × 3：`cnxs_fold{k}_log.csv`（本 PR）；
- 本文件回填后的完整数字。

## 下游

- **#127（干净 stacking）**：`oof_probs.npy` + `test_probs_tta4.csv` 即 runbook 里的 base 成员，
  可立即开跑（CPU-only）；
- **#111（蒸馏）**：3 个 `last.pt` 为教师；`gen_soft_labels.py` 的 auto 选权建议同样固定 EMA（另行小改）；
- **#134（RA 3-fold）**：正在同一实例上跑（ra_fold1，预计 10-25 下午出 OOF），
  其修复版 `oof_predict` 已就位。
