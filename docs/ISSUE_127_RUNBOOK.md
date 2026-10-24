# issue #127 Runbook — 用 fold-trained OOF 做干净 stacking / 权重搜索

> 本文件是 #127 的**可执行手册 + 工具预交付**。核心逻辑是 **CPU-only**（2–4h），
> 成员 OOF 的生成需要一次 GPU 推理（约 1h，可复用 4090）。不占用队长本地 5060。
>
> **泄漏红线**：57245 OOF 只能来自 **fold-trained 模型**（每折模型只预测自己没见过的 held-out 折）。
> 全局 train-only 单模（cb384、Base@384 8ep、Base@512 4ep、Swin 单模）不能用 `folds3` 生成 OOF：
> `folds3` 的 valid 含它们在官方 train 上见过的行（约 2/3 in-sample），会系统性高估。
> fold OOF 还必须固定 `--use ema`（或固定 model）；`auto`/`best.pt` 会按该折 AUROC 选权重，属于选择泄漏。

## 依赖
1. **#126 完成**：3 折 Small@384 的 `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855）；
2. **#118 可选**：3 折 Swin-B 的 OOF/test（同样是 fold-trained），作为第二组异构成员；
3. 现有全局单模**不参与** 57245 OOF；它们只能用于 official valid 5849 的 valid-only 评估（另议）。

## 产物
- `tools/stack_oof.py`：OOF stacking / 权重搜索（numpy-only，含 `--self-test`）；
- `oof/stack_test_tta4.csv`（5855）、`oof/stack_oof_probs.npy`（57245）、`oof/stack_weights.json`；
- `docs/ENSEMBLE_OOF_RESULTS.md`：nested CV、OOF 指标、与 clean baseline 的 bootstrap、失败原因。

## 步骤

### 0) 自测（CPU，秒级，不需要数据）
```bash
python tools/stack_oof.py --self-test
```

### 1) OOF 来源（只允许 fold-trained）
- **#126 base**：`oof/oof_probs.npy` + `oof/test_probs_tta4.csv`（pipeline_109 产物）；
- **#118（可选）**：把 3 折 Swin 的 OOF/test 按同样格式接进来；
- **禁止**用 `cb384`/`Base@384 8ep`/`Base@512 4ep`/`Swin 16ep` 单模生成 57245 OOF（原因见上方红线）；
- **固定权重口径**：OOF 必须用 `last.pt` + `--use ema`（或固定 model）。#126/#118 的 pipeline
  当前默认 `--use auto`，会用该折 AUROC 选 EMA/model；若已用 auto 生成 OOF，需用固定 EMA
  重新生成（TTA4 约 1h），否则本工具的 OOF 指标不可信。

### 2) 跑 stacking（CPU，2–4h）
```bash
python tools/stack_oof.py \
  --labels glaucoma_first/train.csv glaucoma_first/valid.csv \
  --folds folds3 --test-names glaucoma_first/test.csv \
  --base oof/oof_probs.npy::oof/test_probs_tta4.csv \
  --member swin3=oof/swin3_oof.npy::oof/swin3_test_tta4.csv \
  --ens9 base3=oof/oof_probs.npy::oof/test_probs_tta4.csv \
  --out-prefix oof/stack --bootstrap 2000 \
  --report docs/ENSEMBLE_OOF_RESULTS.md
```
- 成员规格 `NAME=OOF::TEST`（`::` 分隔，兼容 Windows 盘符 `C:`）；
- `OOF` 可为 `.npy`（已按官方顺序）或 `.csv`（按 `Image_name` 自动对齐）；
- `--ens9` 只是**基线槽位**：这里用干净的 3 折 Small 平均（base3）作为 baseline；
  **不要**把全局 ens9 的 folds3 OOF 当基线（会泄漏）。若没有干净 ens9 OOF，本工具的
  Δ 只能说明“相对 clean baseline”，不能直接等价于 vs 线上 ens9。
- 也可 `--autodiscover` 自动发现 `oof/*_oof.npy`；目录里只应放 fold-trained 成员。

## 方法（脚本内置，CV 自动选优）
| 方法 | 说明 | 拟合参数 |
|---|---|---|
| `rank` | 各成员 OOF 秩 [0,1] 等权平均 | 0 |
| `weight` | 非负权重单纯形上坐标上升，最大化 OOF AUROC（只用训练折标签） | m |
| `logreg` | logit(成员概率) 上 L2 逻辑回归，λ 由内层 CV 选 | m+1 |

nested 3-fold CV：外层 held-out 折评估，内层只在其余折拟合/选参 → 得到诚实的 OOF 估计。

## 验收 / 门槛
- **初筛**：ΔAUROC vs clean baseline ≥ +0.0003 且 paired bootstrap P(Δ>0) ≥ 0.90；
- **上99门槛**：ΔAUROC ≥ **+0.0015** 且 **P ≥ 0.95**（`--target-delta/--target-p`）；
- test 5855 行、顺序一致、无 NaN；权重/代码可复现；
- OOF 必须 held-out（禁止 valid-selected best.pt；禁止 train+all；禁止全局单模 OOF）；
- 只有 **上99门槛 PASS** 才建议作为线上提交候选；初筛 PASS 但上99 FAIL 只算“有希望”，
  继续等新成员/新 OOF；无增益则如实记录负结果、不提交。

## 已做的 CPU 验证
- `python tools/stack_oof.py --self-test`：合成 57245 样本/4 成员，验证 AUC/AP/bootstrap/nested CV；
- 端到端冒烟：合成 CSV+npy 跑完整 CLI，产物行数/权重/报告正确生成；
- `py_compile` 通过（脚本删除后仅 Python）。

## 备注
- 全局单模如要参与，只能在 official valid 5849 上做 valid-only 评估；不要混进 57245 OOF。
- `#109` 的 3 折 Small OOF 是 #111 蒸馏的教师来源；本单 stack 与 #111 相互独立。
