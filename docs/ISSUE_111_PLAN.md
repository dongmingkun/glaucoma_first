# issue #111 集成蒸馏配方与复现计划

> 关联工单：#111 [P1][训练] 集成蒸馏成可部署单模型/双模型复赛包（云端 24GB+，不占 5060）
> 本目录交付**蒸馏代码 + 配方文档**（纯 CPU 可写、可复现），实际训练在云端 24GB+ 卡完成。

## 目标
把当前最强集成蒸馏成可部署学生模型（优先单模型，其次双模型），在 ≤500MB、T4 ≤100ms 前提下尽量接近教师集成 AUROC。
**验收**：学生 valid AUROC ≥ 教师 −0.001；包体 ≤500MB；T4 p95 ≤100ms（真机或 #81/T4 工单实测）。

## 教师优先级（按队长 #111 评论）
1. **首选**：#109 的 **fold 模型**当教师 —— 用 `--fold-valid` 让 train/valid 每一行只取「没见过它」的 held-out 折教师；test 用全部教师平均。
2. **时间紧**：先用 **ens9（train-only）** 起一版蒸馏；ens10 的 train+all 家族看过 valid，**不能**用于 valid 软标签。
3. **test 软标签**：用当前最强集成（ens10/ens22 或 fold 集成）生成。
> 绝不用看过 valid 的模型当教师去生成 valid 软标签（否则蒸馏评估虚高）。

## 流水线

### Step 1 — 生成教师软标签（`tools/gen_soft_labels.py`，云端 GPU）

**A. fold-OOF（首选，配合 #109 的 fold 模型）**

`--teacher` 与 `--fold-valid` 顺序一一对应；train/valid 每行只取它所在 held-out 折的教师，test 用全部教师加权平均。

```bash
python tools/gen_soft_labels.py \
    --teacher runs/fold0/last.pt --teacher runs/fold1/last.pt \
    --teacher runs/fold2/last.pt --teacher runs/fold3/last.pt --teacher runs/fold4/last.pt \
    --fold-valid folds/fold0_valid.csv folds/fold1_valid.csv \
                 folds/fold2_valid.csv folds/fold3_valid.csv folds/fold4_valid.csv \
    --fold-cache-root cache_all --cache-root cache512 \
    --split train valid test --tta 4 --out soft_fold_oof.csv
```

- `cache_all/train` 必须放 train+valid 的全部 57245 张（fold CSV 混合 `train_*.jpg` / `valid_*.jpg`）；建法见 `folds/README.md` 的 `cp -rs`。
- test 用 `--cache-root cache512`（原始 cache）；train/valid 用 `--fold-cache-root cache_all`。
- fold 教师建议用 `last.pt`（固定 16ep），不要用按 held-out valid 选的 `best.pt`。

**B. 单/普通集成（时间紧的 fallback）**

```bash
python tools/gen_soft_labels.py \
    --teacher runs/ens9/best.pt --teacher-weights 1.0 \
    --split train valid test --tta 4 --out soft_trainonly.csv
```

- 教师必须没看过 valid（ens9 是 train-only；ens10 的 train+all 家族不能用）。
- 输出：`Image_name, Age, Soft_Label`（train/valid 另带 `True_Label`）。
- 权重已归一化，输出是加权概率本身（不再除以教师数）。

### Step 2A — 配方验收（train-only 学生 + 干净 valid）

```bash
python tools/distill.py \
    --arch convnext_base --img-size 512 --cache-root cache768 \
    --epochs 10 --soft-labels soft_fold_oof.csv \
    --kd-alpha 0.5 --kd-temp 2.0 --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 \
    --select-metric auroc --out-dir runs/distill_base512_eval
```

- 不加 `--train-on-all`：只在官方 train 训，valid 是干净评估；用 `best.pt`（valid 选）或 `last.pt` 比较学生 vs 教师。
- 这一步决定配方是否成立；验收：学生 valid AUROC ≥ 教师 −0.001。

### Step 2B — 最终学生（train+valid 全量，不用于验收）

```bash
python tools/distill.py \
    --arch convnext_base --img-size 512 --cache-root cache_all --cache-size 768 \
    --epochs 10 --soft-labels soft_fold_oof.csv \
    --kd-alpha 0.5 --kd-temp 2.0 --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 \
    --train-on-all --out-dir runs/distill_base512_final
```

- `--train-on-all` 必须配合并 cache（`cache_all`）；`--cache-size 768` 让 checkpoint 记录正确的 cache 档位（Base@512 用 cache768 建 cache_all）。
- train-on-all 后 valid 不再独立，**不能**用它的 AUROC 做验收；最终学生按 2A 的配方/epoch 数直接产出。
- `predict.py` 会读 checkpoint 的 `cache_size`，打包时不会用错档。
- 损失：`L = α·L_hard(logits, y) + (1−α)·T²·BCE(sigmoid(logits/T), sigmoid(logit_T(y_soft)/T))`
  - `L_hard` = BCE(pos_weight) 或 Focal（`--focal-gamma>0`）；软标签项经温度 T 软化后与学生对齐（二元 KL 蒸馏）。
- 复用 `train.py` 的 FundusNet / ModelEMA / evaluate / ThreadLoader / 余弦 LR / AMP；`--select-metric` 控制按 AUROC/AUPRC/blend 选 model 或 EMA，并写入 checkpoint 的 `use`。

## 超参建议（来自 #111）
- 学生：ConvNeXt-Base@512 / cache768（或 Base@384，按显存/延迟）；8–12ep；强增广；EMA。
- `kd-alpha` 0.5（硬/软各半）、`kd-temp` 2.0 起；`pos-weight` 5.0；lr 1e-4 / head-lr 5e-4。
- 评估：独立 valid 比较学生 vs 教师（教师须 train-only）；报告 AUROC/AUPRC、与教师 Spearman、与学生单模基线差。

## 交付物（训练完成后）
- 学生 `best.pt` + fp16 slim（Release / 群文件，≤500MB）；
- valid/test 概率 CSV（5855 行 test，顺序一致、无 NaN）；
- `docs/ISSUE_111_RESULTS.md`：教师/学生/基线对比、蒸馏配方、延迟、包体；
- 打包目录 `submission_pkg/`（predict.py + 权重 + README），T4 实测联动 #81 / T4 工单。

## 说明
- 本 PR 仅交付**蒸馏代码与配方**（CPU 可写、语法校验通过、复用 train.py 已验证组件）。
- 实际蒸馏训练 / 软标签生成 / 打包需云端 24GB+ GPU，不占队长本地 5060；由云端实例运行 Step1→Step2 后回填 RESULTS 与 Release。
- 与 #110 一致：未擅自重启云端实例（遵循「不用云GPU」约束），代码就绪后等云端资源即可一键跑通。
