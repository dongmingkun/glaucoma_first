# issue #111 集成蒸馏配方与复现计划

> 关联工单：#111 [P1][训练] 集成蒸馏成可部署单模型/双模型复赛包（云端 24GB+，不占 5060）
> 本目录交付**蒸馏代码 + 配方文档**（纯 CPU 可写、可复现），实际训练在云端 24GB+ 卡完成。

## 目标
把当前最强集成蒸馏成可部署学生模型（优先单模型，其次双模型），在 ≤500MB、T4 ≤100ms 前提下尽量接近教师集成 AUROC。
**验收**：学生 valid AUROC ≥ 教师 −0.001；包体 ≤500MB；T4 p95 ≤100ms（真机或 #81/T4 工单实测）。

## 教师优先级（按队长 #111 评论）
1. **首选**：#109 的 **fold 集成 OOF / fold 模型**当教师 —— 每个 train+valid 行的软标签来自未见过它的模型，评估无污染，比 valid-selected 更可信。
2. **时间紧**：先用 **ens9 / ens10 的 train-only 部分**起一版蒸馏（教师没看过 valid，避免评估虚高）。
3. **test 软标签**：用当前最强集成（ens10/ens22 或 fold 集成）生成。
> 绝不用看过 valid 的模型当教师去生成 valid 软标签（否则蒸馏评估虚高）。

## 流水线
### Step 1 — 生成教师软标签（`tools/gen_soft_labels.py`，云端 GPU）
```bash
# 单教师（或 fold 集成：逐个 fold 模型，各自 arch/img_size 跑后按权重平均）
python tools/gen_soft_labels.py \
    --teacher runs/fold0/best.pt --teacher runs/fold1/best.pt --teacher runs/fold2/best.pt \
    --teacher runs/fold3/best.pt --teacher runs/fold4/best.pt \
    --teacher-weights 0.2 0.2 0.2 0.2 0.2 \
    --split train valid test --tta 4 --out soft_fold_ens.csv
# 时间紧时改用 ens9/ens10 train-only：
#   --teacher runs/ens9/best.pt --teacher runs/ens10/best.pt --teacher-weights 0.5 0.5
```
- 输出 `soft_fold_ens.csv`：`Image_name, Age, Soft_Label`（train/valid 另带 `True_Label` 供参考）。
- 异构集成：每个教师用**自身** arch/img_size 推理后按权重平均概率（概率可比）。
- `use` 字段解析与 `train.py --init-from` 一致（ema / model / state_dict）。

### Step 2 — 蒸馏学生（`tools/distill.py`，云端 GPU）
```bash
python tools/distill.py \
    --arch convnext_base --img-size 512 --epochs 10 \
    --soft-labels soft_fold_ens.csv \
    --kd-alpha 0.5 --kd-temp 2.0 \
    --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 \
    --train-on-all --out-dir runs/distill_base512
```
- 损失：`L = α·L_hard(logits, y) + (1−α)·T²·BCE(sigmoid(logits/T), sigmoid(logit_T(y_soft)/T))`
  - `L_hard` = BCE(pos_weight) 或 Focal（`--focal-gamma>0`）；软标签项经温度 T 软化后与学生对齐（二元 KL 蒸馏）。
- 复用 `train.py` 的 FundusNet / ModelEMA / evaluate / ThreadLoader / 余弦 LR / AMP；valid 用**真标签**评估挑 `best.pt`。
- `--train-on-all`：在 train+valid 全量训（默认），因教师为 train-only，valid 软标签诚实，valid 真标签 AUROC 仍可作信号。

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
