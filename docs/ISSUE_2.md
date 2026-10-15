# Issue #2 — B3@448 微调收尾 + 与 B3@384 集成（完成记录）

> 本工单是**训练实验**，主体已在队长机器上跑完。本文档把结论、命令、回填文案固化下来，
> 供 issue 关闭时引用。代码工具链（`--init-from` 热启动、`--select-metric`、`ensemble.py`、
> `make_submission.py`）均已就绪，无需新增代码。

## 1. 结论（可直接回填 Issue）

| 项 | 结果 |
|---|---|
| B3@448 微调（从 384 best 热启动，4 ep） | AUROC **0.9820** / AUPRC **0.7996** |
| B3@448 + 4x 翻转 TTA | AUPRC **0.8267**（比 384 版 +0.011，AUROC 略低 −0.003） |
| 384 + 448 等权集成 | AUROC **0.98507** / AUPRC **0.82839** |
| 校准版提交文件 | `submission_20251015_ens_cal.csv` |

**分辨率效应（本工单最重要的经验）**：448 相对 384，**AUPRC +0.011、AUROC −0.003**。
更高分辨率显著改善"排序头部质量"，正好补上本项目 AUPRC 的短板；但 AUROC 反而略降，
所以单靠 448 不划算，**必须与 384 集成**才能两头兼得。

**配套修复**：原 `best.pt` 只按 AUROC 选，会把 AUPRC 最差的 epoch 挑出来（B3@512 就因此
过拟合，AUPRC 从 0.81 一路掉到 0.74）。已加 `--select-metric`，默认 `blend`
（0.5·AUROC + 0.5·AUPRC），选 checkpoint 时不再被 AUROC 带偏。

## 2. 复现命令

```bash
VENV=D:\glaucoma_work\venv\Scripts\python.exe

# 1) B3@448 微调：从 384 best.pt 热启动，不要 from scratch
$VENV train.py --arch efficientnet_b3 --img-size 448 --epochs 4 \
  --batch-size 12 --accum 3 --loader workers --workers 3 \
  --init-from "D:/glaucoma_work/runs/<384_exp>/best.pt" --tag b3_448

# 2) 384+448 集成（等权概率平均，最稳健）
$VENV ensemble.py --ckpt <384/best.pt> <448/best.pt> --eval --tta 4      # 看验证集增益
$VENV ensemble.py --ckpt <384/best.pt> <448/best.pt> --submit --out submission.csv

# 3) 走唯一提交入口（评估→集成→校准→写CSV→四口径预期分）
$VENV make_submission.py --ckpt <384/best.pt> <448/best.pt> --tta 4 --calibrate --out submission.csv
```

集成方式结论（`compare_ensembles.py` 已验）：等权概率平均 / 加权 / rank / logit 平均
差异都在抽样噪声内；rank 平均伤 AUROC，logit 平均 AUPRC 略高但 AUROC 略低。**默认等权概率平均**。

## 3. 状态说明

- 训练与集成产物在队长机器（权重不入库，按 README 走网盘/本机分发）
- 本机无 GPU 训练环境、无 cache512、无对应权重，无法重跑验证；数字以队长机器的
  `evaluate.py` / `ensemble.py` 输出为准
- 后续主力方向已转向 **ConvNeXt**（见 PROJECT_NOTES 2026-10-15），B3 家族不再投入

## 4. Issue 回填文案（直接粘贴到 #2）

```
#2 已完成（B3@448 微调 + 384/448 集成）

- B3@448 微调(从 384 best 热启动, 4ep): AUROC 0.9820 / AUPRC 0.7996；4xTTA 后 AUPRC 0.8267
- 384+448 等权集成: AUROC 0.98507 / AUPRC 0.82839
- 校准版提交文件 submission_20251015_ens_cal.csv 已就绪
- 配套修复: best.pt 只按 AUROC 选会挑中 AUPRC 最差 epoch → --select-metric 默认 blend(0.5*AUROC+0.5*AUPRC)
- 分辨率效应: 448 vs 384, AUPRC +0.011 / AUROC -0.003（高分辨率补 AUPRC 短板，需与 384 集成）
- 复现命令与集成方式对比见 docs/ISSUE_2.md
```
