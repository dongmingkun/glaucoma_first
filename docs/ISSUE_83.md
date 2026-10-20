# Issue #83 — 预训练来源：Base@384 × ImageNet-22k（阶段1 负结果，归档 / 不进集成）

> 状态：阶段1 完成，valid TTA4 未达门槛（0.98926 < 0.990，且 < base@384 0.99147）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #83 阶段1 结果的归档；**权威内容以 GitHub issue #83 评论为准**。

---

## 1. 背景

现有 base@384 用 torchvision ImageNet-1k 初始化。本工单只改**预训练权重来源**（单变量）：timm `convnext_base.fb_in22k_ft_in1k`（ImageNet-22k 通用预训练，非医学数据，合规）。代码已就绪（main，commit 54f90aa）：`--arch convnext_base_22k`。

## 2. 训练配置（train-only 8ep）

```bash
python train.py --arch convnext_base_22k --img-size 384 --cache-root cache512   --epochs 8 --batch-size 12 --accum 4 --lr 1e-4 --head-lr 5e-4   --loader workers --workers 3 --seed 42 --tag cnxb_384_22k
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @384，IN-22k 预训练（相对 base@384 仅权重来源不同） |
| 训练方式 | 从 IN-22k 初始化微调，其余配置与 base@384 完全一致 |
| epoch / seed | 8 / 42 |
| 环境处理 | 云上代码非 git 仓库（快照落后），按 commit 54f90aa patch `train.py` 的 TIMM_ARCH 增加 `convnext_base_22k`；huggingface CDN 被墙，训练设 `HF_ENDPOINT=https://hf-mirror.com` 让 timm 从镜像拉取 22k 权重 |
| 显存 / 耗时 | 6.1G / 每 epoch ≈ 8 min，8ep 约 1h |

## 3. 训练曲线

| epoch | val_loss | AUROC(单变) | AUPRC | EMA AUROC | EMA AUPRC |
|---|---|---|---|---|---|
| 1 | 0.1837 | 0.9487 | 0.6963 | 0.9802 | 0.7800 |
| 2 | 0.1983 | 0.9737 | 0.6273 | 0.9838 | 0.7951 |
| 3 | 0.1814 | 0.9755 | 0.7648 | 0.9854 | 0.8232 |
| 4 | 0.1186 | 0.9863 | 0.8215 | 0.9876 | 0.8384 |
| 5 | 0.1164 | 0.9844 | 0.8340 | 0.9899 | 0.8471 |
| 6 | 0.1172 | 0.9863 | 0.8481 | 0.9885 | 0.8527 |
| 7 | 0.1092 | 0.9885 | 0.8581 | 0.9877 | 0.8561 |
| 8 | 0.1125 | 0.9867 | 0.8566 | 0.9872 | 0.8570 |

## 4. 验收

`infer.py --ckpt <best.pt> --split valid --tta 4 --use ema`：

- **valid AUROC = 0.98926 / AUPRC = 0.86715**
- 对照 base@384（IN-1k，valid TTA4 **0.99147**）→ 22k 档 **−0.0022**
- 每 epoch EMA AUROC 在 0.9802→0.9899 间爬升（ep5 峰值 0.9899），ep8 EMA 0.9872

## 5. 交付物（Release `weights-1016`，EMA 口径，与 #76/#79 一致）

| 文件 | 说明 |
|---|---|
| `cnxb_384_22k_valid_tta4.csv` | 干净 valid TTA4 概率，5849 行 |
| `cnxb_384_22k_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 |

## 6. 结论与下一步

- **单变量负结果**：IN-22k 通用预训练**未带来质量增益**，valid TTA4（EMA）0.98926 < 0.990 门槛、且低于 base@384（0.99147）。
- 阶段2（train+all + 伪标签）按工单规则**不触发**（增益为负）。
- 两份概率仍挂出，供队长用干净 valid 离线算它与现有集成的真实增益；预期 w=0。
- **建议归档为「负结果 / 不进集成」**。
