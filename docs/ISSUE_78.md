# Issue #78 — [P1][训练] 目标函数多样性：Base@384 × Focal Loss（阶段1 负结果归档）

> 状态：阶段1 完成，valid TTA4 未达集成增益门槛（ΔAUROC +0.00001，P=0.596）｜ 指派人：JiahaoChen-star-JJ（实际由队长本地执行）｜ 执行环境：本地 RTX 5060
> 本文为 issue #78 阶段1 结果的归档；**权威内容以 GitHub issue #78 评论为准**。

---

## 1. 背景

权重线已关闭（0/20/40/100% = 98.15/98.82/98.80/98.73），剩余空间只能来自**多样性**。现有成员全部用 BCE；本工单只改 **loss**（单变量：`--focal-gamma 2.0`），看 focal loss 能否给出不同的错误结构。

## 2. 训练配置（train-only 8ep，队长本地）

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --focal-gamma 2.0 --loader workers --workers 2 --seed 42 --tag cnxb_384_focal_local
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @384（与 base@384 唯一差异：Focal Loss γ=2.0） |
| 训练方式 | 从零训练（ImageNet-1k 初始化），等效 batch 48（bs8×accum6） |
| epoch / seed | 8 / 42 |
| 执行环境 | 队长本地 RTX 5060（约 5h，未占用云端 GPU） |

对照基线：base@384（BCE，valid TTA4 **0.99147**）。

## 3. 阶段1 结果（队长本地执行，2026-10-20）

| 项 | 值 |
|---|---|
| valid TTA4 | AUROC **0.99054** / AUPRC 0.85576（达到 0.990 门槛） |
| 与 ens9 秩相关 | 0.8078 |
| 最优权重 | w=0.02 |
| Δ valid AUROC | **+0.00001**（配对 bootstrap 1000 次：P(改善)=0.596，95% CI [−0.00009, +0.00011]） |
| 判定 | **无增益 → 阶段2 不触发，关闭** |

## 4. 结论

- focal loss 单模质量（0.99054）略低于 base@384（0.99147）；多样性不错（与 ens9 秩相关 0.81），但对集成的贡献完全落在噪声内（P=0.596）。
- 与 Swin 16ep（valid 0.99233）对比说明：**成员绝对质量优先，多样性只在 ≥0.992 时才可能转化为集成增益**。
- 阶段2（train+all + 伪标签）按工单规则**不触发**（集成增益未达正显著）。

## 5. 交付物

| 文件 | 说明 | 位置 |
|---|---|---|
| `cnxb_384_focal_valid_tta4.csv` | valid TTA4 概率，5849 行 | 队长本地留档（未上传 Release） |
| `cnxb_384_focal_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 | 队长本地留档（未上传 Release） |

> 注：本单由队长本地执行，产物 CSV 仅本地留档，未纳入 GitHub Release `weights-1016`。如需进入集成复现流程，建议后续补传。

## 6. 下一步

- 归档为「负结果 / 不进集成」，释放资源。
- 与 #77/#80/#82 等多样性实验结论一致：<0.992 的成员无法转化为集成增益。
