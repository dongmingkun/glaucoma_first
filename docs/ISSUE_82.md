# Issue #82 — [P1][训练] 增广多样性：Base@384 × Mixup（阶段1 负结果归档）

> 状态：阶段1 完成，valid TTA4 未达门槛（0.98920 < 0.990）且集成增益为零（w=0）｜ 指派人：JiahaoChen-star-JJ（实际由队长本地执行）｜ 执行环境：本地 RTX 5060
> 本文为 issue #82 阶段1 结果的归档；**权威内容以 GitHub issue #82 评论为准**。

---

## 1. 背景

现有成员全部用标准 BCE + 基础增广。本工单只改**训练侧 mixup**（单变量：`--mixup 0.2`），看它能否带来不同的错误结构。

## 2. 训练配置（train-only 8ep，队长本地）

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --mixup 0.2 --loader workers --workers 2 --seed 42 --tag cnxb_384_mixup_local
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @384（与 base@384 唯一差异：Mixup α=0.2） |
| 训练方式 | 从零训练（ImageNet-1k 初始化），等效 batch 48（bs8×accum6） |
| epoch / seed | 8 / 42 |
| 执行环境 | 队长本地 RTX 5060（约 2.7h，未占用云端 GPU） |

对照基线：base@384（无 mixup，valid TTA4 **0.99147**）。

## 3. 阶段1 结果（队长本地执行，2026-10-20）

| 项 | 值 |
|---|---|
| valid TTA4 | AUROC **0.98920** / AUPRC 0.85015（< 0.990 门槛） |
| 与 ens9 秩相关 | 0.8035 |
| 最优权重 | **w=0**（任何正权重都下降） |
| Δ valid AUROC | +0.00000（P=0.000） |
| 判定 | **无增益 → 阶段2 不触发，关闭** |

## 4. 结论

- mixup 让单模质量掉得较多（vs base@384 0.99147，降约 0.0023），多样性（0.80）也救不回来。
- 与同日其他 4 个多样性实验一致：**只有单模 ≥0.992 才可能进集成**。
- 阶段2（train+all + 伪标签）按工单规则**不触发**（valid TTA4 < 0.990；w=0 即零增益）。

## 5. 交付物

| 文件 | 说明 | 位置 |
|---|---|---|
| `cnxb_384_mixup_valid_tta4.csv` | valid TTA4 概率，5849 行 | 队长本地留档（未上传 Release） |
| `cnxb_384_mixup_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 | 队长本地留档（未上传 Release） |

> 注：本单由队长本地执行，产物 CSV 仅本地留档，未纳入 GitHub Release `weights-1016`。

## 6. 下一步

- 归档为「负结果 / 不进集成」，释放资源。
