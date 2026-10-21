# Issue #80 — [P2][训练] 增广多样性：Base@384 × RandAugment（阶段1 负结果归档）

> 状态：阶段1 完成，valid TTA4 未达门槛（0.98953 < 0.990）且集成增益不显著（P=0.602）｜ 指派人：JiahaoChen-star-JJ（实际由队长本地执行）｜ 执行环境：本地 RTX 5060
> 本文为 issue #80 阶段1 结果的归档；**权威内容以 GitHub issue #80 评论为准**。

---

## 1. 背景

Small@512 的 RA 版（#8）是家族里 check 最高的成员（0.9132），但 Base@384 还没有 RA 版。本工单只改**增广**（单变量：`--randaugment 2 9`），补齐这个组合。

## 2. 训练配置（train-only 8ep，队长本地）

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --randaugment 2 9 --loader workers --workers 2 --seed 42 --tag cnxb_384_ra_local
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @384（与 base@384 唯一差异：RandAugment 2 9） |
| 训练方式 | 从零训练（ImageNet-1k 初始化），等效 batch 48（bs8×accum6） |
| epoch / seed | 8 / 42 |
| 执行环境 | 队长本地 RTX 5060（约 3h，未占用云端 GPU） |

对照基线：base@384（无 RA，valid TTA4 **0.99147**）。

## 3. 阶段1 结果（队长本地执行，2026-10-20）

| 项 | 值 |
|---|---|
| valid TTA4 | AUROC **0.98953** / AUPRC 0.86633（< 0.990 门槛） |
| 与 ens9 秩相关 | 0.8574 |
| 最优权重 | w=0.06 |
| Δ valid AUROC | **+0.00001**（P=0.602，95% CI [−0.00007, +0.00010]） |
| 判定 | **无实质增益 → 阶段2 不触发，关闭** |

## 4. 结论

- RandAugment 版 base@384 单模 0.98953，低于无 RA 的 0.99147；与同日其余多样性实验一致，**<0.992 的成员无法转化为集成增益**。
- 阶段2（train+all + 伪标签）按工单规则**不触发**（valid TTA4 < 0.990 门槛；集成增益不显著）。

## 5. 交付物

| 文件 | 说明 | 位置 |
|---|---|---|
| `cnxb_384_ra_valid_tta4.csv` | valid TTA4 概率，5849 行 | 队长本地留档（未上传 Release） |
| `cnxb_384_ra_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 | 队长本地留档（未上传 Release） |

> 注：本单由队长本地执行，产物 CSV 仅本地留档，未纳入 GitHub Release `weights-1016`。

## 6. 下一步

- 归档为「负结果 / 不进集成」，释放资源。
