# Issue #79 — 新尺度：Base@448（阶段1 负结果，归档 / 不进集成）

> 状态：阶段1 完成，valid TTA4 未达门槛（0.98786 < 0.990）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #79 阶段1 结果的归档；**权威内容以 GitHub issue #79 评论为准**。

---

## 1. 背景

现有成员只有 384 / 512 两个分辨率档；Base@512 的 train+all 已被证伪（check 0.875/0.864），Base@384 有效。本工单补中间档 **448**（单变量：分辨率），看它能否在质量与多样性之间取得平衡。

## 2. 训练配置（train-only 8ep）

```bash
python train.py --arch convnext_base --img-size 448 --cache-root cache512   --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4   --loader workers --workers 3 --seed 42 --tag cnxb_448
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @448（cache512 原生下采样到 448，无需新缓存） |
| 训练方式 | 从零训练（ImageNet-1k 初始化，与 base@384 完全一致，仅分辨率不同） |
| epoch / seed | 8 / 42 |
| 显存 / 耗时 | 5.7G / 每 epoch ≈ 16 min，8ep 约 2h |

## 3. 训练曲线（EMA AUROC）

| epoch | 单变 AUROC | EMA AUROC | EMA AUPRC | balmax |
|---|---|---|---|---|
| 1 | 0.9752 | 0.9790 | 0.7867 | 0.9488 |
| 2 | 0.9820 | 0.9860 | 0.8045 | 0.9594 |
| 3 | 0.9771 | 0.9847 | 0.8240 | 0.9543 |
| 4 | 0.9835 | 0.9850 | 0.8308 | 0.9495 |
| 5 | 0.9860 | 0.9850 | 0.8467 | 0.9525 |
| 6 | 0.9796 | 0.9850 | 0.8492 | 0.9486 |
| 7 | 0.9844 | 0.9851 | 0.8505 | 0.9530 |
| 8 | 0.9871 | 0.9859 | 0.8536 | 0.9516 |

## 4. 验收

`infer.py --ckpt runs/cnxb_448/best.pt --split valid --tta 4`（use=ema）：

- **valid AUROC = 0.98786 / AUPRC = 0.86164**
- 对照 base@384（valid TTA4 **0.99147**）→ 448 档 **−0.0036**，且 < 0.990 门槛
- 每 epoch EMA AUROC 在 0.9847–0.9859 间几乎走平（ep5 后无明显上升），峰值 ep8 EMA 0.9859

## 5. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `cnxb_448_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准（EMA 口径，与 #76 一致） |

## 6. 结论与下一步

- **单变量负结果**：分辨率从 384 拔高到 448 未带来质量增益（EMA AUROC 走平、valid TTA4 反而 −0.0036）。
- 阶段2（train+all + 伪标签）按工单规则**不触发**（valid TTA4 < 0.990 门槛未达）。
- test 概率仍挂出，供队长用干净 valid 离线算它与现有集成的真实增益；预期 w=0。
- **建议归档为「负结果 / 不进集成」**，释放 GPU 给后续工单（#83 IN-22k、#84 pos_weight=3）。
