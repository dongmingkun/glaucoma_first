# Issue #58 — train+all 第二批：ConvNeXt-Base@512 全量重训（负结果收口）

> 状态：负结果（8ep / 7ep 均 check REVIEW）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #58 训练结果的归档；**权威内容以 GitHub issue #58 评论为准**。

---

## 1. 背景

与 #57（Base@384 train+all）同批，把 train+all 数据升级杠杆（ens10 +0.67 已验证）扩到 Base@512（valid TTA4 AUROC 0.99089）。结果**负**，本工单记负收口。

## 2. 训练配置

```bash
python train.py --arch convnext_base --img-size 512 --cache-root cache768 \
  --epochs 8 --batch-size 8 --accum 6 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --init-from convnext_base_512_cache768_1016_auprc0.8765.pt \
  --seed 42 --tag cb_512_all
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_base @ 512（cache768） |
| 热启动源 | `convnext_base_512_cache768_1016_auprc0.8765.pt`（fp16 瘦身，use=ema） |
| epoch | 8（首跑）→ 7（退档，step 对齐） |
| 显存 / 耗时 | 6.9G / 每 epoch ≈ 13 min |

## 3. 训练曲线（正常收敛，无发散）

| 版本 | train_loss 各 epoch |
|---|---|
| 8ep | 0.1185 / 0.1117 / 0.1059 / 0.1027 / 0.0929 / 0.0838 / 0.0818 / **0.0802** |
| 7ep | 0.1191 / 0.1111 / 0.1048 / 0.1010 / 0.0913 / 0.0824 / **0.0829** |

## 4. 验收：check_train_on_all.py（两个版本均 REVIEW）

| 版本 | Spearman | mean 漂移 | top200 | 结论 |
|---|---|---|---|---|
| 8ep（首跑） | 0.8753 | 12.6% | 0.950 | ❌ REVIEW |
| 7ep（退档） | **0.8638** | 13.2% | 0.955 | ❌ REVIEW（更差） |

判据 Spearman ≥ 0.90 均未过，且退 7ep 反而更差（0.8638 < 0.8753）→ **问题不在 epoch 过拟合**，而是 Base@512 架构/分辨率本身与 train-only 参考的排序一致性就差，调 epoch 无法解决。

## 5. 同批对比（train+all 家族）

| 成员 | arch/res | Spearman | 结论 |
|---|---|---|---|
| #54 Small@512 | small@512 | 0.9010 | ✅ 贴线过 |
| #57 Base@384 | base@384 | 0.9000 | ✅ 贴线过 |
| **#58 Base@512** | **base@512** | **0.8753 / 0.8638** | ❌ 唯一不过 |

**结论**：Base 架构的高分辨率（512px）train+all 不可靠，Base@512 train+all 方向证伪。Base 架构的 train+all 以 #57（Base@384）为准。

## 6. 交付物

- 无有效交付物（check REVIEW，不提交 LB）。
- 训练产物保留在云 `runs/cb_512_all/`（8ep）、`runs/cb_512_all7/`（7ep），权重不入库。

## 7. 结论

- Base@512 train+all 负结果，不提交 LB，建议队长关闭 #58。
- 教训沉淀：`check_train_on_all` 的 Spearman 门槛对架构/分辨率敏感——small@512、base@384 贴线过，base@512 明确不过且退档无效。高分辨率 base 的 train+all 应谨慎。
