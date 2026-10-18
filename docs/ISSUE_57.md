# Issue #57 — train+all 第二批：ConvNeXt-Base@384 全量重训（最强单模的数据升级）

> 状态：训练 + 验收 + 交付完成（2026-10-18）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #57 训练结果的完整归档；**权威内容以 GitHub issue #57 评论为准**。

---

## 1. 背景与动机

- **train+valid 全量重训已被线上证实是决定性杠杆**：ens10（ens9×0.80 + train+all 家族×0.20）→ **LB 98.82（第 14 名，+0.67）**，test AUROC 0.9815→0.9882，有效泛化差 −0.011 → ≈ −0.0044（台账 2026-10-18）。
- 本工单把这个杠杆扩到**当前最强单模**：Base@384（valid TTA4 AUROC 0.99147），做数据升级版 train+all 重训。

## 2. 前置修复（阻塞解除）

Base 瘦身权重格式为 `use=model`（有 `model`/`state_dict` 键、**无 `ema` 键**、无 auroc 指标），`train.py --init-from` 原逻辑会取 `sd["ema"]` 直接 KeyError 崩溃。

已修复并经队长合并：**PR #59**（`--init-from` 优先尊重 `use` 字段）。修复后本工单热启动正常加载，无任何告警。

## 3. 训练配置

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 8 --batch-size 12 --accum 4 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --init-from convnext_base_384_cache512_1015_auprc0.8644.pt \
  --seed 42 --tag cnxb_384_all
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_base @ 384（cache512） |
| 热启动源 | `convnext_base_384_cache512_1015_auprc0.8644.pt`（fp16 瘦身，use=model） |
| epoch | 8（train-on-all，+11.4% 数据，总 step 较 train-only 8ep 多 ~11%） |
| seed / lr | 42 / 3e-5（backbone）+ 1.5e-4（head），cosine + warmup |
| batch × accum | 12 × 4（有效 batch 48） |
| 训练集 | train + valid 全量 57245 张（阳性率 0.0323） |
| 显存 / 耗时 | 6.1G / 每 epoch ≈ 7.8 min，总 ≈ 63 min |

## 4. 训练曲线（每 epoch train_loss，正常单调收敛无发散）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| train_loss | 0.1139 | 0.1114 | 0.1035 | 0.0997 | 0.1006 | 0.0861 | 0.0853 | **0.0794** |

## 5. 验收：check_train_on_all.py（唯一干净的检查）

⚠️ train-on-all 模型的 valid 指标**无意义**（valid 已并入训练），绝不报告。

对 test 集（5855 张）与 train-only 参考模型（base_384 权重）做 TTA4 一致性：

| 指标 | 值 | 判据 | 结论 |
|---|---|---|---|
| Spearman 秩相关 | **0.9000** | ≥ 0.90 | ✅（贴线通过） |
| mean 概率相对漂移 | 8.2% | ≤ 25% | ✅ |
| top200 重叠率 | 0.950 | 参考值 | ✅ |
| **RESULT** | **PASS** | | |

**SUMMARY**: `cand=cnxb_384_all ref=convnext_base_384 spearman=0.9000 mean_rel_diff=0.082 top200_overlap=0.950 -> PASS`

> 备注：Spearman 0.9000 贴近阈值（与 #54 的 0.9010 同一量级），疑似 8ep（多训 11% step）的轻微过拟合迹象。本轮 check 通过未崩，最终以队长 LB 提交验收；若 LB 异常可退 7ep（step 对齐）重训。

## 6. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `cnxb_384_all_test_probs_tta4.csv` | test TTA4 概率，5855 行，**未校准** |
| `cnxb_384_all_fp16.pt` | fp16 瘦身权重（352.6MB，use=ema，train_on_all=True） |

原始 fp32 best.pt（约 705MB）保留在云 `runs/cnxb_384_all/best.pt`（权重不入库）。

## 7. 结论

- Base@384 train+all 8ep 训练健康收敛，check 一致性 PASS，交付物齐备。
- 待队长将其并入集成候选提交 LB，用线上分数做最终验收（本地无法验证）。
- 与 #58（Base@512 train+all，同批对照）共同构成第二批 train+all 成员。
