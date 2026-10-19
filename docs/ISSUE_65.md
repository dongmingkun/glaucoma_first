# Issue #65 — 冲刺：train+all × 测试域伪标签（Base@384）

> 状态：训练 + 验收 + 交付完成（2026-10-19）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #65 训练结果的归档；**权威内容以 GitHub issue #65 评论为准**。

---

## 1. 背景

冲刺 99 阶段。在已验证的 train+all 杠杆上叠加**测试域伪标签**（转导式学习）：伪标签由当前最佳 ens10（test AUROC 0.9882）生成，213 正 + 639 负，共 852 行。队长机器在跑 small@512 版，云端跑更强的 Base@384 版（互补）。

## 2. 训练配置

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 7 --batch-size 10 --accum 5 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --extra-rows pseudo/pseudo_test_ens10.csv \
  --init-from convnext_base_384_cache512_1015_auprc0.8644.pt --seed 42 --tag cnxb_384_all_pseudo
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_base @ 384（cache512） |
| 热启动源 | `convnext_base_384_cache512_1015_auprc0.8644.pt`（use=model） |
| epoch / seed | 7（step 对齐）/ 42 |
| 额外数据 | 伪标签 852 行（`--extra-rows` 注入，train rows 57245→58097，pos 2061） |
| 显存 / 耗时 | 5.4G / 每 epoch ≈ 9 min，7ep 约 64 min |

## 3. 训练曲线（正常收敛无发散）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| train_loss | 0.1169 | 0.1127 | 0.1067 | 0.1025 | 0.0984 | 0.0866 | **0.0844** |

## 4. 验收：check_train_on_all.py（--ref base_384，PASS）

| 指标 | 值 | 判据 | 结论 |
|---|---|---|---|
| Spearman | **0.9022** | ≥ 0.90 | ✅ |
| mean 漂移 | 5.4% | ≤ 25% | ✅ |
| top200 重叠 | 0.945 | 参考 | ✅ |
| **RESULT** | **PASS** | | |

## 5. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `cnxb_384_all_pseudo_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 |
| `cnxb_384_all_pseudo_fp16.pt` | fp16 瘦身（352.6MB，use=ema） |

## 6. 结论

- Base@384 train+all × 伪标签 7ep 健康收敛、check PASS，交付待队长并入集成提交 LB。
- 与队长本地 small@512 伪标签版互补，构成冲刺 99 的两个伪标签成员。
