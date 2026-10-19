# Issue #69 — 冲刺第二批：Base@384 × train+all × 伪标签 × seeds 7/123

> 状态：训练 + 验收 + 交付完成（2026-10-19）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #69 训练结果的归档；**权威内容以 GitHub issue #69 评论为准**。

---

## 1. 背景

冲刺 99 第二阶段。Base@384 是当前最强单模（train-only valid 0.99147），其「数据升级 + 伪标签 + 种子多样性」版本应是当前最强成员群。本工单是 #65（seed42 版，check 0.9022）的种子扩充，产出 seed 7 / 123 两个成员。

## 2. 训练配置（#65 配方，只换 seed）

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512 \
  --epochs 7 --batch-size 10 --accum 5 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --extra-rows pseudo/pseudo_test_ens10.csv \
  --init-from convnext_base_384_cache512_1015_auprc0.8644.pt \
  --seed {7,123} --tag cnxb_384_all_pseudo_s{7,123}
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_base @ 384（cache512） |
| 热启动源 | `convnext_base_384_cache512_1015_auprc0.8644.pt`（use=model） |
| epoch / seed | 7（step 对齐）/ 7 与 123 |
| 额外数据 | 伪标签 852 行（`--extra-rows`，train rows 57245→58097） |
| 耗时 | 每 epoch ≈ 8.6 min，每 run 约 60 min |

## 3. 训练曲线（均正常收敛无发散）

| seed | train_loss 各 epoch |
|---|---|
| 7 | 0.1207 / 0.1122 / 0.1106 / 0.0967 / 0.0946 / 0.0861 / **0.0865** |
| 123 | 0.1121 / 0.1184 / 0.1083 / 0.1042 / 0.0943 / 0.0894 / **0.0882** |

## 4. 验收：check_train_on_all.py（--ref base_384，均 PASS）

| seed | Spearman | mean 漂移 | top200 | 结论 |
|---|---|---|---|---|
| 7 | **0.9045** | 6.6% | 0.950 | ✅ PASS |
| 123 | **0.9260** | 7.8% | 0.940 | ✅ PASS |

（均优于 seed42 版的 0.9022）

## 5. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `cnxb_384_all_pseudo_s7_test_probs_tta4.csv` | seed7 test TTA4 概率，5855 行，未校准 |
| `cnxb_384_all_pseudo_s7_fp16.pt` | seed7 fp16 瘦身（352.6MB） |
| `cnxb_384_all_pseudo_s123_test_probs_tta4.csv` | seed123 test 概率 |
| `cnxb_384_all_pseudo_s123_fp16.pt` | seed123 fp16 瘦身（352.6MB） |

## 6. 结论

- 两个 seed 均健康收敛、check PASS，作为 Base@384 × train+all × 伪标签家族的成员交付，待队长并入 ens14 提交 LB。
- 与 seed42（#65）共同构成 base@384 伪标签版 3-seed 家族，冲刺 99 的成员群扩充完成。
