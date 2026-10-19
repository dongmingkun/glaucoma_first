# Issue #61 — train+all 家族种子扩充：cs_hi512_all seed 7 / 123

> 状态：训练 + 验收 + 交付完成（2026-10-19）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #61 训练结果的归档；**权威内容以 GitHub issue #61 评论为准**。

---

## 1. 背景

train+all 家族（#54/#55/#57）已被线上证实（ens10 98.82，+0.67）。本工单在已验证的 #54 配方（cs_hi512_all）上做**族内种子多样性**，产出 seed 7 / 123 两个成员，供集成增加多样性。

## 2. 训练配置（#54 配方，只换 seed）

```bash
python train.py --arch convnext_small --img-size 512 --cache-root cache768 \
  --epochs 8 --batch-size 8 --accum 6 --lr 3e-5 --head-lr 1.5e-4 \
  --loader workers --workers 3 --train-on-all \
  --init-from cs_hi512_v2.pt --seed {7,123} --tag cs_hi512_all_s{7,123}
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_small @ 512（cache768） |
| 热启动源 | `cs_hi512_v2.pt`（fp16 瘦身，use=ema） |
| epoch / seed | 8 / 7 与 123 |
| 显存 / 耗时 | 4.8G / 每 epoch ≈ 12 min，各约 1.6h |

## 3. 训练曲线（均正常收敛无发散）

| seed | train_loss 各 epoch |
|---|---|
| 7 | 0.1159 / 0.1184 / 0.1130 / 0.1041 / 0.0997 / 0.0921 / 0.0877 / **0.0844** |
| 123 | 0.1154 / 0.1196 / 0.1141 / 0.1040 / 0.0979 / 0.0898 / 0.0877 / **0.0825** |

## 4. 验收：check_train_on_all.py（--ref cs_hi512_v2，均 PASS）

| seed | Spearman | mean 漂移 | top200 | 结论 |
|---|---|---|---|---|
| 7 | **0.9089** | 12.4% | 0.955 | ✅ PASS |
| 123 | **0.9189** | 10.0% | 0.955 | ✅ PASS |

（均优于同配方 seed42 的 0.9010，种子多样性未引入排序劣化）

## 5. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `cs_hi512_all_s7_test_probs_tta4.csv` | seed7 test TTA4 概率，5855 行，未校准 |
| `cs_hi512_all_s7_fp16.pt` | seed7 fp16 瘦身（199.7MB） |
| `cs_hi512_all_s123_test_probs_tta4.csv` | seed123 test 概率 |
| `cs_hi512_all_s123_fp16.pt` | seed123 fp16 瘦身（199.7MB） |

## 6. 结论

- 两个 seed 均健康收敛、check PASS，作为 train+all 家族成员交付，待队长并入集成候选提交 LB。
- 与 seed42（#54）共同构成 small@512 train+all 的 3-seed 家族，族内多样性扩充完成。
