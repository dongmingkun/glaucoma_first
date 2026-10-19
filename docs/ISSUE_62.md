# Issue #62 — BiNet 双眼配对模型（阶段1 负结果收手）

> 状态：阶段1 负结果（valid < 0.990，不进入阶段2）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #62 阶段1 结果的归档；**权威内容以 GitHub issue #62 评论为准**。

---

## 1. 背景

BiNet 双眼配对：把「本眼 + 对侧眼」骨干特征并联，显式给出不对称信号（|f1-f2|、f1*f2），利用青光眼双眼不对称的临床先验。test 集成对结构（2655 对 + 545 单张）。此前只跑过 1ep（0.9886，未收敛）。

工单要求两阶段：阶段1 train-only 过 0.990 门槛 → 阶段2 才叠加 train+all。

## 2. 训练配置（队长拍板 convnext_base@384）

```bash
python train_binet.py --arch convnext_base --img-size 384 --cache-root cache768 \
  --epochs 4 --batch-size 6 --accum 4 --lr 1e-4 --head-lr 5e-4 \
  --init-from convnext_base_384_cache512_1015_auprc0.8644.pt --seed 42 --tag binet_v2
```

| 项 | 值 |
|---|---|
| backbone | convnext_base（FundusNet features，热启动 base_384 权重 use=model） |
| head | 双眼中枢 4·d+age_dim → 512 → 1 |
| 训练集 | 51396 对（含反向），valid 5849 |
| 耗时 | 每 ep ≈ 19 min，4ep 约 76 min |

（前置：train_binet.py `--init-from` use 字段修复，PR #63。）

## 3. 训练曲线（valid，train-only 干净可评估）

| epoch | AUROC | AUPRC | EMA AUROC | EMA AUPRC |
|---|---|---|---|---|
| 1 | 0.9815 | 0.7277 | 0.9853 | 0.7851 |
| 2 | 0.9828 | 0.7796 | 0.9874 | 0.8251 |
| 3 | 0.9875 | 0.8425 | **0.9882** | **0.8545** |
| 4 | 0.9877 | 0.8384 | 0.9878 | 0.8423 |

## 4. 结论

**峰值 EMA AUROC = 0.9882（ep3）< 0.990 门槛** → 按工单约定「阶段1 若 < 0.990 就停，不要跳过门槛直接做 train+all」，记负收手，不进入阶段2。

与此前 1ep 的 0.9886 一致——BiNet 在该配置下收敛到 ~0.988，未达成员门槛。双眼配对方向（至少 convnext_base@384 + 当前 head）负结果。

## 5. 交付物

- 无有效交付物（未达门槛，不产出成员）。
- 训练产物保留在云（exp 路径为 train_binet.py 硬编码的 `D:\glaucoma_work\runs\...`，权重不入库）。
