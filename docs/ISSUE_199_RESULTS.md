# Issue #199 结果记录 — ConvNeXtV2-Base@512 阶段 1（Kaggle T4）：训练 NaN 发散，未完成，建议判负关闭或授权缓解重跑

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/199 。执行平台：Kaggle T4
> （工单允许的 ≥16GB 之外按 #193 先例用 T4 15GB；指派人账号 linglingyusc）。
> **结论先行：配方在 lr 1e-4 下于 ep3 中段发散成 NaN（单 batch 爆炸→全局污染），16ep 无法完成；
> 已有 2 个 epoch 的部分证据（valid AUROC 0.611/0.703，EMA 0.703→0.367）远低于 0.9920 硬门槛。
> 按工单"失败也要归档，不占用下一条路线"如实归档；是否授权一次缓解重跑由队长裁定。**

## 执行档案

| 项 | 值 |
|---|---|
| 命令 | 模型超参按工单（`--arch convnextv2_base --img-size 512 --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --randaugment 2 9 --pretrained 1 --seed 42 --validation-tta 4 --save-every-steps 200`）；runner 因 T4 使用 `--loader workers --workers 4`，cache 为重建路径 `/kaggle/working/cache512` |
| 平台 | Kaggle script kernel `g199-cnxv2b512-stage1` v3，Tesla T4×2（单卡训练），runner `tools/kaggle_199_runner.py`（PR #203） |
| 数据 | 用户数据集 `linglingyusc/glaucoma` 的 cache_all（57245）按官方文件名前缀精确重建 cache512（51396/5849/5855 计数硬校验通过；train/valid 图像与 #193 成功交付所用同一数据集） |
| 耗时 | 01:07 → 07:09 ≈ 6h02m（训练 5.92h 后 exit=1） |
| 速度 | 0.86 s/micro-batch（fp32 on T4）→ **2.07h/epoch → 16ep ≈ 33h ≈ 3 场会话**（重要物流发现：即使不发散，Kaggle T4 也需 3 场续传） |

## 训练时间线（完整曲线：本 PR `log.csv`）

```
ep1: train_loss 0.50535  val AUROC 0.61114  EMA 0.70272（7447s）
ep2: train_loss 0.46801  val AUROC 0.70338  EMA 0.36660（7302s）  ← EMA 已崩
ep3: train_loss 0.43~0.45 区间正常下降至 it5950 loss=0.4547
     it6000 起 loss=nan（单 batch 爆炸→全局污染），持续 nan 至 epoch 末
ep3 末 evaluate() 遇 NaN 概率崩溃退出（exit=1，traceback 见 evidence 日志）
```

## 诊断（三条证据链指向同一结论：模型发散，非数据/管线问题）

1. **valid 管线正确**：valid AUROC 从 0.611 → 0.703 缓慢爬升（若 valid 图/标签错位，应恒定 ~0.5 不动）；
   train/valid 行数与正例数与官方完全一致（51396/1657、5849）；
2. **收敛速度异常**：V1 同配方同 lr（#128）ep1 valid AUROC 0.96884、train_loss 0.401；V2 仅 0.611/0.505
   ——**ConvNeXtV2（GRN）在 lr 1e-4 下病态**：学得慢、EMA 在 ep2 即塌到 0.367（低于随机），
   直至 ep3 全局 NaN；
3. **NaN 起点**：ep3 it5950（loss=0.4547 正常）→ it6000（nan），50 个 micro-batch 窗口内单点爆炸；
   `--save-every-steps 200` 的 resume.pt 同样被污染（最后一次落盘在 NaN 区间内）。

## 与工单门槛的关系（如实）

- **门槛一未完成**：0.9920 硬门槛要求 16ep 完整 valid TTA4；本记录只有 2 个 epoch 的病态中间值
  （`gate1.json` AUROC 0.367 系 last.pt=ep2 EMA 的产物，**标记无效**，不作为门槛判定依据）；
- **按工单精神判负**：预注册配方（lr 1e-4，不扫参）无法完成 16ep——这本身就是阶段 1 的否决性证据；
- **阶段 2 不启动**（前置"阶段 1 通过"未满足）。

## 失败归档与下一步（由队长二选一）

1. **关闭 #199（推荐）**：V2/GRN + 本任务 3.2% 正例 + lr 1e-4 组合不稳定，且 Kaggle T4 完成本配方需
   3 场会话（物流成本过高）；失败原因已归档，不占用下一条路线；
2. **授权一次缓解重跑**（需要队长明确批准，属预注册外改动）：可选项
   `--lr 5e-5 --head-lr 2.5e-4`（#193 Large 所用、更稳）+ 梯度裁剪 + `bs4×accum12`，
   且必须在 4090 实例跑（T4 fp32 33h 不现实）。**未经批准不自行改配方。**

## 交付物（本 PR）

- `docs/ISSUE_199_RESULTS.md`（本文件）；
- `cnxv2b_512_e16_ra_log.csv`（2 个 epoch 的完整曲线）+ `cnxv2b_512_e16_ra_gate1.json`（无效产物标记）；
- `docs/audit_199_evidence.txt`（runner 日志关键节选：cache 重建校验、NaN 起点、traceback）；
- runner 代码不在本 PR；以 #203 合并版 `tools/kaggle_199_runner.py` 为准。
- 完整 runner.log（3.4MB）在 Kaggle 内核 output：`linglingyusc/g199-cnxv2b512-stage1` v3。

## 成本

- Kaggle T4 配额：约 6h（周额度 30h 的 20%）；
- 无线上提交（预注册纪律：不拿 test 做任何选择）。
