# 98.87 核验材料（回复 2026-11-10 材料请求）

> 目标：`submission.csv` = LB 98.87（ID 22205），MD5 `4bf37b7c062f7d45a9c5a8e287fff1eb`，
> SHA-256 `e4092e45518b95557d3256763de80f22b782978fbe4e155df543c85db556f8f7`，5855 行。
>
> 本文件逐条回答《给队友_98.87核验材料请求_20261110.md》的 §1–§5。
> 所有成员 CSV 与复现件都已入库；权重在既有 Release；一键核验：
> `python tools/verify_98_87_materials.py`（离线，不读 test 标签）。

## 0. 结论先行

| 问题 | 答复 |
|---|---|
| 98.87 能否复现 | **能**。`python tools/reproduce_98_87.py --check-md5` → MD5 **MATCH**，SHA-256 一致，5855 行，mean 0.500000 |
| 98.87 是几个模型 | **12 个 checkpoint**（ens10 槽位 6 + seedavg 2 + RA 2 + Swin 2）；ens10 的 “10” 是候选版本号，不是成员数 |
| 是否有 ≤500MB / T4 ≤100ms 的 98.87 包 | **没有**（§3）。公开成员权重合计 **3,695,849,219 B ≈ 3.70 GB**，Swin 两份 ckpt 还未公开 |
| 能否 0/100 不一致地输出 98.87 | **不能承诺**。98.87 是 12 个 checkpoint 的 rank 融合，任何子集都不会字节一致；且 Swin ckpt 缺失，无法重建该包 |
| 是否用 test 标签/伪标签 | **没有**。全部成员为官方 train 或 train+valid（train+all）训练，无 test 伪标签、无外部医学权重 |

三个必须先说清的口径（照抄公式会复现失败的原因）：

1. **`RA_avg` 是“先各自 rank 再 0.5/0.5 平均”**，不是 `rank((best+last)/2)`。
   后者是仓库 `tools/rank_blend.py` 的旧写法，会把 98.87 算成 `md5=740fb841`。
2. **`seedavg` 的输入是 `seed7_s42_test_tta4.csv`**（seed7/seed42 在 **logit 空间** 0.5/0.5 平均），
   不是“两个概率 CSV 的概率平均”（概率平均读法会得到 `md5=c94db095`）。
3. **`ens10` 是一个已固化的 CSV**，由 `ens9×0.80 + train+all 家族×0.20` 融合并校准得到；
   仓库没有 ens10 的逐成员构建脚本，无法用公开的 ens9/家族 CSV 逐字节重建（见 §1.2）。

## 1. 成员清单

### 1.1 最终公式与 rank 口径

```text
score = 0.60 * rank(ens10)
      + 0.10 * rank(seed7_s42_test_tta4.csv)
      + 0.15 * (0.5*rank(ra_best) + 0.5*rank(ra_last))     # 先 rank 再平均，不重新排名
      + 0.15 * rank(swin_pair_test_tta4.csv)
```

- `rank(v)`：在 5855 张测试图内取 `(rank-1)/(n-1)`（0-based，并列取平均秩）；
- 权重按上式顺序累加、不重新归一（`0.6+0.1+0.15+0.15=1.0`，顺序会影响 1e-6 边界）；
- 输出：CRLF + 表头 `Image_name,Probability,Pred_Label` + `%.6f`，
  `Pred_Label = score >= 0.5`（该列不参与评分）。

### 1.2 `ens10` 槽位（重要澄清）

- `ens10` **不是 10 成员集成**，而是：
  `ens10 = calibrate( 0.80 × ens9 + 0.20 × train+all 家族 )`；
  家族 = `cs_hi512_all` 与 `cs_hi512_all_ra` 各 0.5，因此两者在 ens10 内各占 **0.10**。
- `ens9`（单尺度 TTA4 + 校准，valid AUROC 0.99260）在 ens9 内的权重：
  `cb512 0.591 + hi512_v2 0.224 + cb384 0.141 + b512 0.044 = 1.000`。
- 权威字节件是 **`submission_20251018_ens10_cal.csv`**（MD5 `ce91bbf48d6c8e05d7f67a43d902a5ae`）。
  98.87 的 ens10 成员就是这份 CSV。
- **已知缺口（如实说明）**：仓库只保存了融合结果，没有 ens10 的逐成员构建脚本与校准参数。
  我们试过用已发布的 ens9 + 家族概率 CSV 重建 ens10：最优（0.6/0.4 概率平均）与权威件
  Spearman 仅 0.99978，按文档记录的 0.8/0.2 为 0.99913，**都不字节一致**。
  因此不能声称“从 6 个权重即可完整复现 ens10”，只能声称“用该 CSV 可完整复现 98.87”。

### 1.3 12 个 checkpoint

| 槽位 | 成员 | 架构 @ 分辨率 | 缓存 | 训练数据 | 伪标签 | 权重 |
|---|---|---|---|---|---|---|
| ens10 / ens9 | cb512 | ConvNeXt-Base @512 | cache768 | train-only | 无 | ens9 内 0.591 |
| ens10 / ens9 | hi512_v2 | ConvNeXt-Small @512 | cache768 | train-only（续训 6ep） | 无 | ens9 内 0.224 |
| ens10 / ens9 | cb384 | ConvNeXt-Base @384 | cache512 | train-only | 无 | ens9 内 0.141 |
| ens10 / ens9 | b512 | EfficientNet-B3 @512 | cache512 | train-only | 无 | ens9 内 0.044 |
| ens10 / family | cs_hi512_all | ConvNeXt-Small @512 | cache_all768 | train+all（57,245 行） | 无 | 0.10（家族内 0.5） |
| ens10 / family | cs_hi512_all_ra | ConvNeXt-Small @512 + RA(2,9) | cache_all768 | train+all（57,245 行） | 无 | 0.10（家族内 0.5） |
| seedavg | seed7 | ConvNeXt-Base @512 + RA(2,9) | cache_all768 | train+all，16ep，seed 7 | 无 | logit 平均 0.5 |
| seedavg | seed42 | ConvNeXt-Base @512 + RA(2,9) | cache_all768 | train+all，16ep，seed 42 | 无 | logit 平均 0.5 |
| RA_avg | ra_best | ConvNeXt-Base @512 + RA(2,9) | cache768 | train-only，16ep，seed 7 | 无 | 先 rank，0.5 |
| RA_avg | ra_last | 同上（last/EMA） | cache768 | train-only，16ep，seed 7 | 无 | 先 rank，0.5 |
| Swin_pair | swinb_384_e16 | Swin-B @384 | cache512 | train-only，16ep，seed 42 | 无 | 概率平均 0.5 |
| Swin_pair | swinb_384_e16_s7 | Swin-B @384 | cache512 | train-only，16ep，seed 7 | 无 | 概率平均 0.5 |

> `train+all` = 官方 train + valid 共 57,245 行一起训练（无干净 valid，权重取 EMA）。
> **不含 test 图、不含 test 伪标签**；`MixedDS` 里虽有 test 路径回退，仅用于伪标签实验，
> 本清单 12 个成员均未触发。

### 1.4 权重资产（SHA-256 / 字节 / 下载）

| 成员 | Release 资产 | 字节 | SHA-256 | Release |
|---|---|---|---|---|
| cb512 | `convnext_base_512_cache768_1016_auprc0.8765.pt` | 176,328,486 | `2b23c73c691cdef5b8a698fd40a781c0b8498c1bba2008fbaa76f3a244af53e5` | weights-1016 |
| hi512_v2 | `convnext_small_512_cache768_1016_auprc0.8752.pt` | 99,843,809 | `0212cd537f16e28ccec6bdf58836ef2b92325b3aeeb49e0da14a6db39473e16d` | weights-1016 |
| cb384 | `convnext_base_384_cache512_1015_auprc0.8644.pt` | 176,328,125 | `a389f41cb367132b4c23e2f9195bf15ac87c9df1214d359656f36ee63aff710e` | weights-1016 |
| b512 | `effnet_b3_512_cache512_1014_auprc0.8142.pt` | 23,544,137 | `693bf2fc0bfa96c75898d0c536ec6bc1a2bafc20e6a8a1d01195e8972d260e86` | weights-1016 |
| cs_hi512_all | `cs_hi512_all_fp16.pt` | 199,660,651 | `3e04d1906965742d8b13657b16e539063b05dab622c5272454939f57105f492e` | weights-1016 |
| cs_hi512_all_ra | `cs_hi512_all_ra_fp16.pt` | 199,662,799 | `ef92911ac1454c28b6f99a974e7deb9ff1ffab91d0ab082ab58dc13a64fe4362` | weights-1016 |
| seed7 | `cnxb_512_e16_ra_all_last.pt` | 705,015,055 | `51e7f3b1e6b71279d2f158cbe647a2979ed4d818f19808328cb28598cb48a980` | ra_all-1025 |
| seed42 | `cnxb_512_e16_ra_all_s42_last.pt` | 705,015,055 | `34c806b094ce0df09c3b75611031e3a25d151f2008c0c8ec37d0e46df7c47374` | ra_all_s42 |
| ra_best | `cnxb_512_e16_ra_best.pt` | 705,225,551 | `7ac2bb70bdc0405451c6f7a821b6c589f34cecd70f092b7248950dd7d229d296` | cnxb_512_e16_ra-1024 |
| ra_last | `cnxb_512_e16_ra_last.pt` | 705,225,551 | `842e38f67df2f51f427f992d6d52706c0fe4cebc882704f550dc44135f836364` | cnxb_512_e16_ra-1024 |
| swinb_384_e16 | **只有概率 CSV** `swinb_384_test_probs_tta4.csv`（158,120 B） | — | `58014e45398d77ae5ef74f836c26855ec2d3b825d4a4392b464b80ab91f01881` | weights-1016 |
| swinb_384_e16_s7 | **只有概率 CSV** `swinb_384_e16_s7_test_probs_tta4.csv`（158,120 B） | — | `e848a78a468dd7b14828ddd34683325e62321ce8c29927aacca9ec6221391f7f` | weights-1016 |

> 下载：`https://github.com/dongmingkun/glaucoma_first/releases/download/<release>/<资产>`。
> 未公开 ckpt：Swin stage1 / s7 的 `.pt` 不在任何 Release（全部 Release 资产已逐一核对），
> 只有 test/valid 概率 CSV。同结构 Swin-B@384 fp16 `swinb_384_all_v3_fp16.pt` 为 349,881,971 B，可作体积参照。

### 1.5 组合方式（已验证数值）

- `seedavg`：`seed7_s42_test_tta4.csv` = `sigmoid(0.5*logit(seed7) + 0.5*logit(seed42))`，
  与两文件 logit 平均的最大差 **5e-7**；若改成概率平均，最大差 0.0928，**不是同一个文件**。
- `RA_avg`：`ra_best` 用 `best.pt`（按 0.5×(AUROC+AUPRC) 选出），`ra_last` 用 `last.pt`（EMA）；
  两者**各自 rank 后** 0.5/0.5 平均，不重新排名。
- `Swin_pair`：`swinb_384_e16` 与 `swinb_384_e16_s7` **概率 0.5/0.5 平均**（maxdiff 5e-7），
  valid TTA4 AUROC **0.99240292**；`swin_pair_test_tta4.csv` MD5 `08250726…`。

### 1.6 预处理与 TTA

- 输入：官方缓存 JPEG（cache512 或 cache768，长边缩放后的方形缓存）；
- 推理：RGB → `resize(img_size, img_size, BILINEAR)` → ToTensor → ImageNet 均值/方差归一化；
  模型同时吃年龄辅助输入（`age_tensor`）；
- TTA4：identity + hflip + vflip + 双向翻转，**sigmoid 概率求平均**（`infer.py` / `tools/oof_predict.py`）；
- ens9 特意用**单尺度** TTA4：多尺度 TTA 在 AUROC 上轻微负作用（−0.00005~−0.00015）。

### 1.7 训练命令（可核对的成员）

```bash
# RA_best / RA_last（train-only，16ep，RandAugment 2/9，seed 7）
python train.py --arch convnext_base --img-size 512 --cache-root cache768 \
  --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --pretrained 1 \
  --randaugment 2 9 --loader workers --workers 8 --seed 7 --tag cnxb_512_e16_ra

# seed7 / seed42（train+all，16ep，RandAugment 2/9；seed42 如下，seed7 改 --seed 7）
python train.py --arch convnext_base --img-size 512 --cache-root cache_all768 \
  --train-on-all --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 42 \
  --loader workers --workers 8 --tag cnxb_512_e16_ra_all_s42

# Swin-B@384 stage1（train-only，16ep，seed 42；s7 为 --seed 7，其余相同）
python train.py --arch swin_base --img-size 384 --cache-root cache512 \
  --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --loader workers --workers 3 --seed 42 --tag swinb_384_e16

# TTA4 概率（以 test 为例）
python tools/oof_predict.py --ckpt runs/cnxb_512_e16_ra/best.pt \
  --csv glaucoma_first/test.csv --split-dir test --img-size 512 --tta 4 --use auto \
  --out cnxb_512_e16_ra_test_best_tta4.csv
```

ens9 的 4 个成员（cb512 / hi512_v2 / cb384 / b512）的训练配方见
[`docs/WEIGHTS.md`](WEIGHTS.md)（资产名内含 arch/img/cache/valid AUPRC）。

### 1.8 成员 CSV（CRLF，官方名序，5855 行）

| 槽位 | 文件 | 字节 | MD5 | SHA-256 |
|---|---|---|---|---|
| ens10 | `submission_20251018_ens10_cal.csv` | 158,120 | `ce91bbf48d6c8e05d7f67a43d902a5ae` | `ec6e0f477492fcba8ae8b17fb0cac21ec6040cc3e2294abdd4f9411d2bd63787` |
| seedavg | `seed7_s42_test_tta4.csv` | 158,120 | `2658ddfe1376a929e2107ca8c13c42ca` | `c0aed73cdeb5365e8d77419e074e2e63fc33ec2b736677e8bc767b8aa79cedde` |
| （seed7） | `cnxb_512_e16_ra_all_test_tta4.csv` | 158,120 | `b89f39809899f42a89f49c6bde6f0664` | `cdfd7bbd1a4c478964a222bb0e7fc24952ecc5a21b34b95a6ceb3ccb7372cfe1` |
| （seed42） | `cnxb_512_e16_ra_all_s42_test_tta4.csv` | 158,120 | `85e6eb57c3e0ffa3ee7e79f66594a3f4` | `fc367451831dd59fe853afb249d7c2538b8c3455c90f245288f3773542cd8184` |
| RA best | `cnxb_512_e16_ra_test_best_tta4.csv` | 158,120 | `6f8c7308afa7076b7397f52e61b07b82` | `3e639109a6d4e658bb8e123ab061d8d45ba9b8737461db19407071483ebaeafc` |
| RA last | `cnxb_512_e16_ra_test_last_tta4.csv` | 158,120 | `692c6e7b483784d7a55cf04d98b90c95` | `e7000f5db24a83e344da58b0ad1e67860fbce4283b99fb1707969e42ccb45b57` |
| Swin pair | `swin_pair_test_tta4.csv` | 158,120 | `0825072672651298f36e3e7991ab60ad` | `4d15c48af56cd92252819ab1c9335fbf8648c8799c6190d19e019aad6f3129d9` |
| **复现件** | `docs/audit_98_87/FINAL_REPRO_98_87.csv` | 158,120 | `4bf37b7c062f7d45a9c5a8e287fff1eb` | `e4092e45518b95557d3256763de80f22b782978fbe4e155df543c85db556f8f7` |

> 上表为 **CRLF** 版本哈希；仓库 blob 可能是 LF，内容逐字节相同。LF/CRLF 两套哈希见
> [`docs/audit_98_87/inputs_sha256.json`](audit_98_87/inputs_sha256.json) 与
> [`docs/audit_98_87/manifest_98_87.json`](audit_98_87/manifest_98_87.json)。

## 2. 精确融合代码与 CSV 复现

- 代码：[`tools/reproduce_98_87.py`](../tools/reproduce_98_87.py)（numpy-only，约 130 行，
  含 `norm_rank`、成员名序校验、CRLF/`%.6f` 输出）；
- 一键核验：[`tools/verify_98_87_materials.py`](../tools/verify_98_87_materials.py)
  （校验 8 份 CSV 的字节/行数/双口径哈希，再重算融合）；
- 复现件：[`docs/audit_98_87/FINAL_REPRO_98_87.csv`](audit_98_87/FINAL_REPRO_98_87.csv)；
- 结论 JSON：[`docs/audit_98_87/audit_98_87.json`](audit_98_87/audit_98_87.json)。

```bash
python tools/reproduce_98_87.py --check-md5 --out docs/audit_98_87/FINAL_REPRO_98_87.csv
# rows=5855  mean=0.500000  range=[0.002148, 0.999974]
# md5=4bf37b7c062f7d45a9c5a8e287fff1eb
# target=4bf37b7c062f7d45a9c5a8e287fff1eb -> MATCH
```

5855 行校验：行数 5855、官方 test 名序、无 NaN、score ∈[0,1]、
`Pred_Label == (Probability >= 0.5)` 全部一致；输出 SHA-256 `e4092e45…`。

**为什么别的脚本复算会差一点**：

| 读法 | 结果 |
|---|---|
| 本脚本（RA 先 rank 再平均 + 用 `seed7_s42_test_tta4.csv`） | **`4bf37b7c` MATCH** |
| RA 先概率平均再 rank（`tools/rank_blend.py` 现状） | `740fb841` |
| seedavg 用 seed7/seed42 概率平均 | `c94db095`（#186 实测） |

L1 候选（`submission_l1_large384.csv`，MD5 `e244a37e`）用同一口径也能精确复现，
说明“先 rank 再平均”是队长流水线的真实口径，而不是为 98.87 特调。

## 3. 是否有 ≤500MB / T4 ≤100ms 的 98.87 包

**没有。** 明确回答：

- **体积**：98.87 的 12 个 checkpoint 中，公开的 10 份合计 **3,695,849,219 B = 3.70 GB**；
  仅 ens10 的 6 个成员就 **875,368,007 B ≈ 875 MB**。
  Swin 两份 ckpt 未公开；按同结构 fp16 参照各 349,881,971 B，合计约 **4.40 GB**。
  即使全部转 fp16 瘦身，最小理论合计仍约 **1,699,706,251 B = 1.70 GB**（Base@512 四份各 ~168MB、
  Swin-B 两份各 ~176MB），远超 500MB 红线。
- **延迟**：12 个 checkpoint × TTA4 = **48 次前向/图**。3 模型 TTA1 在 RTX 5060 上 52–58ms/图，
  线性外推 ≥600ms/图；98.87 包本身未实测（也不可能有合规包），结构上不可能 ≤100ms。
- **0/100 一致**：没有可交付的 98.87 包，无法做该比对；单个/子集模型更不会与 rank 融合结果逐字节一致。
- **现有合规包只能到 98.72**：`compliant_pkg-1028` = cb384 + cs_hi512_all + cs_hi512_all_ra(1028 版)，
  475,820,470 B（3 模型），复现 MD5 `b664…`，**不能输出 98.87**。
  该包已按组委会模板 T4 复测：F1 平均 **96.61ms**（100 图×3 轮，不含加载/预热）、
  F2 二值/排序不一致 0/100、F3 权重 475.82MB（≤500MB）；
  因数据集仅含 cache512、无原图，官方 CSV 级复现仍未闭环，详见 [`COMPLIANCE_98_72.md`](COMPLIANCE_98_72.md)。

给组委会的**体积豁免**口径：完整成员权重 **3,695,849,219 B（公开 10 份，含 fp32/EMA 原版）**；
Swin 两份补齐后约 **4,395,613,161 B**；若全部 fp16 瘦身约 **1.70 GB**。
需要人工核验时请按上表逐一比对 SHA-256。

## 4. 文件交付

- **材料 Release**：`materials-98-87-20261110`
  （`https://github.com/dongmingkun/glaucoma_first/releases/tag/materials-98-87-20261110`）
  附件含 7 份成员 CSV、复现件、`manifest_98_87.json`、`reproduce_98_87.py`、`verify_98_87_materials.py`。
- **直链模板**：`https://github.com/dongmingkun/glaucoma_first/releases/download/materials-98-87-20261110/<文件名>`。
- **权重**：不重复上传（单份 176–705MB），按 §1.4 的既有 Release 下载，SHA-256 已列。
  4 份 705MB 文件为 fp32 + EMA 原版；`last.pt` 即训练产出的 EMA 权重。
- **缺件**：Swin stage1 / s7 的 `.pt` 不存在的公开副本 —— 只有概率 CSV。
  如需完整重建 Swin 成员，需向产出方（队长机器）索取 `swinb_384_e16/best.pt` 与
  `swinb_384_e16_s7/best.pt`。

## 5. 时间与合规声明

- 本回复日期：**2026-11-10**；材料与复现均在 CPU 本地完成，无 GPU 训练、无平台提交。
- 未使用 test 标签、未使用 test 伪标签、未做“测试集概率查表”；98.87 的每个输入都是模型概率。
- 未修改任何已提交 CSV；复现件是脚本重算产物，与目标 MD5/SHA-256 完全一致。
- 若组委会要求“包内模型与该 CSV 对应”，本仓库只能提供 **12 checkpoint 的清单 + 既有权重**，
  **没有 ≤500MB 的合规打包版**；请走人工核验/体积豁免，而不要用 98.72 的 3 模型包冒充 98.87。

## 附：一键核验输出（本机实测）

```text
$ python tools/verify_98_87_materials.py
target: md5=4bf37b7c062f7d45a9c5a8e287fff1eb sha256=e4092e45... rows=5855
PASS  ens10 / seedavg / seed7 / seed42 / ra_best / ra_last / swin_pair / fin_repro
PASS  reproduce: md5=4bf37b7c... sha256=e4092e45...
weights: published 3695849219 B (3524.6 MiB) in 10 assets; Swin checkpoints unpublished
compliant_98_87: False
ALL PASS
```
