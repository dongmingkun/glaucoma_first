# #167 ConvNeXt-Large@512 train+all + Large 融合

> 状态：训练 ✅ / test TTA4 ✅ / 6 组候选 ✅ / 权重已入 Release ✅ / **未自动提交**（按工单要求）
> 全链路 08:50:27 → 14:03:12（5h12m45s），其中纯训练 18,504s（5h08m），TTA4 + 候选构建约 4.5 分钟。4090D 全程无 OOM（VRAM 7.8G）。

## 1. 训练

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Large（ImageNet 预训练）/ 512 |
| 数据 | train+all 57,245 行（正类 1,848 / 3.23%），无干净 valid |
| 超参 | 16 epoch，bs 4 × accum 12，lr 1e-4，head-lr 5e-4，randaugment 2 9，seed 123 |
| cache | `cache_all768`（与 img_size 512 配对，复现时必须一致） |
| 单 epoch 用时 | 1,138–1,183 s（均值 1,157s，无抖动） |
| loss | 0.39327 → 0.09268（16 ep 单调下降，末段 lr 1e-6） |
| 权重 | `cnxl_512_e16_ra_all_s123_last.pt`，1,576,423,247 B，md5 `71a271a642c9ee9165067a4dbd66f35b`（Release `cnxl_512_all_s123`） |

说明：train-on-all 没有干净验证集，按工单固定 `last.pt` + EMA，**不报 valid AUROC/AUPRC**（train-on-all 的 val 指标无意义）。

## 2. Large512 成员概率

`oof/cnxl_512_e16_ra_all_s123_test_tta4.csv`（md5 `5e359303`）：5,855 行 / 官方顺序 / 无 NaN / mean 0.03006 / pos@0.5 = 123。

与其他成员的相关性——这是本单的核心信息，**Large512 是最“陌生”的一路**：

| 对比 | Spearman |
|---|---|
| vs seed7（同家族） | 0.8444 |
| vs seed42 | 0.8281 |
| vs s123 | 0.8092 |
| vs Large384（#161） | 0.7687 |
| vs ens10 | **0.7171** |
| vs Swin_pair | **0.6215** |

分辨率从 384 提到 512 后与 L384 的 rho 仍有 0.7687，说明两档分辨率不是同一信号。

## 3. 候选（对照 = 重建的 98.87 配方 `blend167_ctl_987.csv`，md5 `0dc64aed`）

| 候选 | 构成 | mean | rho_ctl | top200 | md5 |
|---|---|---|---|---|---|
| C1 | ens10 + Large512（logit 0.10） | 0.0764 | 0.9837 | 196 | `7cd00a43` |
| **C2** | ens10 + Large512（logit 0.20） | 0.0701 | **0.9848** | 194 | `873f86e0` |
| C3 | ens10 + Large512（logit 0.30） | 0.0640 | 0.9820 | 194 | `19067843` |
| **V4+L512** | 0.35 ens10 + 0.30 RA_avg + 0.20 Swin_pair + **0.15 L512** | 0.5 | 0.9820 | 193 | `2e1b2023` |
| **双分辨率（推荐）** | 0.30 ens10 + 0.25 RA_avg + 0.20 Swin_pair + 0.15 L384 + **0.10 L512** | 0.5 | **0.9803** | 193 | `fb119ec9` |
| 激进 | 0.30 / 0.30 / 0.20 / **0.20 L512** | 0.5 | 0.9733 | 191 | `3be099fe` |

rank 变体（ens10 + L512 的 rank 融合 w10/20/30）一并入库：`df21d2ca` / `33ad86fa` / `6b221b6f`。

**判负线（rho ≥ 0.995 且 top200 = 200）全部未触发** —— 六组都是真位移，不是 no-op。
校验：11 个 csv 全部 `rows=5855 / order=True / bad=0`，**RESULT PASS**。

## 4. 推荐

**主推：双分辨率 `blend167_dual_w3025201510.csv`（`fb119ec9`）**

- 384 与 512 两档容量信号同时进融合，这是本单唯一“两个新信号叠加”的方案，位移最大且不靠单一成员；
- 保留 RA/Swin 结构（只把 0.35/0.30/0.20 微调到 0.30/0.25/0.20），不是把老结构推倒；
- 头部挪 7 个名额（top200=193），量级与 #161 的 e3（6 个）相当，属于可被榜接纳的位移。

**备选：C2 `..._ens10_logit_w20.csv`（`873f86e0`）** —— 最保守的新信号注入（只动 ens10 与 Large512 的配比，rho 0.9848），如果只想试水 512 分辨率而不改 V4 结构，选它。

**不推荐：激进版 `3be099fe`（0.30/0.30/0.20/0.20）** —— 位移最大（0.9733/191）但把 ens10 压到 0.30、L512 提到 0.20；历史上 0.35/0.25 的过冲版本线上只拿 98.85，这类压法风险已被证伪一次。

## 5. 交付物

| 类型 | 内容 |
|---|---|
| 权重 | Release `cnxl_512_all_s123`：`cnxl_512_e16_ra_all_s123_last.pt`（1.58GB，md5 `71a271a6`）—— 不入库，符合合规条款 |
| 概率 | `oof/cnxl_512_e16_ra_all_s123_test_tta4.csv` + 6 个候选 + 3 个 rank 变体 |
| 训练曲线 | `docs/cnxl_512_e16_ra_all_s123_log.csv` |
| 提交包 | `submit_167_dual_20261028/submission.csv`（= dual，md5 `fb119ec9`） |

## 6. 可复现性（吸取 #173 审计教训）

本单候选的成员概率**全部已在公开仓库**：swin_pair 的两个成员在 #175 补齐（`0a774bfd` / `23262b6f`），Large384 与 Large512 分别在 #169 / 本单入库。

```bash
python tools/rank_blend.py \
  --member ens10=0.30:submission_20251018_ens10_cal.csv \
  --member ra_avg=0.25:cnxb_512_e16_ra_test_best_tta4.csv,cnxb_512_e16_ra_test_last_tta4.csv \
  --member swin_pair=0.20:oof/swinb_384_e16_test_probs_tta4.csv,oof/swinb_384_all_v3_test_probs_tta4.csv \
  --member l384=0.15:oof/cnxl_384_e16_ra_all_s123_test_tta4.csv \
  --member l512=0.10:oof/cnxl_512_e16_ra_all_s123_test_tta4.csv \
  --out /tmp/dual_repro.csv
```

## 7. 缺口（如实）

1. **98.87 原始 CSV（md5 `4bf37b7c`）仍只在队长本地**，本表 rho/top200 是对重建配方算的；
2. 本单无 valid，故不提供 AUROC/AUPRC —— 与 #161 口径一致；
3. 提交与否由队长定，本单**未自动提交**，台账「计划」行已登记（md5 `fb119ec9`）。
