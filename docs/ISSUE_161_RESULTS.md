# #161 ConvNeXt-Large@384 train+all 真机结果 + V4 融合

> 状态：训练 ✅ / test TTA4 ✅ / 候选 ✅ / 权重已入 Release ✅ / **未自动提交**（按工单要求）
> 机器：RTX 4090 24GB（云端实例，未占本机 5060）

## 1. 结论速览

| 项 | 结果 |
|---|---|
| 训练 | `convnext_large` @384，train+all，16ep，**4h43m**（22:55:34 → 03:39:09） |
| loss | 0.39168 → **0.10104**（16ep 单调下降，无发散） |
| 权重 | `runs/cnxl_384_e16_ra_all_s123/last.pt` **1,576,423,247 B** → Release `cnxl_384_all_s123` |
| Large TTA4 | 5855 行 / 官方顺序 / 无 NaN / mean **0.03352** / pos@0.5 **158** |
| 与 RA 家族距离 | vs seed7 **0.8311**、seed42 **0.7990**、s123 **0.8286**（RA 内部 0.887–0.900） |
| 最佳候选 | `blend161_v4l_w35302015`：rho_ctl **0.9853**、top200 **194**、md5 `e3ddd565` |
| 判负线 | rho≥0.995 且 top200=200 —— **未触发** |

## 2. 训练（与工单命令一致，未降级）

```bash
python train.py --arch convnext_large --img-size 384 --cache-root cache_all768 \
  --train-on-all --epochs 16 --batch-size 4 --accum 12 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 123 \
  --loader workers --workers 8 --tag cnxl_384_e16_ra_all_s123 \
  --out-dir runs/cnxl_384_e16_ra_all_s123
```

- **没有 OOM**：bs4×accum12 在 24G 上 VRAM 峰值 **6.2G**，未触发 bs2×accum24 降级；
- train+all：51,396 → **57,245**（+11.4%），每 ep 14,311 步；
- 单 ep 1,015–1,074s，16ep 合计 **16,821s**；含 TTA4 与落盘墙钟 **17,015s（4h43m）**，快于工单预估 5–7h；
- train+all 无干净 valid，**不报 valid AUROC**，固定 **last.pt + EMA**（与 #138/#145/#157 同口径）。

训练曲线见 `docs/cnxl_384_e16_ra_all_s123_log.csv`。

## 3. test TTA4 校验

| 检查 | 结果 |
|---|---|
| 行数 | 5,855 ✅ |
| 顺序 | 与 `glaucoma_first/test.csv` 完全一致 ✅ |
| NaN / 越界 | 0 ✅ |
| mean / pos@0.5 | 0.03352 / 158 |
| 范围 | [0.0002, 0.9980] |

### 与其它信号的距离（Spearman / top200 重叠）

| 对比 | Spearman | top200 |
|---|---|---|
| Large384 vs RA seed7 | 0.8311 | 178 |
| Large384 vs RA seed42 | 0.7990 | 179 |
| Large384 vs RA s123 | 0.8286 | 178 |
| ens10 vs Large384 | 0.80919 | 177（重叠 0.885） |

RA 家族内部（seed7/42/s123）0.887–0.900，**Large384 落在 0.80–0.83**，比 RA 换 seed 远得多、比 #157 新 Swin（0.70–0.78）近一些 —— 确实是**新的容量信号**，不是又一条 rho≈0.99 的打平线。

## 4. 候选（全部 test-only，对照 = 重建的 98.87 配方）

对照 `blend161_ctl_987.csv` = `0.60 ens10 + 0.10 seedavg + 0.15 RA_avg + 0.15 Swin_pair`（rank 空间）。

| 候选 | 文件 | 构成 | mean | rho_ctl | top200 | md5 |
|---|---|---|---|---|---|---|
| C1 | `..._test_tta4_ens10_logit_w10.csv` | ens10 + Large 0.10（logit） | 0.0765 | 0.9828 | 196 | `e6de3d6b` |
| C2 | `..._ens10_logit_w20.csv` | ens10 + Large 0.20 | 0.0703 | 0.9847 | 195 | `5dd1f33e` |
| C3 | `..._ens10_logit_w30.csv` | ens10 + Large 0.30 | 0.0643 | 0.9849 | 193 | `f1f19b14` |
| **V4+Large（推荐）** | `blend161_v4l_w35302015.csv` | 0.35 ens10 + 0.30 RA_avg + 0.20 Swin_pair + **0.15 Large** | 0.5（rank） | **0.9853** | **194** | `e3ddd565` |
| 激进版 | `blend161_agg_w30302020.csv` | 0.30 / 0.30 / 0.20 / **0.20 Large** | 0.5（rank） | 0.9793 | 190 | `225ec768` |

- 全部 5,855 行 / 官方顺序 / 无 NaN / [0,1]：**RESULT PASS**（构建脚本内置校验）；
- C1–C3 是 logit 空间融合，mean 保持 0.06–0.08，pos@0.5 会漂，**只用于榜单排序**；
- V4+Large 用 rank 融合，mean 归一到 0.5，与 98.87 的提交口径一致。

## 5. 推荐

**首选 `blend161_v4l_w35302015.csv`（md5 `e3ddd565`）**
- 保留 V4 的 RA/Swin 结构不动，只把 Large 作为第四路 0.15 插入 —— 单变量、可控；
- rho_ctl 0.9853 / top200 194：主体稳定，但真的挪了 6 个头部名额；
- 与 98.87 的差异大于 #157 的 B1（0.9933/198），**是本轮最值得投一次的信号**。

**次选（更激进）`blend161_agg_w30302020.csv`（`225ec768`）**：Large 权重 0.20、ens10 降到 0.30。**风险更高**：ens10 是最强单路（权重被压到 0.30），若 Large 泛化弱于预期会掉分。

**不建议**：C1–C3（丢掉了 RA_avg+Swin_pair 结构，等于退回 ens10 单路 + Large）。

**若只有一个配额**：优先本单 V4+Large，其次等 #167 的 Large@512（512 分辨率信号通常比 384 更强，正在训）。

## 6. 合规 & 产物

- 仅 ImageNet 预训练（`--pretrained 1`），**非医学预训练**；
- 仅官方 train/valid（train+all 模式，无 test 伪标签）；
- Large 权重**不入 git**：Release `cnxl_384_all_s123` / `cnxl_384_e16_ra_all_s123_last.pt`（1.47 GiB）；
- 大集成只用于榜单；最终 ≤500MB 包另走 `compliant_pkg`；
- 本 PR 文件：Large TTA4 概率 1 + ens10 融合 3 + rank 融合 3（含对照）+ 本文件 + 训练曲线 + `SUBMISSIONS_LOG` 计划行。

## 7. 缺口与事故（如实）

1. **98.87 原始 CSV 缺失**：`FINAL_SUBMIT_20261028/submission.csv`（md5 `4bf37b7c`）只在队长本地，仓库/云端都没有 —— 本表 rho/top200 是对**重建配方**算的，不等于对原文件算的绝对值。若能提供原文件，可一键重算。
2. **报告时间晚于 08:00 截止**：训练与候选在 **03:39** 就已全部跑完（产物在云端 `/root/gf145/oof/`），但写文档/开 PR/贴评论因本地会话中断拖到 **08:5x**。训练本身没超时，**是交付动作超时**，这条如实记在这里。
3. **磁盘**：overlay 30G，161 的 last.pt+best.pt 共 3.0G。权重传到 Release 验证成功后，已删 161/157 ckpt 与 soup3 目录，腾出空间给 #167 的 512 训练（overlay 剩 ~8G）。
4. **权重上传**：`curl --data-binary @1.57GB` 在云端报 `out of memory`（curl 把整个文件读进内存），改 `curl -T` 流式上传才成功 —— 记进 runbook，以后 >700MB 一律用 `-T`。

## 8. 下一步

- #167（Large@512）已在 #161 训练结束后启动，走同一条链（训练 → TTA4@512 → 候选 → 校验），产物目录 `runs/cnxl_512_e16_ra_all_s123`；
- 提交与否由队长定；本单**未自动提交**。
