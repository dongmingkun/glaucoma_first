# #145 结果：train+all Base@512 RA seed42（seed 平均 + ens10 融合，test-only）

> 训练与推理已在 **4090 24GB** 上真实执行完毕；test-only 家族无干净 valid，**不报任何 valid AUROC/AUPRC**。
> 线上是否提分只能由一次平台提交回答（每日 1 次配额，候选不自动提交）。

## 一、结论摘要

| 项 | 结果 |
|---|---|
| seed42 训练 | ✅ 完成 16ep（`runs/cnxb_512_e16_ra_all_s42/{last,best}.pt`，均 EMA） |
| test TTA4 | ✅ `oof/cnxb_512_e16_ra_all_s42_test_tta4.csv`，**5855 行 / 官方顺序 / 无 NaN / [0, 1]** |
| 与 seed7 的 test Spearman | **0.8868**（同配方换 seed；**低于 #145 预期的 >0.95**，见第四节说明） |
| check_train_on_all | ⚠️ 只做了**分布自检**（本实例无 `#128` 的 train-only `runs/cnxb_512_e16_ra/last.pt`，无 ref 即无正式 PASS/REVIEW 判定） |
| 融合候选 | ✅ 11 个（seed 平均 + ens10 w=0.10/0.15/0.20；seed42 直接融 ens10 logit/rank w=0.1/0.2/0.3） |
| 权重 | ✅ Release `ra_all_s42`：`cnxb_512_e16_ra_all_s42_last.pt`（705,015,055 B） |
| 追加实验（soup） | ✅ seed7+seed42 权重平均（0.5/0.5）已跑通，2 分钟出 `soup` 单成员 + 6 个融合候选（见第八节） |
| 线上验证 | ⏳ **待提交回填**（见第六节；失败不降低当前最好 98.84） |

## 二、训练

```bash
python train.py --arch convnext_base --img-size 512 --cache-root cache_all768 \
  --train-on-all --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 42 \
  --loader workers --workers 8 --tag cnxb_512_e16_ra_all_s42 \
  --out-dir runs/cnxb_512_e16_ra_all_s42
```

- 环境：RTX 4090 24GB，torch 2.10+cu126，CUDA 可用；`cache_all768` = **57245 / 57245 / 5855**（软链合成，未复制图像）
- 数据：train-on-all 生效 `[train-on-all] rows 51396 -> 57245 (+11.4%)`，`pos=1848 (0.0323)`
- 耗时：**约 2.91h**（653+635+626+634+621+640+643+639+664+668+677+669+662+663+667+671 s），单 epoch ≈ 11 min，VRAM 稳定 6.9G
- 训练损失（`cnxb_512_e16_ra_all_s42_log.csv`）：ep1 **0.38795** → ep8 0.13890 → ep16 **0.10049**，单调下降无发散
- 产物：`last.pt` / `best.pt` 各 705,015,055 B（均为 EMA）；已上传 Release **`ra_all_s42`**

## 三、test TTA4 与分布自检

| 项 | 值 |
|---|---|
| 行数 | **5855**（与 `glaucoma_first/test.csv` 顺序逐行一致） |
| 越界 / NaN | **0** |
| 概率范围 | `[0.000035, 0.999684]` |
| 分布（cand=seed42） | mean **0.05075**，pos_rate@0.5 = **4.70%**（275/5855），p50 0.00034，p90 0.02469，p99 **0.99864** |
| 同类对照（seed7） | mean 0.05319，pos_rate@0.5 = 4.82%（282/5855） |

两 seed 的分布形状高度接近（均值差 4.6%，正例数 275 vs 282），说明 seed42 训练正常收敛、没有塌缩。

> ⚠️ `check_train_on_all.py` 未拿到 train-only 参照（本克隆机上没有 `#128` 的 `runs/cnxb_512_e16_ra/last.pt`），
> 因此只输出 `No --ref given: distribution self-check only`，**没有 PASS/REVIEW 结论**。
> 该脚本给出的真实判定（Spearman/漂移/top200 vs train-only）需在有 ref 的机器上补跑：
> `python check_train_on_all.py --ckpt runs/cnxb_512_e16_ra_all_s42/last.pt --ref runs/cnxb_512_e16_ra/last.pt --tta 4 --cache-root cache_all768`

## 四、与 seed7 的一致性（#145 验收项）

直接对两份 member 概率算 Spearman（不做平均，避免虚高）：

| 对 | Spearman | top200 重叠 |
|---|---|---|
| seed42 vs seed7 | **0.8868** | 191 / 200 |
| seed 平均 vs seed7 | 0.9699 | — |
| seed 平均 vs seed42 | 0.9705 | — |
| ens10 vs seed7 | 0.8099 | — |
| ens10 vs seed42 | 0.8161 | — |
| ens10 vs seed 平均 | **0.8339** | 190 / 200 |

**如实解读**：同配方换 seed，member 级 Spearman 只有 **0.8868**，低于 #145 写的「应 >0.95」。
这与历史记录一致（#51 的 RA 三 seed 排名相关 0.9118–0.9404）：眼底图 test 集里绝大多数是易分负例，seed 差异主要体现在负例内部排序。
含义有两面：① **不能**把 seed 平均当成「几乎不变的保险」；② 两个 seed 确实提供了额外多样性（平均后与 ens10 的相关 0.8161→0.8339，更贴近主集成），对融合是有利的。

## 五、候选清单（全部 test-only，5855 / 官方顺序 / 无 NaN）

| # | 文件 | 含义 | mean | pos@0.5 | 与 98.84（seed7 w10）Spearman | top200 |
|---|---|---|---|---|---|---|
| 0 | `cnxb_512_e16_ra_all_s42_test_tta4.csv` | seed42 单模 TTA4 | 0.05075 | 275 | 0.8327 | 191 |
| 1 | `seed7_s42_test_tta4.csv` | 纯 seed 平均（0.5/0.5 logit） | 0.05157 | 278 | 0.8563 | 191 |
| 2 | `seed7_s42_ens10_logit_w10.csv` | ens10×0.9 + seed平均×0.1 | 0.07765 | 399 | **0.9998** | **200** |
| 3 | `seed7_s42_ens10_logit_w15.csv` | ens10×0.85 + seed平均×0.15 | 0.07525 | 390 | 0.9995 | 200 |
| 4 | `seed7_s42_ens10_logit_w20.csv` | ens10×0.8 + seed平均×0.2（= 0.1 seed7 + 0.1 seed42） | 0.07302 | 377 | 0.9985 | 200 |
| 5 | `cnxb_512_e16_ra_all_s42_test_tta4_ens10_logit_w10.csv` | ens10×0.9 + seed42×0.1 | 0.07724 | 400 | 0.9991 | 199 |
| 6 | `..._ens10_logit_w20.csv` | ens10×0.8 + seed42×0.2 | 0.07234 | 368 | 0.9967 | 199 |
| 7 | `..._ens10_logit_w30.csv` | ens10×0.7 + seed42×0.3 | 0.06816 | 352 | 0.9910 | 198 |
| 8–10 | `..._ens10_rank_{w10,w20,w30}.csv` | rank 空间融合（探索，mean≈0.5，不建议提交） | ≈0.4999 | — | 0.90 量级 | 193 |

md5（前 8 位）：seed42 `85e6eb57`｜seed 平均 `2658ddfe`｜avg_w10 `789aedb8`｜avg_w15 `6a753bde`｜avg_w20 `33354c5a`｜s42_w10 `26f41fc4`｜s42_w20 `fb20255a`｜s42_w30 `aeaefaec`

## 六、提交建议与线上验证

**推荐顺序**（同口径校验过：`seed7` 的 98.84 反解 = `sigmoid(0.9·logit(ens10)+0.1·logit(seed7))`，max_err 5e-7）：

0. **（soup 实验后新增，见第八节）首选 `cnxb_512_e16_ra_all_soup_test_tta4_ens10_logit_w10.csv`** — ens10×0.9 + soup×0.1，rho(98.84)=**0.9992**、top200 **200/200**、均值漂移仅 1.8%；既检验 soup 又有最小的排序扰动。
1. `seed7_s42_ens10_logit_w10.csv`（#2）— 与 98.84 **同结构同总权重**（家族 0.1），只是把 0.1 摊成 0.05+0.05。排序 rho=0.9998、top200 200/200，是「概率空间降方差」最干净的检验，预期落在 98.83–98.85。
2. `seed7_s42_ens10_logit_w20.csv`（#4）— 家族总权重翻倍（0.1 seed7 + 0.1 seed42），对应 SUBMISSIONS_LOG 里「提高家族权重」的下一步；rho=0.9985、top200 200/200。
3. 第三顺位 `..._s42_logit_w30.csv`（#7，家族 0.3 单 seed42）更激进（rho=0.9910），仅在前面无增益时再试。

提交动作（人为，平台每日 1 次配额，**候选不自动提交**）：

```bash
# 1) 生成平台要求的文件名（仓库根的 package_submission.py，已内置 5855/顺序/范围校验）
python package_submission.py --csv seed7_s42_ens10_logit_w10.csv \
  --name issue145_seed7_s42_w10 --expected 98.84
# -> D:/glaucoma_work/submissions/<date>_issue145_seed7_s42_w10/submission.csv + manifest.md
# Linux/云端无 D: 盘时追加 --out-root submit_ready
# 2) 登录比赛平台上传该目录下的 submission.csv，记录 ID 与分数
```

**线上验证回填（待办）**

| 项 | 值 |
|---|---|
| 提交文件 | ⏳ |
| 平台 ID | ⏳ |
| 线上分数 | ⏳ |
| 对比 98.84 | ⏳ |
| 判定（是否纳入正式集成） | ⏳ |

> 判定规则：≥98.86 → 纳入并考虑继续提家族权重；98.83–98.85 → 与 98.84 等价（方差降低但未提分）；<98.83 → seed 平均无收益，回退 98.84，本单按「失败不降低」收口。

## 七、交付物与复现

- 权重：Release `ra_all_s42` → `cnxb_512_e16_ra_all_s42_last.pt`（672 MiB）
- 概率：本目录根下 11 个 CSV（见第五节）+ `cnxb_512_e16_ra_all_s42_log.csv`
- 复现：`bash tools/pipeline_145.sh`（一键：训练→TTA4→check→融合；`PYTHON=/root/miniconda3/bin/python`）
- 已知偏差：本次运行用的是 `issue145-seed-avg` 分支合并前的版本，其 seed7 输入默认写 `oof/cnxb_512_e16_ra_all_test_tta4.csv`（仓库实际放在根目录），导致首轮跳过 seed 平均。已按根目录路径手工补跑（本文数据即补跑后产出）；main 上的 `tools/pipeline_145.sh` 现已默认根目录 + `oof/` 回退，后续按手册运行不会复现该问题。

## 八、追加实验：seed7 + seed42 model soup（队长 #145 评论 2026-10-26 14:09Z，PR #148）

**目的**：与概率空间平均（`tools/seed_average.py`）互补——权重空间平均只需推理**一个**模型，可能把两个 seed 的噪声一起平滑。

**执行（同一台 4090，全程 2 分钟）**

```bash
python tools/seed_soup.py --ckpts runs/cnxb_512_e16_ra_all/last.pt \
                                  runs/cnxb_512_e16_ra_all_s42/last.pt \
                          --weights 0.5 0.5 --out runs/cnxb_512_e16_ra_all_soup/last.pt
python tools/oof_predict.py --ckpt runs/cnxb_512_e16_ra_all_soup/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --use ema --out oof/cnxb_512_e16_ra_all_soup_test_tta4.csv
python tools/fusion_ens10.py --ens10 submission_20251018_ens10_cal.csv \
  --new oof/cnxb_512_e16_ra_all_soup_test_tta4.csv --out-dir oof
```

- soup 产出 `runs/cnxb_512_e16_ra_all_soup/last.pt`（704,988,303 B，`arch=convnext_base img=512 weights=[0.5,0.5]`，同时平均 `ema` 与 `model`）
- 时间线：22:30:56 开始 → 22:31 soup 落盘 → 22:32:55 TTA4+融合+校验全部结束

**soup 单成员校验**

| 项 | soup | seed7 | seed42 |
|---|---|---|---|
| 行数 / 官方顺序 | 5855 / ✅ | 5855 / ✅ | 5855 / ✅ |
| NaN / 越界 | 0 | 0 | 0 |
| 概率范围 | [0.001637, 0.941034] | — | [0.000035, 0.999684] |
| mean | **0.08149** | 0.05319 | 0.05075 |
| pos@0.5 | 413 | 282 | 275 |

> 注意：soup 的均值从 ~0.05 抬到 **0.0815**（更接近 ens10 的 0.0830）。这是**权重平均带来的校准变化**，不是排序问题；AUROC 只看排序，故不影响口径，但**若日后要用 0.5 阈值出 Pred_Label，必须重新校准**（本次候选的 Pred_Label 由各自 0.5 阈值生成，仅作格式完整，不参与评分）。

**soup 与各成员的一致性（Spearman / top200）**

| 对比 | Spearman | top200 |
|---|---|---|
| soup vs seed7 | 0.88212 | 194 |
| soup vs seed42 | 0.89149 | 194 |
| soup vs seed 概率平均 | 0.91202 | 197 |
| soup vs ens10 | 0.85025 | 189 |
| soup vs seed7 w10（98.84 候选） | 0.86468 | 190 |

soup 与两个 seed 的相关（0.882/0.891）都略低于两 seed 之间的 0.8868 量级，但**比任一单 seed 更靠近 ens10**（0.850 vs 0.810/0.816），说明权重平均确实起了「居中」作用。

**soup 融合候选（全部 5855/官方顺序/无 NaN）**

| 文件 | 构成 | mean | pos@0.5 | rho(98.84) | top200 |
|---|---|---|---|---|---|
| `cnxb_512_e16_ra_all_soup_test_tta4_ens10_logit_w10.csv` | ens10×0.9 + soup×0.1 | 0.08258 | 418 | **0.9992** | **200** |
| `..._ens10_logit_w20.csv` | ens10×0.8 + soup×0.2 | 0.08223 | 419 | 0.9982 | 200 |
| `..._ens10_logit_w30.csv` | ens10×0.7 + soup×0.3 | 0.08195 | 422 | 0.9952 | 200 |
| `..._ens10_rank_w10/20/30.csv` | rank 空间（探索） | ≈0.4999 | — | 0.90 量级 | 193 |

md5（前 8 位）：soup `acb54eac`｜soup_w10 `4ab5a54e`｜soup_w20 `f9fbb066`｜soup_w30 `14d361e0`

> 仓库内 CSV 为 LF；`submit_ready_soup/submission.csv` 为上传用 CRLF 文件，md5 为 `4ab5a54e`；两者逐行内容一致。

**结论**：soup 是把「两 seed 的权重噪声一起平滑」的低成本路径（2 分钟、单模型推理），其融合候选与 98.84 的排序几乎重合（w10 rho=0.9992、top200 200/200），因此**可以安全地作为下一次线上提交的候选**；收益方向与概率平均一致但机制不同，值得各占一次配额验证。
