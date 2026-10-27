# #157 执行手册：Swin-B@384 train+all 第二 seed（seed123）+ 融合

> deadline **2026-10-28 08:00**；线上最好 **98.87**（ID 22205，RA_avg+Swin 全证据），下一档 **98.90**。
> 本单只加 **新的架构信号**（Swin-B train+all 第二 seed），**不再做 seed/soup 打平类候选**。
> 候选**不自动提交**，由队长选定唯一一份。

## 一、前置
| 项 | 要求 |
|---|---|
| GPU | **4090 24GB**（16ep Swin-B@384 约 3–3.5h；本地 5060 禁止硬跑） |
| 缓存 | `cache_all768` = 57245/57245/5855（Swin 训练 resize 到 384，复用 768 缓存） |
| 组件 | `submission_20251018_ens10_cal.csv`（ens10）、`cnxb_512_e16_ra_test_{best,last}_tta4.csv`（RA_avg）、`swinb_384_e16_test_probs_tta4.csv` + `swinb_384_all_v3_test_probs_tta4.csv`（Swin_pair）、`oof/seed7_s42_test_tta4.csv`（seedavg，仅重建对照用） |
| 参照 | 98.87 的 `FINAL_SUBMIT_20261028/submission.csv`（md5 `4bf37b7c`）**不在仓库/云端**，脚本改为**用同配方重建对照**（0.60 ens10 + 0.10 seedavg + 0.15 RA_avg + 0.15 Swin_pair），并如实标注 |

## 二、两步执行
```bash
# 1) 训练 + test TTA4（约 3.2h）
PYTHON=/root/miniconda3/bin/python nohup bash /root/train_157.sh > /root/run157.nohup 2>&1 &

# 2) 训练完成后：融合 + 校验（约 1 分钟）
PYTHON=/root/miniconda3/bin/python LOG=/root/run157b.log bash tools/build_157_candidates.sh
```

## 三、候选设计
| 编号 | 构成 | 说明 |
|---|---|---|
| C1 | `fusion_ens10 --weights 0.10 0.20 0.30`（ens10 + new Swin，logit） | 工单第 1 项；Swin 只做小幅掺入 |
| **B1** | rank：**0.50 ens10 + 0.20 RA_avg + 0.15 Swin_pair + 0.15 new Swin** | 工单第 2 项重点候选 |
| **B2** | rank：0.55 ens10 + 0.20 RA_avg + 0.10 Swin_pair + 0.15 new Swin | 更保守的 Swin_pair，new Swin 占比不变 |
| B3 | rank：0.45 ens10 + 0.20 RA_avg + 0.10 Swin_pair + **0.25 new Swin** | 冲 98.9 的激进档（Swin 暴露最大） |
| B4 | 同 B1 但用 soup(v3+new) 代替 new Swin | 仅当 v3 ckpt 含 `ema` 可读；否则按工单「不在就跳过」 |

rank 融合口径：成员内多文件先**概率空间平均**，再 `rank/(n-1)` 归一，按权重线性组合（`tools/rank_blend.py`）。

## 四、必须回填的校验（脚本自动打印）
- 5855 行 / 官方顺序（逐行比对 `glaucoma_first/test.csv`）/ 无 NaN / `[0,1]` / mean / pos@0.5；
- **new Swin vs seed7 / seed42 / seed123** 的 Spearman + top200（跨架构，预期 0.8 左右，这是本单唯一真新信息）；
- 每个候选 vs **重建的 98.87 对照** 的 rho + top200（原 98.87 文件缺失，已如实说明）；
- 选 1–2 个用 `package_submission.py` 打包并记录 md5。

## 五、判负线与风险
- 工单判负线：**若所有候选 vs 98.87 的 rho ≥0.995 且 top200 = 200/200 → 建议不交**；
- Swin 历史迁移不稳（ens20 98.81、ens22/ens23 未达预期），但 train+all Swin **是唯一没试过的架构信号**；
- Swin 概率分布与 ConvNeXt 家族差异大（ens10 vs RA 0.769 量级），掺入比例过高会改变 top-200 结构 → B3 风险最高，只在 B1/B2 不足时考虑。

## 六、交付
- `runs/swinb_384_all_s123/{last,best}.pt` → Release（大文件不入库）
- `oof/swinb_384_all_s123_test_tta4.csv` + 全部候选 CSV
- `docs/ISSUE_157_RESULTS.md`（训练耗时/校验表/推荐）
- 本 issue 评论：路径、md5、校验数字、**推荐哪一个 + 理由**；**不自动提交**

## 七、踩坑记录（沿用 #152 的教训）
- **watchdog 必须有重启上限**：无上限会在确定性失败上空转（#152 曾空转 4.5h）；
- **组件 CSV 位置不统一**：ens10/RA/seed7 在仓库根、`seed42/seed123/soup3` 在 `oof/`、Swin 成员在 `/root/autodl-tmp/`（本脚本已把它们落到仓库目录再引用）；
- **权重平均会抬校准**，rank 融合则输出 0..1 均匀分（mean≈0.5），`Pred_Label` 无阈值意义，仅格式完整；
- 云端 `git push` 可能被代理挡（502/SSL）→ 用 GitHub Contents API（urllib）直接推文件。
