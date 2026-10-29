# Issue #186 结果 — 98.87 独立复现 + 提交前合规复核（CPU-only）

## 结论

| 项 | 结果 |
|---|---|
| **Task A 独立复现** | ✅ **PASS** —— 用仓库公开成员独立重算，输出 md5 = `4bf37b7c062f7d45a9c5a8e287fff1eb`，**与目标完全一致** |
| **Task B 提交前格式复核** | ✅ **PASS**（10/10 项）—— 对字节级相同的复现件核验；目标原文件不在本机 |
| 口径 | `RA_avg` = 先各自 rank 再 0.5/0.5 平均；`seedavg` = `seed7_s42_test_tta4.csv` |
| 缺件 | 目标 CSV、compliant_pkg 包体、cost.py/prediction.csv（均不在本机） |
| 是否上平台 / 改 CSV / 用 test 标签 | **均未做** |

> 目标文件 `FINAL_SUBMIT_20261028/submission.csv` 仍只在队长机器（仓库、Release、全部分支与历史均无），
> 因此无法与原文件逐行 diff；**但 MD5 完全一致 ⇒ 字节级相同 ⇒ 逐行最大差 0、标签翻转 0**，
> 这比逐行 diff 更强，不需要"用 98.87 反推"。

## 1. 口径是怎么定位的

工单给的公式里有两处写法会改变字节输出，直接照抄任一读法都无法命中：

| 分歧点 | 读法 | 结果 |
|---|---|---|
| `RA_avg` | ① `rank((best+last)/2)`（先概率平均再 rank，即 `tools/rank_blend.py` 现状）<br>② `0.5*rank(best)+0.5*rank(last)`（先各自 rank 再平均，**不重新排名**） | ① md5 `c94db095`；② **命中目标** |
| `seed7_s42_avg` | ① seed7/seed42 两个 CSV 概率平均<br>② 直接用 `seed7_s42_test_tta4.csv` | ① md5 `c94db095`；② **命中目标** |

定位方法：**先用队长已公开、且已知 md5 的 L1 反推口径**，再把锁定的口径套到 98.87。

- L1（Release `l1-large384-1028/submission_l1_large384.csv`，md5 `e244a37e6e64cb184d895ce63cb8db13`，5855 行）配方为 #161 的 `0.35 ens10 + 0.30 RA_avg + 0.20 Swin_pair + 0.15 Large384`；
- 用读法 ①（prob-mean）复算 → md5 不符；用读法 ② → **md5 `e244a37e` 完全一致**（逐行差 0）→ 锁定队长口径为「先 rank 再平均」；
- 再套到 98.87：`RA` 用读法 ②、`seedavg` 用文件 ② → md5 `4bf37b7c062f7d45a9c5a8e287fff1eb`，**命中**。

补充：`seed7_s42_test_tta4.csv` 与「seed7/seed42 概率平均」的 Spearman 0.9966、maxdiff 0.0928，
二者**不是同一个东西**（前者是 logit 空间平均）——工单文字写的"概率平均"与队长实际使用的输入文件不一致，
这是首个照抄公式会失败的原因。

同一批成员若改用 `tools/rank_blend.py`（prob-mean RA），98.87 会得到 `740fb841`；
这解释了为什么此前每个新包用仓库脚本复算都会差一点点（#179/#180 的同类结论）。

## 2. 复现命令

```bash
# 在仓库根目录，全部成员文件均已在 main
python tools/reproduce_98_87.py --check-md5 --out docs/audit_98_87/FINAL_REPRO_98_87.csv
# rows=5855  mean=0.500000  range=[0.002148, 0.999974]
# md5=4bf37b7c062f7d45a9c5a8e287fff1eb
# target=4bf37b7c062f7d45a9c5a8e287fff1eb -> MATCH
```

输出格式与队长产物一致：**CRLF** + `Image_name,Probability,Pred_Label` + `%.6f` + `Pred_Label = score >= 0.5`。
不加 `--out` 时不落盘，只打印 md5（供他人独立复核）。

`score` 的 10 个 6 位小数边界上，`%.6f` 的舍入依赖浮点运算顺序，因此脚本严格复刻了
「按 ens10 → seedavg → RA → Swin_pair 顺序累加、权重不重新归一」的写法；实测权重归一与否对 md5 无影响
（`0.6+0.1+0.15+0.15 ≠ 1.0` 只在 1e-16 量级），但累加顺序会。

## 3. 输入校验与哈希（全部 5855 行、官方名序、无 NaN、均在 [0,1]）

| 键 | 文件 | 字节 | SHA-256 | MD5（LF，仓库内） |
|---|---|---|---|---|
| ens10 | `submission_20251018_ens10_cal.csv` | 152264 | `aa886d901a354345413356315e4de75c7ed17dbfce6bdf09e09ba7dade9a9b44` | `7bed58a6a300a9c38245f48e5e5aed19` |
| seedavg | `seed7_s42_test_tta4.csv` | 152264 | `9f9962b85056872a80f6968601e1da29e66b877d88209a1ae0862435987215a6` | `3b1aa97fd5add2f9add3d9f998648cdc` |
| ra_best | `cnxb_512_e16_ra_test_best_tta4.csv` | 152264 | `f434008b0b1bac3e3cc35b965147616a19b23f82647181833787636e1e99af58` | `077d600f9c0f392be7f3afe22da1cbdf` |
| ra_last | `cnxb_512_e16_ra_test_last_tta4.csv` | 152264 | `f36602d2fc689f2ce9eed7cd8ff204612045dd58c74bbf386f6347c2efe44d7b` | `1262969287a9d71accab8d7587cb161e` |
| swin_pair | `swin_pair_test_tta4.csv` | 152264 | `760e3242495a1cb3fda7ec7e645bdcd2ef9f9dc34c01a9df4941fde341baa567` | `8cea0b01575933f11896d7a961294171` |
| （参考）seed7 | `cnxb_512_e16_ra_all_test_tta4.csv` | 152264 | `3bea1c6e5dbf9e53dfb6b9009b75600aade480d9bb51057bb2884df2989e02f8` | `c6fbf63687770608d1e86b2c247a1fc1` |
| （参考）seed42 | `cnxb_512_e16_ra_all_s42_test_tta4.csv` | 152264 | `bd0ca82194bd2b267ca424dcd636be2cc6dd87a1bea286ba201d6bc20e4810a6` | `25a109d85094916a373a19ffb6027c09` |
| 官方名序 | `glaucoma_first/test.csv` | 117115 | `38aa66608cb07a7dad643550d416c8832920d4da7ae9b026c2df809d69c0cca5` | `438887acdfa61244dd7af97851e57218` |

**MD5 口径澄清（不是版本分歧）**：工单写的 `ce91bbf4`（ens10）与 #178 文档写的 `08250726`（swin_pair）
都是 **Windows CRLF 版本**的 md5；仓库 blob 是 LF，内容逐字节相同。归一化后：
`ens10` CRLF md5 = `ce91bbf48d6c8e05d7f67a43d902a5ae`、`swin_pair` CRLF md5 = `0825072672651298f36e3e7991ab60ad`，与台账吻合。

## 4. Task B：提交前格式 / 合规复核

对复现件（字节级等于目标）逐项检查，**10/10 通过**：

| 检查 | 结果 |
|---|---|
| 5855 行 / 三列 / 表头 `Image_name,Probability,Pred_Label` | ✅ |
| 官方 test.csv 名序、5855 个唯一名 | ✅ |
| 全部 finite、∈[0,1] | ✅ |
| `Pred_Label` 与 `Probability >= 0.5` 一致 | ✅ |
| CRLF（5856 个）、无 BOM | ✅ |
| 不含 test 伪标签（输入全部是模型概率，未读任何 test 标签） | ✅ |

- **compliant_pkg**：包体不在本机，无法解压核验；但按 Release 资产字节数核对为
  `cb384` 176,328,125 + `cs_hi512_all` 199,660,651 + `cs_hi512_all_ra` 99,831,694 = **475,820,470 B = 475.82 MB**，
  与"475.82MB、3 模型"一致。三个成员均为 ImageNet 预训练骨干 + 官方 train/valid 微调，
  **不含** 98.87 的 seedavg / rank 融合等 test-only 成分，包说明无需（也不应）写入这些成分。
- **cost.py / prediction.csv**：本机不存在，按工单标记 **未验证**。

## 5. 缺件清单

| 缺件 | 影响 | 处理 |
|---|---|---|
| `FINAL_SUBMIT_20261028/submission.csv`（md5 `4bf37b7c`） | 无法与原文件直接 diff | 已用 MD5 完全一致替代（更强）；若需要，队长提供后本脚本可 1 条命令重算 |
| compliant_pkg 3 个 `.pt` 打包件 | 无法解压核验 | 仅按 Release 字节数核对 475.82MB |
| `cost.py` / `prediction.csv` | 无法核对二者与 submission 一致性 | 标记未验证 |

## 6. 交付物

| 路径 | 内容 |
|---|---|
| `tools/reproduce_98_87.py` | 独立复现脚本（numpy-only，`--check-md5` 一条命令验证） |
| `docs/audit_98_87/FINAL_REPRO_98_87.csv` | 复现件，md5 `4bf37b7c062f7d45a9c5a8e287fff1eb`（158,120 B） |
| `docs/audit_98_87/audit_98_87.json` | Task A/B 结论、口径、逐项检查、缺件 |
| `docs/audit_98_87/inputs_sha256.json` | 各输入 SHA-256 / MD5（LF 与 CRLF） |
| `D:/glaucoma_work/runs/feedback_20261024/audit_98_87/` | 同上三个文件（本机 D: 根目录不可写，同构落在 `issue186/deliverables/...`，内容一致） |

## 7. 后续建议（未动手，等确认）

`tools/rank_blend.py` 的 RA_avg 仍是「先概率平均再 rank」，与队长流水线不一致，
导致任何用它复算的包都会被审计判成"差一点点"（本单 98.87 差成 `740fb841`，L1 差成 max_err 0.0593）。
建议加 `--ra-rank-mean` 开关并把默认改为「先 rank 再平均」；这会改变历史候选的 md5，
所以不在本单擅自修改。
