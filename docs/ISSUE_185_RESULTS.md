# Issue #185 结果 — Base@512 RA train+all seed123（第 3 个 seed）

> **状态：本条已被 PR #155（#152）完全覆盖，无需再跑 GPU。** 本单只补两件事：
> ① 用 CPU 口径核验成员并给出 `check_train_on_all` 的替代判据；② 补一个此前没人算过的判据——**把 seedavg 换成 3-seed 后，98.87 配方实际位移多少**。

## 一、为什么不需要租卡

#152（PR #155，2026-10-2x 合并）已在 **RTX 4090 24GB 真机**上按**完全相同**的命令跑完 seed123 16ep，产物全部在 main 上：

| #185 要求 | 实际产物 | 位置 | 校验 |
|---|---|---|---|
| `runs/cnxb_512_e16_ra_all_s123/last.pt` | 705,015,055 B（EMA） | Release **`ra_all_s123`** → `cnxb_512_e16_ra_all_s123_last.pt` | ✅ 存在 |
| test TTA4 5855 行 / 官方序 / 无 NaN | mean **0.052785**、pos@0.5 **285** | `cnxb_512_e16_ra_all_s123_test_tta4.csv` | ✅ 全部通过 |
| 与 seed7/42 的 Spearman | **0.8996 / 0.8887** | 本文第二节 | ✅ >0.88 |
| `docs/ISSUE_185_RESULTS.md` | 本文（#152 的完整结果在 `docs/ISSUE_152_RESULTS.md`） | — | ✅ |
| `check_train_on_all.py` | **未执行**（需 ref ckpt + cache_all768） | — | ⚠️ 见第三节替代口径 |

#152 另外还产出了 3-seed 平均、3-seed soup 与 24 个融合候选（`seed7_s42_s123_test_tta4*.csv`、`soup3_test_tta4*.csv`），并推荐了 `soup3_test_tta4_ens10_logit_w10.csv`（md5 `6e00216d`）。这些都在 main 上，本单不需要重复劳动。

## 二、成员校验（CPU，`tools/check_seed123_cpu.py`）

全部 5855 行 / 官方 test.csv 名序 / 无 NaN / ∈[0,1]：

| 成员 | 文件 | mean | pos@0.5 | SHA-256 |
|---|---|---|---|---|
| seed7 | `cnxb_512_e16_ra_all_test_tta4.csv` | 0.053191 | 282 | `cdfd7bbd1a4c478964a222bb0e7fc24952ecc5a21b34b95a6ceb3ccb7372cfe1` |
| seed42 | `cnxb_512_e16_ra_all_s42_test_tta4.csv` | 0.050754 | 275 | `fc367451831dd59fe853afb249d7c2538b8c3455c90f245288f3773542cd8184` |
| **seed123** | `cnxb_512_e16_ra_all_s123_test_tta4.csv` | **0.052785** | **285** | `b36f8d104ca6c16b0fa9c98f2161dd7ab85ed349dc9ff05ba74a501609dea6d1` |
| 2-seed 平均 | `seed7_s42_test_tta4.csv` | 0.051570 | 278 | — |
| 3-seed 平均 | `seed7_s42_s123_test_tta4.csv` | 0.051787 | 282 | `1af99c3fe12eb93bfd98628752c17e629b223a9957172e1f4d4d49a9a29cf093` |

两两 Spearman / top200：

| | seed42 | seed123 | avg2 | avg3 |
|---|---|---|---|---|
| seed7 | 0.8868 / 191 | 0.8996 / 190 | 0.9699 / 196 | 0.9615 / 195 |
| seed42 | — | 0.8887 / 186 | 0.9705 / 195 | 0.9575 / 194 |
| seed123 | | — | 0.9193 / 190 | 0.9667 / 192 |
| avg2 | | | — | 0.9886 / 198 |

mean 漂移：seed123 0.052785 vs seed7 0.053191（**−0.0004**）、vs seed42 0.050754（+0.0020）→ 无系统性偏移。
3-seed 平均 = `sigmoid(mean(logit(seed7), logit(seed42), logit(seed123)))`，与仓库文件逐行最大差 **0.000000**。

## 三、`check_train_on_all` 为什么没跑（以及替代口径）

`check_train_on_all.py` 需要 `runs/cnxb_512_e16_ra/last.pt`（#128 train-only 权重）+ `cache_all768`（60k 张、数十 GB），本机无 GPU 也无缓存，按工单不硬凑。

它的三个输出都能用纯 CSV 得到，上表已覆盖：
- **Spearman**：0.8996 / 0.8887（对 seed7 / seed42）；
- **mean 漂移**：−0.0004 / +0.0020；
- **top200**：190 / 186。
（`check_train_on_all` 原本比的是「新 ckpt vs 旧 ckpt」的 test 概率，与这里用现成 CSV 比等价。）

⚠️ 工单的验收线「与 seed7/42 的 test Spearman >0.95」**不成立**：seed7↔seed42 自身只有 **0.8868**，
任何单个 seed 都达不到 0.95。建议把阈值改成「≥ seed7↔seed42 的 0.887」或「>0.85」。

## 四、补的判据：把 seedavg 换成 3-seed，98.87 会动多少

98.87 配方（#186 已 byte-exact 复现，md5 `4bf37b7c`）里 seedavg 权重只有 0.10。把 2-seed 平均换成 3-seed 平均：

| 指标 | 值 |
|---|---|
| Spearman | **0.999858** |
| top200 重合 | **200 / 200** |
| **top200 变化张数** | **0** |
| 逐行最大差 | 0.031 |
| md5（2-seed） | `4bf37b7c062f7d45a9c5a8e287fff1eb`（= 98.87，反向验证脚本正确） |
| md5（3-seed） | `06aa5698e6c2fad3f76d42bc6df090aa` |

**结论：头部 200 张一张不动。** 前两次 seed 扩展（单 seed → 2-seed 平均）曾带来 +0.03（98.84 → 98.87），
但那次位移体现在 top200 上；这次 top200 完全不变，**预期打平，不建议单独占一次提交配额**。
若仍想验证第三 seed 的信息量，#152 推荐的 `6e00216d`（soup3×ens10 w10）机制不同（权重空间平均），比纯平均更值得试。

## 五、复现

```bash
python tools/check_seed123_cpu.py          # 打印两两 Spearman/top200 与 98.87 配方位移
# 结果写到 docs/audit_185/check_seed123.json（含各文件 SHA-256）
```

## 六、缺口（如实）

1. `check_train_on_all.py` 未执行（无 GPU/cache）；第三节给出等价 CPU 口径；
2. valid AUROC 按工单不报（train+all，无干净 valid）；
3. seed123 线上效果未知——本单不提交，是否提分只能由一次提交回答，失败不降低 98.87；
4. 本单与 #152 重复派单，建议后续新建训练单前先查 `SUBMISSIONS_LOG.md` 与 Release `ra_all_s123`。
