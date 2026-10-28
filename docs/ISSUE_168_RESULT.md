# Issue #168 结果记录 — 同权重 RA_soup V6（判打平：V6a ≡ V4，排序完全一致）

> 状态：**已完成，判打平**。V6a 与 98.87（V4）的 Spearman = 1.0000、top200 = 200/200——
> 按工单"若 V6a rho≥0.995 且 top200=200，如实判打平"——**触发**。
> 环境：本机 5060（均衡），纯 CPU；RA_soup 沿用 #162 产物（ckpt 权重平均 0.5/0.5，非概率平均，已确认）。

## V6a 配方与判定

- V6a = `0.60 ens10 + 0.10 seedavg + 0.15 RA_soup + 0.15 Swin_pair`（rank）
- V4（98.87）= `0.60 ens10 + 0.10 seedavg + 0.15 RA_avg + 0.15 Swin_pair`（rank）
- **唯一变量：RA_avg（概率平均）→ RA_soup（权重平均）**

| 对照 | Spearman | top200 |
|---|---|---|
| V6a vs V4（98.87 复现） | **1.0000** | **200/200** |
| V6a vs V5a | 0.9874 | 197/200 |

**打平判定：触发**（rho 1.0 ≥ 0.995 且 top200=200）。工单条件"V6a 比 V5a 更接近 98.87 结构且非打平才推荐 V6a"——打平，**不推荐 V6a**。

## 机制解释（本单的核心发现）

RA_soup（best+last **权重**平均后推理）与 RA_avg（best+last **概率**平均）在 5855 张 test 上
**排序完全一致**（rho=1.0）。原因：RA_best 与 RA_last 本就来自同一训练轨迹的相邻 checkpoint，
权重线性插值 ≈ 输出概率插值，rank 作为单调不变量把两者拉平。

**结论：对同一轨迹的相邻 ckpt，"checkpoint soup" 与 "概率平均" 是排序等价操作**——
soup 的价值只可能出现在**跨轨迹/跨超参的多个独立 ckpt** 之间（如 #149 的 seed7+seed42 soup，
那是不同 seed 的独立训练）。RA 这对 ckpt 上验证不出 soup 增益，属预期内的阴性结果。

## 对候选格局的影响

- **soup 路线的代表仍是 V5a**（md5 `1eb4b12e...`，RA_soup 0.35 + Swin_pair 0.25，vs V4 rho 0.9872）
- V6a（打平）**不推荐提交**——线上分数预计与 98.87 逐位一致的复刻，浪费一次提交机会
- 明天的主候选判定不变：V5a vs #161 Large，谁有信号用谁

## 产物

- `oof/V6_V6a.csv`（md5 `1b1864c09a7fe988bd02e9cb49abe6c9`，5855 行/官方顺序/无 NaN）
- 打包：`submission_20261027_V6a_rasoup_tied.zip`（作为打平证据，不推荐提交）
- 复现：rank 融合脚本（#168 内联），成员文件全部来自 #160/#162 已上传版本

## 文件上传（工单"先上传/确认"部分）

- `oof/cnxb_512_e16_ra_soup_test_tta4.csv`（5855）——待上传群文件/Release
- V5a 最终 zip：`submission_20261027_V5a_rasoup_swinpair_cal.zip`（md5 `9854c208...`，**zip 内 CSV = 1eb4b12e... 已验证**）
- RA_soup 构成确认：**ckpt 权重平均（0.5/0.5，seed_soup.py），非概率平均** ✅
