# RA_avg + Swin 全证据融合结果（ID 22205）

> 线上 2026-10-27 19:24 提交，ID 22205，**98.87**，+0.03 vs 98.84。首次验证 RA/Swin 方向。

## 候选
- 文件：`FINAL_SUBMIT_20261028/submission.csv`（内部 `4bf37b7c`）
- 公式（rank fusion）：
  `0.60*rank(ens10) + 0.10*rank(seed7/42平均) + 0.15*rank((RA_best+RA_last)/2) + 0.15*rank(Swin_pair)`
- 5855 行 / 官方顺序 / 无 NaN。

## 离线证据
- RA_avg+Swin 部分在 ens9 干净 valid（5849）上：`0.70*ens9 + 0.15*RA_avg + 0.15*Swin = 99.319871`，
  Δ+0.000595 vs ens9 99.260372，配对 bootstrap P≈1.0000。
- 相对 98.84 的 top-191 Jaccard 0.9691；不是 seed/soup 那种 rho≈0.999 的打平候选。

## 线上结果与位置
- ID 22205，98.87；历史最好从 98.84 → **98.87**。
- 全国榜：98.87 段约第 43 名 / 669（同分看提交时间；本单时间较晚）。
- 西北：约第 8 / 92；陕西第 4。NW 第 7 是兰州大学 98.87（更早提交），要越过它需要 ≥98.88。

## 下一步
- 下一档候选：`0.55*ens10 + 0.10*seedavg + 0.20*RA_avg + 0.15*Swin`（rank），md5 `f2b0612b`；
  RA/Swin 部分 valid Δ+0.000660，目标 98.88/西北第 7。
- 更激进备选：`0.50*ens10 + 0.25*RA_avg + 0.20*Swin + 0.05*cb384`（valid Δ+0.000754，但 top-191 Jaccard 0.949，风险更高）。
- 继续避免 seed123/soup3 这类 rho≈0.999 的打平候选。

## 后续：V4 激进版线上结果（ID 22704）
- 2026-10-28 00:01 提交 `NEXT_SUBMIT_98.9_v4/submission.csv`（md5 `fb62324b`），线上 **98.85**，低于 98.87。
- V4 = 0.40 ens10 +0.35 RA_avg +0.25 Swin_pair；把 RA/Swin 从 0.15/0.15 提到 0.35/0.25 属于过冲。
- 结论：98.87 的 0.15/0.15 已接近该方向最优点；继续加 RA/Swin 权重不再作为首选。下一手改新架构信号 B1（md5 `bbd3520d`）或等 #161 Large。

## 后续：S1 小步 new Swin 线上结果（ID 22732）
- 2026-10-28 00:06 提交 S1（md5 `50ce9244`），线上 **98.87**，与最好打平。
- S1 = 0.60 ens10 +0.15 RA_avg +0.15 Swin_pair +0.10 new Swin（把 seedavg 0.10 换成 new Swin）。
- 新 Swin 0.10 不足以跨档；下一手 B1（new Swin 0.15，md5 `bbd3520d`）或等 #161 Large。
