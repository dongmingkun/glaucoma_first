# MANIFEST — submission_pkg_1model

## 权重（来源 Release `weights-1016`，本地 torch CPU 瘦身；不改权重内容）

| 文件 | arch | img_size | cache_size | use | 实际大小 | md5(8) |
|---|---|---|---|---|---|---|
| `cnxb_384_all_pseudo.pt` | convnext_base | 384 | 512 | ema | 176.3 MB | `96c91286` |

- **包体（仅权重）：176.3 MB ≤ 200 MB ✅**（验收口径：单模型目录 ≤200 MB）
- 本包权重与 `submit_pkg/cnxb_384_all_pseudo.pt` 同文件（md5 `96c91286`，use=ema）；
  `assemble.py` 可从 Release `weights-1016` 重新拉取并瘦身，结果一致。
- **5060 实测（2026-10-23，全量 5855 张，bs32，prep-workers 8）**：
  mean **32.65 ms** / p50 **29.18 ms** / **p95 53.27 ms** / p99 75.58 ms。
  目标 p95 ≤40 ms **未达到**；相对 100 ms 红线约 **1.9×** 余量。
- **T4：未实测**（当前无 T4 环境）；100 ms 红线不能据此认证，详见 README。

## 文件清单

| 文件 | 说明 |
|---|---|
| `predict.py` | 推理脚本（与 `submit_pkg/predict.py` 主版本逐行一致，未改推理逻辑） |
| `cnxb_384_all_pseudo.pt` | fp16 单态权重（assemble 后生成） |
| `measure_latency.py` | 逐批计时（mean/p50/p95，剔除前 2 warmup 批） |
| `assemble.py` | 从 Release `weights-1016` 拉权重 + 瘦身 cnxb |
| `README.md` | 使用说明 / T4 策略 / 命令 |
| `MANIFEST.md` | 本文件 |
