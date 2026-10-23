# MANIFEST — submission_pkg_2model

## 权重（来源 Release `weights-1016`，本地 torch CPU 瘦身；不改权重内容）

| 文件 | arch | img_size | cache_size | use | 实际大小 | md5(8) |
|---|---|---|---|---|---|---|
| `cb384.pt` | convnext_base | 384 | 512 | model | 176.3 MB | `b7aabbdd` |
| `cnxb_384_all_pseudo.pt` | convnext_base | 384 | 512 | ema | 176.3 MB | `96c91286` |

- **包体（仅权重）：352.6 MB ≤ 400 MB ✅**（验收口径：双模型目录 ≤400 MB）
- 两个权重与 3 模型包内文件一致（`cb384.pt` use=model、`cnxb_384_all_pseudo.pt` use=ema）；
  `assemble.py` 可从 Release `weights-1016` 重新拉取，结果一致。
- **5060 p95：待测**（测量时 GPU 被训练任务占用；同配置单模型 p95 为 53.27 ms，2 模型预计更高）。
- **T4：未实测**（当前无 T4 环境）；100 ms 红线不能据此认证，详见 README。

## 文件清单

| 文件 | 说明 |
|---|---|
| `predict.py` | 推理脚本（与 `submit_pkg/predict.py` 主版本逐行一致，未改推理逻辑） |
| `cb384.pt` | fp16 单态权重（assemble 后生成） |
| `cnxb_384_all_pseudo.pt` | fp16 单态权重（assemble 后生成） |
| `measure_latency.py` | 逐批计时（mean/p50/p95，剔除前 2 warmup 批） |
| `assemble.py` | 从 Release `weights-1016` 拉权重 + 瘦身 cnxb |
| `README.md` | 使用说明 / T4 策略 / 命令 |
| `MANIFEST.md` | 本文件 |
