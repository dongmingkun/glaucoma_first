# MANIFEST — submission_pkg_1model

## 权重（来源 Release `weights-1016`，本地 torch CPU 瘦身；不改权重内容）

| 文件 | arch | img_size | cache_size | use | 预期大小 | 参考 md5(8) |
|---|---|---|---|---|---|---|
| `cnxb_384_all_pseudo.pt` | convnext_base | 384 | 512 | ema | 176.3 MB | `96c91286` |

- **预期包体（仅权重）：176.3 MB ≤ 200 MB ✅**（验收口径：单模型目录 ≤200 MB）
- 实际 md5 / 瘦身由 `assemble.py` 拉取 Release `cnxb_384_all_pseudo_fp16.pt`（336 MB 双态）
  → 本地 torch CPU 瘦身单态后生成并回填此处。瘦身仅移除冗余 ema/model 副本，
  保留的 `use=ema` 态不变。
- 与 3 模型包内 `cnxb_384_all_pseudo.pt` 同权重（use=ema，见 `docs/ISSUE_81.md` 审计）。
- **T4：未实测**（当前无 T4 环境）；5060 p95 ≤40 ms 为安全边际目标，详见 README。

## 文件清单

| 文件 | 说明 |
|---|---|
| `predict.py` | 推理脚本（与 `submit_pkg/predict.py` 主版本逐行一致，未改推理逻辑） |
| `cnxb_384_all_pseudo.pt` | fp16 单态权重（assemble 后生成） |
| `measure_latency.py` | 逐批计时（mean/p50/p95，剔除前 2 warmup 批） |
| `assemble.py` | 从 Release `weights-1016` 拉权重 + 瘦身 cnxb |
| `README.md` | 使用说明 / T4 策略 / 命令 |
| `MANIFEST.md` | 本文件 |
