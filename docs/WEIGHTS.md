# 权重共享清单（WEIGHTS）

> 跨机器集成的权重统一在此登记。权重文件**不入库**，通过 GitHub Release 附件或群文件共享。
> 命名规范（#14）：`<arch>_<img>_<cache>_<MMDD>_auprc<0.xxxx>.pt`

## 权重清单

| 权重文件 | 下载链接 | arch | img_size | cache-root | TTA4 AUROC | TTA4 AUPRC | 用途 |
|---|---|---|---|---|---|---|---|
| `convnext_small_512_cache768_1016_auprc0.8726.pt` | 待上传 | convnext_small | 512 | cache768 | 0.98994 | **0.87256** | 当前最强单模（hi512 配方） |
| `convnext_base_384_cache512_1016_auprc0.8644.pt` | 待上传 | convnext_base | 384 | cache512 | 0.99117 | 0.86439 | 架构最强单模 |
| `convnext_tiny_384_cache512_1015_auprc0.8558.pt` | 待上传 | convnext_tiny | 384 | cache512 | 0.98945 | 0.85584 | 集成成员 |
| `efficientnet_b3_384_cache512_1014_auprc0.8159.pt` | 待上传 | efficientnet_b3 | 384 | cache512 | 0.98504 | 0.81592 | 集成成员 |
| `efficientnet_b3_448_cache512_1014_auprc0.8267.pt` | 待上传 | efficientnet_b3 | 448 | cache512 | 0.98210 | 0.82672 | 集成成员（分辨率多样性） |
| `efficientnet_b3_512_cache512_1014_auprc0.8141.pt` | 待上传 | efficientnet_b3 | 512 | cache512 | 0.98370 | 0.81415 | 集成成员 |

> 上传方法（网页打不开时用 API）：
> ```bash
> curl -X POST -H "Authorization: Bearer <token>" \
>   -H "Content-Type: application/octet-stream" --data-binary @权重.pt \
>   "https://uploads.github.com/repos/dongmingkun/glaucoma_first/releases/<release_id>/assets?name=<文件名>"
> ```

## 体积预算表（复赛红线 ≤500MB，压缩前）

| 成员 | 单份 fp32 | 单份 fp16（瘦身后） |
|---|---|---|
| ConvNeXt-Small | ~200MB | ~100MB |
| ConvNeXt-Base | ~356MB | ~178MB |
| ConvNeXt-Tiny | ~113MB | ~56.5MB |
| EfficientNet-B3 | ~50MB | ~25MB |

### 当前最优集成（Small@512 + Base + B3×3）

| 方案 | 体积 | 判定 |
|---|---|---|
| fp32 单份 | 200 + 356 + 50×3 = **706MB** | ❌ 超标 |
| **fp16 瘦身**（`tools/slim_weights.py`） | 100 + 178 + 25×3 = **353MB** | ✅ 合规 |

> 结论：**必须走 fp16 瘦身**（`tools/slim_weights.py` 一键完成，见下）。

## 瘦身工具

```bash
# 读 best.pt（含 EMA 双份 fp32）→ 只留推理份 → 转 fp16 → 自检
python tools/slim_weights.py --in runs/<exp>/best.pt [--use auto|ema|model]
```

- 自动命名 `<arch>_<img>_<cache>_<MMDD>_auprc<0.xxxx>.pt`
- 打印瘦身前/后大小 + 自检（`FundusNet.half()` 加载 + 一次 fp16 前向）
- 产物配合 `submit_pkg/predict.py`（自动识别 `arch`/`img_size`/`cache_size`）
