# 权重共享清单（WEIGHTS）

> 跨机器集成的权重统一在此登记。权重文件**不入库**，通过 GitHub Release 附件或群文件共享。
> 命名规范（#14）：`<arch>_<img>_<cache>_<MMDD>_auprc<0.xxxx>.pt`
> **Release：[weights-1016](https://github.com/dongmingkun/glaucoma_first/releases/tag/weights-1016)** —— 队长机器 8 个 fp16 瘦身权重 + 云 4090D 的 seed2（#35 成员 A），下载即用

## 权重清单（全部为 fp16 瘦身版，由 `tools/slim_weights.py` 产出）

| 权重文件（点击下载） | arch | img_size | cache-root | TTA4 AUROC | TTA4 AUPRC | 用途 |
|---|---|---|---|---|---|---|
| [`swinl_384_e16_test_probs_tta4.csv`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/swinl_384_e16_test_probs_tta4.csv) | swinl_384_e16 | 384 | cache512 | 0.98732 | — | Swin-L@384 容量上探（195M） |
| [`swinb_384_e16_s7_test_probs_tta4.csv`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/swinb_384_e16_s7_test_probs_tta4.csv) | swinb_384_e16_s7 | 384 | cache512 | 0.99189 | — | Swin-B@384 seed7 复跑（验证 Swin 线增量） |
| [`swinb_384_all_v3_fp16.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/swinb_384_all_v3_fp16.pt) | swin_base | 384 | cache512 | —（train-on-all，valid 污染不报） | — | ⭐ 阶段2 train-on-all 异构成员（Swin-B@384，Spearman 0.9066 PASS）— 待集成 |
| [`convnext_small_512_cache768_1016_auprc0.8752.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_small_512_cache768_1016_auprc0.8752.pt) | convnext_small | 512 | **cache768** | 0.99085 | **0.87520** | ⭐ hi512_v2（续训 6ep）— **#13 的 640 热启动用它** |
| [`convnext_small_512_cache768_1016_seed2_auprc0.8750.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_small_512_cache768_1016_seed2_auprc0.8750.pt) | convnext_small | 512 | **cache768** | **0.99064** | **0.87503** | ⭐ seed2 多样性成员（seed 7 热启动 hi512_v2，6ep，云 4090D）— 待集成确认 |
| [`convnext_base_512_cache768_1016_auprc0.8765.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_base_512_cache768_1016_auprc0.8765.pt) | convnext_base | 512 | **cache768** | 0.99089 | **0.87646** | ⭐ 当前最强单模（Base@512） |
| [`convnext_base_384_cache512_1015_auprc0.8644.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_base_384_cache512_1015_auprc0.8644.pt) | convnext_base | 384 | cache512 | 0.99117 | 0.86439 | #11 的 Base@512 热启动基底 |
| [`convnext_small_384_cache512_1015_auprc0.8637.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_small_384_cache512_1015_auprc0.8637.pt) | convnext_small | 384 | cache512 | 0.98994 | 0.86371 | 集成成员 / 热启动基底 |
| [`convnext_tiny_384_cache512_1015_auprc0.8558.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/convnext_tiny_384_cache512_1015_auprc0.8558.pt) | convnext_tiny | 384 | cache512 | 0.98945 | 0.85584 | 集成成员 |
| [`effnet_b3_384_cache512_1014_auprc0.8159.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/effnet_b3_384_cache512_1014_auprc0.8159.pt) | efficientnet_b3 | 384 | cache512 | 0.98504 | 0.81592 | 集成成员 |
| [`effnet_b3_448_cache512_1014_auprc0.8267.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/effnet_b3_448_cache512_1014_auprc0.8267.pt) | efficientnet_b3 | 448 | cache512 | 0.98210 | 0.82672 | 集成成员（分辨率多样性） |
| [`effnet_b3_512_cache512_1014_auprc0.8142.pt`](https://github.com/dongmingkun/glaucoma_first/releases/download/weights-1016/effnet_b3_512_cache512_1014_auprc0.8142.pt) | efficientnet_b3 | 512 | cache512 | 0.98370 | 0.81415 | 集成成员 |

> **加载方式**：`predict.py` 自动从 checkpoint 读取 `use / arch / img_size`，并从训练 cache-root 推断 `cache_size`（512/768）。
> **热启动训练**：`train.py --init-from <下载的.pt> --arch <arch> --img-size <img> --cache-root <对应cache>`，建议 `--lr 3e-5`
> ⚠️ **瘦身版只有推理权重**（无 EMA 双份、无优化器状态）→ 可用于热启动，**不能续训**；要续训请向队长索取原始 `best.pt`

## 体积预算表（复赛红线 ≤500MB，压缩前）

| 成员 | 单份 fp32 | 单份 fp16（瘦身后，实测） |
|---|---|---|
| ConvNeXt-Small | ~200MB | **95MB** |
| ConvNeXt-Base | ~356MB | **168MB** |
| ConvNeXt-Tiny | ~113MB | **56.5MB** |
| EfficientNet-B3 | ~50MB | **22MB** |

### 当前最优集成（hi512_v2×2 + Base@384 + B3×3）

| 方案 | 体积 | 判定 |
|---|---|---|
| fp32 单份 | 200×2 + 356 + 50×3 = **906MB** | ❌ 超标 |
| **fp16 瘦身** | 95×2 + 168 + 22×3 = **424MB** | ✅ 合规（余量 ~76MB） |

> 结论：**复赛包必须走 fp16 瘦身**（`tools/slim_weights.py` 一键完成）。

## 瘦身工具用法

```bash
python tools/slim_weights.py --in <训练产出的 best.pt> --out <输出路径，不填则用规范命名>
# 产出：fp16 权重（BatchNorm running 统计保持 fp32）+ 自检（FundusNet.half() 加载 + 一次 fp16 前向，防 NaN）
# 文件键：state_dict / <use> / arch / img_size / use / fp16 / source（predict.py 可直接加载）
```

## 变更记录
- 2026-10-16 队长上传 8 个 fp16 瘦身权重至 `weights-1016`（含最新 hi512_v2 与 Base@512），回填本表链接
- 2026-10-16 修复瘦身工具与 predict.py 的键名集成问题 + 输出目录自动创建（PR #19 追加提交）
- 2026-10-16 云 4090D 训练 seed2 成员（#35 成员 A）并上传 Release：
  ```bash
  python train.py --arch convnext_small --img-size 512 --cache-root <cache768> \
    --epochs 6 --batch-size 8 --accum 6 --lr 3e-5 --head-lr 1.5e-4 \
    --loader workers --workers 3 --init-from <hi512_v2.pt> --seed 7 --tag cs_hi512_seed2
  # TTA4：AUROC 0.99064 / AUPRC 0.87503（vs hi512_v2 0.98994/0.87256，+0.00247）
