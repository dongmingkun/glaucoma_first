# 复赛提交包说明

## 内容
- `predict.py`：自包含推理脚本（只需 torch/torchvision/pillow/numpy，官方镜像内全部自带）
- `model_weights.pt`：训练好的模型权重（含 ema/model 两套 + 元信息，由训练脚本的 best.pt 直接拷贝）
- 本 README

## 官方环境（与本地训练环境一致）
- GPU: NVIDIA T4 16GB；CUDA 12.8 + cuDNN 9
- Python 3.12；PyTorch 2.10.0 + torchvision 0.25.0
- 模型大小：单模型 ≈ 50MB（efficientnet_b3）~ 350MB（convnext_base），多模型合计 <500MB 限制（fp16 瘦身见 `docs/WEIGHTS.md`）

## 延迟红线与推荐配置（≤100ms 单图端到端）

| 配置 | GPU 延迟（T4 估算）| 预处理 | 单图端到端 | 判定 |
|---|---|---|---|---|
| 3 模型 + TTA4 | ~115ms | ~40ms | **~155ms** | ❌ 超标 |
| 2 模型 + TTA4 | ~85ms | ~30ms | ~115ms | ⚠️ 临界 |
| **2 模型 + TTA1（推荐）** | ~35ms | ~30ms | **~75ms** | ✅ 合规 |

**推荐复赛配置**（hi512_v2×2 + Base，`--tta 1`）：
```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cs_hi512_v2.pt cs_hi512_v2.pt cnxb.pt --blend 1 1 1 \
  --tta 1 \
  --calib-shift <logit常数> \
  --out submission.csv
```
> 若实测 p95 仍超 100ms，进一步降级：减少集成成员、或降低输入分辨率。

## 用法

### 单模型
```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights model_weights.pt \
  --out submission.csv
```

### 多模型集成（加权概率平均）
```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cnxs_512.pt cnxb_384.pt b3_384.pt b3_448.pt b3_512.pt \
  --blend 1 0.4 0.2 0.2 0.2 \
  --out submission.csv
```

### 多模型 + 烘焙校准
```bash
# --calib-shift 为 calibrate_submission.py 在 valid 上求得的 logit 偏移常数
# （单调平移，不改变排序，只把最优工作点对准 0.5，与线上口径一致）
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights a.pt b.pt c.pt --blend 1 0.8 0.5 \
  --calib-shift 2.214 \
  --out submission.csv
```

参数：
- `--weights`：一个或多个 `.pt`（多模型时按 `--blend` 加权概率平均）
- `--blend`：与 `--weights` 一一对应的权重（默认等权，自动归一化）
- `--calib-shift`：烘焙校准的 logit 偏移常数（默认 0，即不校准）
- `--tta 4`（默认）：4 个翻转变体平均；**复赛推荐 `--tta 1`**（延迟红线）
- `--batch-size 64`：可按机器/显存调整（768 缓存模型较大时建议下调）

## 输出格式
UTF-8 CSV，三列：`Image_name,Probability,Pred_Label`（Probability ∈ [0,1]，Pred_Label 以 0.5 为阈值）

## 设计说明
- **多架构**：`predict.py` 从每个 checkpoint 的 `arch` 字段自动构建模型，支持
  `efficientnet_b0/b3/b4`、`convnext_tiny/small/base`、`resnet50`（与训练脚本一致）
- **每个模型自己的预处理**：`img_size` 和 `cache_size`（512 vs 768）均从 checkpoint 元信息推断，
  768 缓存模型（如 Small@512）会自动用 768 方图裁剪再 resize 到输入尺寸，与训练时一致
- **预处理去重（#20）**：同 `cache_size` 的模型共享一次解码+裁剪（而非每模型×每翻转各做一次），
  翻转变体在张量上用 `torch.flip` 完成，避免回到 PIL
- **draft 尺度对齐（#20）**：JPEG draft = `max(768, cache_size*1.5)`（512 缓存→768、768 缓存→1152），
  与训练缓存档位一致，概率漂移从 ~0.008 压到 ~1e-4
- **自包含**：脚本内部实现眼底裁剪（阈值找圆形视野 bbox → 居中方形 → cache_size → resize），
  不依赖训练时的任何缓存文件

## 已验证
- [x] 本地 RTX 5060 上单模型全流程跑通（见项目根目录 infer.py 的验证记录）
- [ ] 多模型 + 烘焙校准在官方 docker 镜像内实测（T4 / Python 3.12）— 待权重与 logit 常数就绪后执行
- [ ] 预处理去重 + 张量翻转的延迟实测（p50/p95）与概率一致性（±1e-3）— 需 T4 环境

## 注意
- ⛔ **合规红线**：本提交包与任何提交文件**禁止包含 RetFound 系列权重**（外部视网膜数据预训练，与"仅可用主办方数据"规则冲突；仅限内部消融参考，2026-10-16 决策）。提交模型一律为 ConvNeXt / EfficientNet 系列
- 若官方测试图片命名或路径结构与本地不同，`predict.py` 会按 `Image_name` 的主文件名自动匹配 `.JPG/.jpg/.jpeg/.png/.PNG` 扩展名
- 年龄列缺失或异常（<1 或 >110）时用训练中位数 58 代替，与训练逻辑一致
- 多模型会增加端到端耗时（每模型一次前向），若单图 p95 超 100ms，降级顺序：`--tta 4→1` → 减少集成成员 → 降低输入分辨率
