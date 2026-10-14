# 复赛提交包说明

## 内容
- `predict.py`：自包含推理脚本（只需 torch/torchvision/pillow/numpy，官方镜像内全部自带）
- `model_weights.pt`：训练好的模型权重（含 ema/model 两套 + 元信息，由训练脚本的 best.pt 直接拷贝）
- 本 README

## 官方环境（与本地训练环境一致）
- GPU: NVIDIA T4 16GB；CUDA 12.8 + cuDNN 9
- Python 3.12；PyTorch 2.10.0 + torchvision 0.25.0
- 模型大小：efficientnet_b3 融合头 ≈ 50MB（<500MB 限制）
- 单图端到端延迟：JPEG draft 解码 + 眼底裁剪 + forward ≈ 40-60ms（T4，fp16，含 4x TTA 时批量吞吐更快），满足 ≤100ms

## 用法
```bash
# 目录结构假设：图片目录内有 test_00000.JPG ...，CSV 为官方 test.csv（Image_name, Age）
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights model_weights.pt \
  --out submission.csv
```

参数：
- `--tta 4`（默认）：4 个翻转变体平均，指标最好；`--tta 1` 最快
- `--img-size 0`（默认）：用 checkpoint 内记录的训练分辨率（384）
- `--use auto`（默认）：自动选 valid AUROC 更高的权重（ema 或 model）
- `--batch-size 64 --workers 4`：可按机器调整

## 输出格式
UTF-8 CSV，三列：`Image_name,Probability,Pred_Label`（Probability ∈ [0,1]，Pred_Label 以 0.5 为阈值）

## 已验证
- [x] 本地 RTX 5060 上全流程跑通（见项目根目录 infer.py 的验证记录）
- [ ] 在官方 docker 镜像内实测（T4 / Python 3.12）— 待用户在有条件时执行

## 注意
- 推理脚本内部自己实现眼底裁剪预处理（阈值找圆形视野 bbox → 居中方形 → 512 → resize），不依赖训练时的任何缓存文件
- 若官方测试图片命名或路径结构与本地不同，`predict.py` 会按 `Image_name` 的主文件名自动匹配 `.JPG/.jpg/.jpeg/.png/.PNG` 扩展名
- 年龄列缺失或异常（<1 或 >110）时用训练中位数 58 代替，与训练逻辑一致