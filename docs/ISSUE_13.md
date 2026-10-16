# Issue #13 — ConvNeXt-Small@640 + 1024 缓存探测（复现手册 + 状态记录）

> 训练工单。代码侧已就绪，**训练需 GPU 机器 + cache768 + 热启动权重**（本机无，未实跑）。
> 本文档固化复现命令、验收判据、已有结果与合规红线，供训练机直接照做。

## 1. 代码侧验收结论（本 PR 已完成）

- `train.py`：`--arch convnext_small --img-size 640 --cache-root <dir> --init-from <pt> --lr/--head-lr` 全部支持，无需改代码。
- `preprocess.py`：`--out --size 1024 --draft 1536 --workers` 支持；**新增 `--data-root`**（原 DATA_ROOT 硬编码队长机器路径，训练机数据路径不同会直接失败——本 PR 补上可移植性）。

## 2. 复现命令

### 任务 A（必做）ConvNeXt-Small@640 on 768 缓存

```bash
# 热启动（强烈建议，省时）
python train.py --arch convnext_small --img-size 640 --cache-root D:\glaucoma_work\cache768 \
  --epochs 4 --batch-size 6 --accum 8 --loader workers --workers 3 \
  --lr 3e-5 --head-lr 1.5e-4 --init-from <cs_hi512/best.pt> --tag cs_hi640

# 或冷启动（显存 ~5.8GB；OOM 就 --batch-size 5 --accum 10）
python train.py --arch convnext_small --img-size 640 --cache-root D:\glaucoma_work\cache768 \
  --epochs 6 --batch-size 6 --accum 8 --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --tag cs_hi640
```

### 任务 B（探索）1024 缓存是否还有收益

```bash
# 生成 cache1024（~1.5h / ~15GB；--data-root 按训练机实际路径填）
python preprocess.py --data-root <原始图目录> --out D:\glaucoma_work\cache1024 \
  --size 1024 --draft 1536 --workers 8

# 探测 2 epoch
python train.py --arch convnext_small --img-size 640 --cache-root D:\glaucoma_work\cache1024 \
  --epochs 2 --batch-size 6 --accum 8 --lr 2e-5 --head-lr 1e-4 \
  --init-from <cs_hi512/best.pt> --tag cs_1024c
```

## 3. 验收判据与已有结果

- **判据（任务 B）**：2 epoch 后 AUPRC ≥ 任务 A 同 epoch 值 → 值得继续；否则分辨率到头，算力转投 #8 探索池。
- **已有结果（2026-10-16 12:55 状态更正）**：
  - Small@640 探测**目前未在运行**：三次尝试（06:51 / 09:31 / 11:51）都没跑出 epoch 2。
  - 唯一完成的 epoch-1 结果：`cs_hi640` EMA **AUPRC 0.86431，低于 Small@512 的 0.87256**（尚未兑现"先退步后提升"，按规律需 epoch 2 才能判定）。
  - 训练机当前在跑 RetFound ViT-L@224（12:24 启动），本方向暂停。
- **基线对照**：Small@512（768 缓存）TTA4 = AUROC 0.98994 / AUPRC **0.87256**。

## 4. ⚠️ 合规红线（必须遵守）

1. **RetFound（外部视网膜数据预训练）存在"仅可用主办方数据"合规冲突风险 → 不得进入任何提交文件**。
2. **ViT-L 体积超 500MB 复赛红线**（需瘦身，见 `docs/WEIGHTS.md`）。

## 5. 恢复步骤（等 RetFound 结果与合规判断明确后）

1. 向队长要 `cs_hi512/best.pt` 热启动权重；
2. 按任务 A 热启动命令跑 4 epoch（**别在 epoch1 退步时放弃**，规律是 epoch2 才显提升）；
3. epoch2 后 `python evaluate.py --ckpt <best.pt> --tta 4` 生成 probs + 指标；
4. 指标表按 `exp / img_size / cache-root / AUROC / AUPRC / balmax / epoch用时` 回填；
5. 权重上传 Release（命名规范见 #13 正文）。

## 6. Issue 回填模板

```
[#13] exp=cs_hi640 img_size=640 cache-root=cache768 AUROC=<x> AUPRC=<y> balmax=<z> epoch用时=<t>
结论=<继续 1024 探测 / 分辨率到头转 #8>
```
