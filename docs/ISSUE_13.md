# Issue #13: [P0][训练] 高分辨率进阶：ConvNeXt-Small@640 + 1024 缓存探测

> 状态：open ｜ 指派人：Yulingling-sc
> 本文件是 GitHub issue #13 的离线快照（正文 + 全部评论），供队友的 AI 工具/离线环境直接读取。
> 同步时间：2026-10-16 ｜ 权威内容以 GitHub issue 为准。

## 正文

## 背景
分辨率是**已验证最强、且尚未见饱和**的杠杆：
| 配置 | 缓存 | TTA4 AUPRC |
|---|---|---|
| Small@384 | 512 | 0.86371 |
| Small@512 | **768** | **0.87256** |

## 任务 A（必做）ConvNeXt-Small@640 on 768 缓存
```bash
python train.py --arch convnext_small --img-size 640 --cache-root D:\glaucoma_work\cache768 \
  --epochs 6 --batch-size 6 --accum 8 --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --tag cs_hi640
```
- 显存约 5.8GB（OOM 就 `--batch-size 5 --accum 10`）；~100 分钟/epoch
- **强烈建议热启动省时**：向队长要 `cs_hi512/best.pt`，加 `--init-from <路径> --lr 3e-5 --head-lr 1.5e-4 --epochs 4`
- ⚠️ 规律：**epoch1 验证分会先退步、epoch2 才显现提升**（适应新分辨率），别中途放弃

## 任务 B（探索）1024 缓存是否还有收益
```bash
python preprocess.py --out D:\glaucoma_work\cache1024 --size 1024 --draft 1536 --workers 8   # ~1.5h / ~15GB
python train.py --arch convnext_small --img-size 640 --cache-root D:\glaucoma_work\cache1024 \
  --epochs 2 --batch-size 6 --accum 8 --lr 2e-5 --head-lr 1e-4 --init-from <cs_hi512/best.pt> --tag cs_1024c
```
**判据**：2 epoch 后 AUPRC ≥ 任务 A 同 epoch 值 → 值得继续；否则分辨率到头，收手把算力投到 #8 探索池。

## 交付物（提 PR）
1. `best.pt` + `probs_valid_full_tta4.npy`（`python evaluate.py --ckpt <best.pt> --tta 4` 生成）
2. 指标表：`exp / img_size / **cache-root** / AUROC / AUPRC / balmax / epoch 用时`
3. 完整复现命令 + 权重上传到 Release（命名规范见 #13）
4. 基线对照：Small@512（768缓存）TTA4 = AUROC 0.98994 / AUPRC **0.87256**

## 评论

### @dongmingkun (2026-10-16 04:56)

## 状态更正（2026-10-16 12:55，防止误判进度）

- Small@640 探测**目前未在运行**：三次尝试（06:51 / 09:31 / 11:51）都未跑出 epoch 2（前两次无产物，第三次在 epoch 1 中途被替换）
- 唯一一次完成的 epoch-1 结果：`cs_hi640` EMA AUPRC **0.86431，低于 Small@512 的 0.87256**（尚未兑现本工单正文预期的"先退步、后提升"，按规律需 epoch 2 才能判定）
- 训练机当前在跑的是 **RetFound ViT-L@224**（12:24 启动），本工单方向暂停
- 两个与 RetFound 相关的约束会影响本方向的取舍：① RetFound 用**外部视网膜数据**预训练，与"仅可用主办方数据"存在合规冲突风险；② ViT-L 体积（fp32 ≈1.2GB / fp16 ≈0.6GB）**超 500MB 复赛红线**，需量化瘦身才可能用于提交
- 结论：本工单保持 open；等 RetFound 结果与合规判断明确后，再决定 640 探测是否继续（也可直接关闭，把算力留给集成优化）