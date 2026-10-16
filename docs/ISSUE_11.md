# Issue #11: [P0][训练] 高分辨率模型矩阵：ConvNeXt-Small@512 + Base@512（768 缓存）

> 状态：closed ｜ 指派人：Yulingling-sc
> 本文件是 GitHub issue #11 的离线快照（正文 + 全部评论），供队友的 AI 工具/离线环境直接读取。
> 权威内容以 GitHub issue 为准。

## 正文

## 背景
**768px 缓存 + 512 输入已验证是当前最强杠杆**（队长机器实测）：
- ConvNeXt-Small@384（512缓存）：TTA4 AUROC 0.98994 / AUPRC 0.86371
- ConvNeXt-Small@512（**768缓存**）：TTA4 AUROC 0.98994 / AUPRC **0.87256**（+0.0089）
- 当前最优集成（hi512×1.5 + Base + B3×3）：AUPRC **0.87733** / blend 93.42 / 校准后 balACC 0.9584

## 前置（一次性，约 1 小时）
```bash
# 数据放到 glaucoma_first/{train,valid,test} 后：
python preprocess.py --out D:\glaucoma_work\cache768 --size 768 --draft 1152 --workers 8
export GLAUCOMA_CACHE=D:\glaucoma_work\cache768    # CMD 用 set
python verify_cache.py                                # 应无 missing / bad
```
> **参数必须与上面一致**（size=768, draft=1152），否则裁剪结果与队长机器不同、无法直接集成。

## 任务 A（必做）ConvNeXt-Small@512，8 epoch
```bash
python train.py --arch convnext_small --img-size 512 --cache-root D:\glaucoma_work\cache768 \
  --epochs 8 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --tag cs_hi512
```
显存约 5GB，~60 分钟/epoch。
> ⚠️ **第 1 个 epoch 验证分会先退步、第 2 个 epoch 才显现提升**（适应新分辨率），不要中途放弃。

## 任务 B（有余力）ConvNeXt-Base@512，6 epoch
```bash
python train.py --arch convnext_base --img-size 512 --cache-root D:\glaucoma_work\cache768 \
  --epochs 6 --batch-size 5 --accum 8 --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --tag cb_hi512
```
显存约 5.5GB（OOM 就把 batch-size 降到 4、accum 提到 10）。可向队长要 Base@384 权重用 `--init-from` 热启动，省一半时间。

## 交付物（完成后提 PR）
1. `runs/<exp>/best.pt` + `runs/<exp>/probs_valid_full_tta4.npy`（`python evaluate.py --ckpt <best.pt> --tta 4` 生成）
2. PR 里写：指标表（exp / img_size / **cache-root** / AUROC / AUPRC / balmax / epoch 用时）+ 完整复现命令 + 权重共享链接（Release 附件或群文件，**不要入库**）
3. 基线对照：Small@512 AUPRC ≥ 0.8726 为达标

## 参考
- 环境/流程：`docs/TEAM_HANDOFF.md`、提 PR：`docs/HOW_TO_PR.md`
- 铁律：同时只跑一个 GPU 任务；训练固定 `--workers 3`；权重不入库

## 评论

### @dongmingkun (2026-10-16 04:13)

## 关闭说明（交付物已达成，2026-10-16）

**任务 A（必做）ConvNeXt-Small@512（768 缓存）** ✅　**任务 B ConvNeXt-Base@512（768 缓存）** ✅

证据：
- 两个模型均已训练完成并进入共享通道：Release **[weights-1016](https://github.com/dongmingkun/glaucoma_first/releases/tag/weights-1016)** 含
  `convnext_small_512_cache768`（TTA4 AUPRC **0.8752**，达标线 0.8726 ✅）与 `convnext_base_512_cache768`（**0.8765**，当前最强单模）
- 结果与集成对比已记录（PROJECT_NOTES：「记录 Base@512 训练结果（新最强单模 AUPRC 0.87646）与集成对比」）
- 两个模型已在当前最优提交候选 **ens7**（hi512_v2×2 + Base@384 + B3×3）中使用 → 交付物已处于生产使用中

注意（沿用 #14 结论）：Release 中的 fp16 为**瘦身推理权重，不能用于续训**；需要续训请向队长索取原始 best.pt。