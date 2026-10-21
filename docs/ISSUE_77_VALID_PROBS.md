# Issue #77 补充交付 — ViT-B@384 valid TTA4 逐图概率（5849 行）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/77 。
> 阶段1 训练曲线 / test 概率见 `docs/ISSUE_77_RESULTS.md`；本文件补的是**工单点名要、之前漏交**的那一项：
> `infer.py --split valid --tta 4` 的**逐图概率文件**（队长要用它在干净 valid 上离线算集成增益，只有指标数字算不了）。

## 交付物

| 文件 | 内容 | 校验 |
|---|---|---|
| `vitb_384_valid_probs_tta4.csv` | valid 5849 行 × 3 列（`Image_name, Probability, Pred_Label`） | 163,807 B；无 NaN；`Pred_Label` 与 `prob≥0.5` 全对 |

- 概率范围：0.000977 – 0.906586，mean 0.04654
- valid 正样本率：3.27%（193/5849）
- 与 test 侧口径一致：`Image_name` 直接取 `valid.csv` 顺序，不改名、不排序

## 复现命令

```bash
python infer.py --ckpt runs/vit_base_384_1020_221628_vitb_384/best.pt \
  --split valid --tta 4 --batch-size 16 --workers 4 \
  --out vitb_384_valid_probs_tta4.csv
```

- ckpt 内 `use=ema`、`cache_root=D:/glaucoma/cache512`、`csv_root=D:/123/glaucoma_first/glaucoma_first`，都是绝对路径，换机器时按 `paths.py` 覆盖即可
- 实测耗时：RTX 5060 Laptop 8GB，**174s**（TTA4 × 5849 ≈ 23.4k 次前向）
- 重算指标：**AUROC 0.96918 / AUPRC 0.67908**，与阶段1 回报的 0.96919 / 0.67918 差 ≤1e-4（cudnn 非确定性），口径一致

## 结论

单模 valid TTA4 = 0.9692，仍低于 0.990 门槛（Swin-B 0.99233 依旧是最强异构成员）。
按工单约定照交：本文件的用途是让队长在**干净 valid** 上算它与现有集成的真实增益，不靠猜。

- 增益 > 0 → 触发阶段2（train+all+pseudo，单次连续 schedule，禁止 LR 重启式续训）
- 增益 ≤ 0 → #77 判负结果收口，不再跑 16ep（16ep 配方更新里队长自己写的提前收手线是 valid TTA4 <0.985，本条 0.9692 已触发）
