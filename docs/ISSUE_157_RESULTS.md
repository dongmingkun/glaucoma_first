# #157 结果：Swin-B@384 train+all 第二 seed（seed123）+ 融合

> 训练与推理已在 **RTX 4090 24GB** 真机完成；train+all 无干净 valid，**不报任何 valid AUROC/AUPRC**。
> 候选**不自动提交**，由队长选定唯一一份。

## 一、结论摘要
| 项 | 结果 |
|---|---|
| Swin-B@384 seed123 训练 | ✅ 16ep，**19:35 → 22:47（3.20h，11,527s）**，loss **0.37313 → 0.08604** 单调下降 |
| 权重 | `runs/swinb_384_all_s123/last.pt` **699,524,517 B**（EMA）→ Release **`swinb_384_all_s123`** |
| test TTA4 | ✅ 5855 / 官方顺序 / 无 NaN / `[0,1]`（range 0.0001–1.0000），mean **0.05405**、pos@0.5 280 |
| **与新架构信号的距离** | vs seed7 **0.7519**、vs seed42 **0.6960**、vs s123 **0.7821**（top200 175–180）→ **明显比 RA 家族内部（0.89）更“远”，正是工单要的新信号** |
| 候选 | 7 个（ens10+new logit w10/20/30；B1/B2/B3 rank；B4 v3+new soup） |
| 判负线检查 | **未触发**（多数 rho 0.98–0.993、top200 194–198；仅 B2 的 0.9951 贴近线但 top200=197≠200） |
| 推荐 | **B1 `blend157_b1_w50201515.csv`**（md5 `b903cbb3`）；激进档 B3（`819657b7`） |

## 二、训练
```bash
python train.py --arch swin_base --img-size 384 --cache-root cache_all768 \
  --train-on-all --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --seed 123 \
  --loader workers --workers 8 --tag swinb_384_all_s123 --out-dir runs/swinb_384_all_s123
```
- 环境：RTX 4090 24GB，`cache_all768` = 57245/57245/5855，`train rows=57245 pos=1848`
- 单 epoch **703–740s**（比 ConvNeXt-Base@512 略慢），16ep 合计 **11,527s ≈ 3.20h**，VRAM 6.3G
- loss：ep1 0.37313 → ep8 0.13686 → ep16 **0.08604**

## 三、new Swin 的基础校验与“距离”
| 项 | 值 |
|---|---|
| 行数 / 顺序 / NaN | 5855 / ✅ / 0 |
| mean / pos@0.5 | 0.05405 / 280 |
| vs seed7（RA ConvNeXt） | **0.7519**，top200 178/200 |
| vs seed42 | **0.6960**，top200 175/200 |
| vs seed123 | **0.7821**，top200 180/200 |

对比参照：RA 三个 seed 之间 0.887–0.900、RA_avg 与 ens10 0.769。**Swin 与 ConvNeXt 家族的相关只有 0.70–0.78**，说明这确实是不同架构信号（工单要的那类），不是又一条 rho≈0.99 的打平线。

## 四、候选（全部 5855 / 官方顺序 / 无 NaN；对照 = 重建的 98.87 配方）
| 候选 | 构成 | mean | rho vs 98.87(重建) | top200 | md5 |
|---|---|---|---|---|---|
| `..._ens10_logit_w10` | ens10×0.9 + new Swin×0.1 | 0.07855 | 0.9838 | 197 | `3d26aea4` |
| `..._ens10_logit_w20` | ens10×0.8 + new Swin×0.2 | 0.07455 | 0.9846 | 196 | `96a7aac7` |
| `..._ens10_logit_w30` | ens10×0.7 + new Swin×0.3 | 0.07099 | 0.9812 | 196 | `6e2b5bc0` |
| **`blend157_b1_w50201515`（推荐）** | **0.50 ens10 + 0.20 RA_avg + 0.15 Swin_pair + 0.15 new Swin** | 0.50000 | **0.9933** | **198** | `b903cbb3` |
| `blend157_b2_w55201015` | 0.55 / 0.20 / 0.10 / 0.15 | 0.50000 | 0.9951 | 197 | `936ead3d` |
| `blend157_b3_w45201025`（激进） | 0.45 / 0.20 / 0.10 / **0.25** | 0.50000 | 0.9850 | 196 | `819657b7` |
| `blend157_b4_soup_w50201515` | B1 但 new Swin → soup(v3+new) | 0.50000 | 0.9933 | 194 | `4b114424` |

> **对照不是原提交文件**：98.87 的 `FINAL_SUBMIT_20261028/submission.csv`（md5 `4bf37b7c`）只在队长本地 D 盘，仓库与云端均无。本表用台账记载的同配方重建：
> `0.60 rank(ens10) + 0.10 rank(seedavg) + 0.15 rank((RA_best+RA_last)/2) + 0.15 rank(Swin_pair)`。
> 若提供原 CSV，rho/top200 可一键重算。

## 五、推荐
**首选 B1（`blend157_b1_w50201515.csv`，md5 `b903cbb3`）**
1. **保留已验证结构**：98.87 的核心是 `ens10 + RA_avg + Swin_pair`，B1 只把 new Swin 以等额（0.15）加入，Swin 总暴露 0.30；
2. 与 98.87 的 rho 0.9933 / top200 **198/200**：主体不动，但比 seed/soup 那些 0.997+ 的候选**真的挪了 2 个头部名额**；
3. 未触发判负线。

**激进档 B3（`819657b7`，Swin 总暴露 0.35）**：若想最大化新架构信号、愿意承担更大结构改动（rho 0.9850 / top200 196）。

**不建议单独投**：`B2`（rho 0.9951 / 197，几乎就是 98.87 的微调）、`_ens10_logit_w*`（完全丢掉 RA_avg，等于退回 98.84 的结构）、`B4`（top200 掉到 194，soup 把 Swin_pair 的信息吃掉了）。

**优先级提醒**：本单的新 Swin 与 Swin_pair **同配方不同 seed**，边际可能有限（历史两次 seed 平均都打平）。真正的新架构信号是 **#161 的 ConvNeXt-Large@384**——如果只有一个配额，我建议优先投 #161 的 Large 候选，本单留作备选。

## 六、交付物
- 权重：Release **`swinb_384_all_s123`** → `swinb_384_all_s123_last.pt`（699,524,517 B）
- 概率（本 PR）：`swinb_384_all_s123_test_tta4.csv` + 3 个 ens10 融合 + 4 个 rank/soup 组合 + 对照 `blend157_ctl_987.csv`
- 训练曲线：`swinb_384_all_s123_log.csv`

## 七、事故与修正（如实）
1. **串行链首启失败**：`train_161.sh` 被我 patch 时用了 Windows 默认换行，写出 **CRLF**，云端 bash 报 `set: -: invalid option` / `ambiguous redirect`，链在 22:51 瞬间返回 0。已把脚本转成 LF 并手动重启，**#161 训练 22:56 才真正开始**（比原计划晚 5 分钟，不影响 08:00）。
2. 为给 Large 腾磁盘，链在 22:51 删除了 `runs/swinb_384_all_s123/best.pt`（`last.pt` 保留并已上 Release），另外删除了可再生的 soup 与已在 Release 的 seed7/seed42 `last.pt`。
