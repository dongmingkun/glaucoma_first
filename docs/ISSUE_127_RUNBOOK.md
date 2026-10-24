# issue #127 Runbook — 用 #109 的 3 折 OOF 做干净 stacking / 权重搜索

> 本文件是 #127 的**可执行手册 + 工具预交付**。核心逻辑是 **CPU-only**（2–4h），
> 但成员 OOF 的生成需要一次 GPU 推理（约 1h，可复用 4090）。不占用队长本地 5060。
>
> 关键原则：**只在 held-out OOF 上做 nested 3-fold CV 选权重/正则，禁止在 valid 上直接调参**；
> 不给 train+all 成员生成 OOF（它们看过 valid，会泄漏）。

## 依赖
1. **#126 完成**：产出 #109 的 `oof/oof_probs.npy`（3 折 Small，57245）+ `oof/test_probs_tta4.csv`（5855）；
2. 现有 **train-only** 成员的权重（用于生成 OOF）；train+all 成员**不参与** OOF。

## 产物
- `tools/stack_oof.py`：OOF stacking / 权重搜索（numpy-only，含 `--self-test`）；
- `tools/make_member_oof.sh`：按 `folds3` 生成单个 train-only 成员的 57245 OOF + test 概率；
- `oof/stack_test_tta4.csv`（5855）、`oof/stack_oof_probs.npy`（57245）、`oof/stack_weights.json`；
- `docs/ENSEMBLE_OOF_RESULTS.md`：nested CV、OOF 指标、与 ens9/ens10 对比、bootstrap、失败原因。

## 步骤

### 0) 自测（CPU，秒级，不需要数据）
```bash
python tools/stack_oof.py --self-test
```

### 1) 生成各 train-only 成员的 OOF（GPU 推理，一次性）
对每个成员（`cb384`/`Base@384 8ep`/`Base@512 4ep`/`Swin-B seed7/seed123`）：
```bash
# @384 成员（cache512 合成 cache_all）
NAME=cb384 CKPT=runs/<cb384>/best.pt CACHE=cache_all IMG=384 bash tools/make_member_oof.sh
# @512 成员（cache768 合成 cache_all768）
NAME=base512_4ep CKPT=runs/<base512_4ep>/best.pt CACHE=cache_all768 IMG=512 bash tools/make_member_oof.sh
# 基线 ens9（若无现成权重，可用其 test/valid 概率拼装；见下方“ens9”）
```
- `cache_all` / `cache_all768` 的合成：`cp -rs` 把 train/valid 全量（57245）链进 train/ 与 valid/（见 `folds3/README.md`）；
- **不要**对 train+all 成员跑本步。

### 2) 跑 stacking（CPU，2–4h）
```bash
python tools/stack_oof.py \
  --labels glaucoma_first/train.csv glaucoma_first/valid.csv \
  --folds folds3 --test-names glaucoma_first/test.csv \
  --base oof/oof_probs.npy::oof/test_probs_tta4.csv \
  --member cb384=oof/cb384_oof.npy::oof/cb384_test_tta4.csv \
  --member base384_8ep=oof/base384_8ep_oof.npy::oof/base384_8ep_test_tta4.csv \
  --member base512_4ep=oof/base512_4ep_oof.npy::oof/base512_4ep_test_tta4.csv \
  --member swin_s7=oof/swin_s7_oof.npy::oof/swin_s7_test_tta4.csv \
  --member swin_s123=oof/swin_s123_oof.npy::oof/swin_s123_test_tta4.csv \
  --ens9 ens9=oof/ens9_oof.npy::oof/ens9_test_tta4.csv \
  --out-prefix oof/stack --bootstrap 2000 \
  --report docs/ENSEMBLE_OOF_RESULTS.md
```
- 成员规格用 `NAME=OOF::TEST`（`::` 分隔，兼容 Windows 盘符 `C:`）；
- `OOF` 可为 `.npy`（已按官方顺序）或 `.csv`（按 `Image_name` 自动对齐）；
- 也可 `--autodiscover` 自动发现 `oof/*_oof.npy`（配同名 `*_test_tta4.csv`）。

## 方法（脚本内置，CV 自动选优）
| 方法 | 说明 | 拟合参数 |
|---|---|---|
| `rank` | 各成员 OOF 秩 [0,1] 等权平均 | 0 |
| `weight` | 非负权重单纯形上坐标上升，最大化 OOF AUROC（只用训练折标签） | m |
| `logreg` | logit(成员概率) 上 L2 逻辑回归，λ 由内层 CV 选 | m+1 |

nested 3-fold CV：外层 held-out 折评估，内层只在其余折拟合/选参 → 得到诚实的 OOF 估计。

## 验收 / 门槛（#127）
- **ΔAUROC vs ens9 ≥ +0.0003 且 paired bootstrap P(Δ>0) ≥ 0.90** 才算有希望；
- test 5855 行、顺序一致、无 NaN；权重/代码可复现；
- OOF 必须 held-out（禁止 valid-selected best.pt；禁止 train+all 成员）；
- 若 OOF 无增益，**如实记录负结果，不提交** `stack_test_tta4.csv` 作为候选。

## 已做的 CPU 验证
- `python tools/stack_oof.py --self-test`：合成 57245 样本/4 成员，验证 AUC/AP/bootstrap/nested CV，断言全过；
- 端到端冒烟：合成 CSV+npy 跑完整 CLI，产物 `stack_test_tta4.csv`（行数对齐）、`stack_oof_probs.npy`、`stack_weights.json`、report 均正确生成。

## 备注
- #127 正文要求「禁止在 valid 上直接调参」——本工具全程只用 OOF，官方 valid 仅作为 OOF 的一部分（每行由未见它的模型预测），不参与选择。
- #109 的 3 折 Small OOF 是 #111 蒸馏的教师来源，本单的 stack 结果与 #111 相互独立，可并行。
