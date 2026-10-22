# folds/ — issue #109 Task 1：train+valid 分层 5 折切分

> 关联工单：#109 [P0][训练] ConvNeXt-Base 16ep × 5-fold cross-fitting on train+valid
> 本目录仅交付**切分本身**（纯 CPU、零依赖、可复现）。实际训练在云端 24GB+ 卡完成，不占队长本地 5060。

## 切分方案

- 合并 `glaucoma_first/train.csv` + `glaucoma_first/valid.csv` = **57245** 行（正例 **1848**，正例率 0.0323）。
- 按 `True_Label` **分层**切 5 折：正例与负例各自用 `seed=42` 洗牌后按 `index % 5` 轮转分配，保证每折正例比例≈全局。
- 对第 `k` 折：
  - `fold{k}_valid.csv` = held-out 的 1/5（**OOF 评估用，绝不参与训练**）；
  - `fold{k}_train.csv` = 其余 4/5（训练时按 #109 用 `--extra-rows` 把本折 4/5 的 valid 行并入训练）。
- `test.csv` 完全不触碰（顺序/内容不变）。

## 文件

| 文件 | 说明 |
|------|------|
| `fold{0..4}_train.csv` | 每折训练集（其余 4 折，约 45795–45797 行） |
| `fold{0..4}_valid.csv` | 每折 held-out 验证集（约 11448–11450 行） |
| `make_folds.py` | 切分脚本（纯 Python 标准库，零依赖） |
| `split_manifest.json` | 每折行数/正例数/比例 + 切分参数（可复现） |

## 复现

```bash
cd folds
python make_folds.py                 # 默认自动探测 ../glaucoma_first/{train,valid}.csv
# 或指定数据目录：
python make_folds.py --data-dir ../glaucoma_first --out-dir . --seed 42 --n-folds 5
```

## 校验结论（split_manifest.json，verified=true）

- 5 折 valid 并集 == 全部 57245 行，且无重复、无 train/valid 重叠（脚本内断言已通过）。
- 每折正例比例严格一致（0.0322–0.0323），分层均衡。
- 每折 valid 约 11448–11450 行（正例 369–370），train 约 45795–45797 行（正例 1478–1479）。

| 折 | train 行 | train 正例 | valid 行 | valid 正例 |
|----|---------|-----------|---------|-----------|
| 0 | 45795 | 1478 | 11450 | 370 |
| 1 | 45795 | 1478 | 11450 | 370 |
| 2 | 45796 | 1478 | 11449 | 370 |
| 3 | 45797 | 1479 | 11448 | 369 |
| 4 | 45797 | 1479 | 11448 | 369 |

## 后续（训练侧，不在本目录）

- 每折从 ImageNet 初始化训 ConvNeXt-Base@384/cache512，16ep，lr 1e-4，head-lr 5e-4；
- 用 `--extra-rows` 把本折 4/5 的 valid 并入训练，held-out 1/5 单独 `predict_split` 评估；
- 汇总 OOF（57245 行，按原始 train+valid 顺序）、5 折 Spearman、fold 集成增益、test 概率（5855 行）。
