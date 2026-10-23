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

## 训练侧接入（重要：不要用 `--extra-rows`）

`train.py --extra-rows` 要求列名 `Label`，且 `FundusDS` 只在 `cache_root/train` 下找图；本切分的 `fold{k}_train.csv` 列名是 `True_Label`，且混入了 `valid_*.jpg`（原图在 `cache_root/valid`）。直接传 `--extra-rows` 会 KeyError / FileNotFoundError。

正确做法是 **每折一个 csv-root + 一个合并 image cache**，不需要改 `train.py`：

```bash
# 1) 每折 csv-root：train.csv/valid.csv 直接复制 fold CSV
for k in 0 1 2 3 4; do
  mkdir -p folds/fold${k}_csvroot
  cp folds/fold${k}_train.csv folds/fold${k}_csvroot/train.csv
  cp folds/fold${k}_valid.csv folds/fold${k}_csvroot/valid.csv
done

# 2) 合并 image cache（Linux 符号链接，不占磁盘）：
#    cache_all/train 和 cache_all/valid 都放 train+valid 的全部 57245 张
mkdir -p cache_all/train cache_all/valid
cp -rs /path/to/cache512/train/. cache_all/train/
cp -rs /path/to/cache512/valid/. cache_all/train/
cp -rs /path/to/cache512/train/. cache_all/valid/
cp -rs /path/to/cache512/valid/. cache_all/valid/

# 3) 训练第 k 折（从 ImageNet 初始化，16ep）
python train.py --arch convnext_base --img-size 384 --cache-root cache_all \
  --csv-root folds/fold${k}_csvroot --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 42 --tag cnxb_fold${k}
```

- `FundusDS` 用 `cache_all/train`、`cache_all/valid` 下的图；行由 csv-root 的 `train.csv`/`valid.csv` 决定，所以两张表可以混排原始 train/valid。
- 评估时用 `predict_split(..., split="valid", ...)` 读 `fold{k}_csvroot/valid.csv`。
- **OOF 一律用 `last.pt`（固定 16ep 的最终 EMA），不要用 `best.pt`**：`train.py` 默认按 held-out valid AUROC 选 best，等于在 OOF 折上做模型选择，会高估 OOF。要么固定 16ep 用 `last.pt`，要么在训练折内部再切一个小验证集做 early stop。
- **不要用 `--init-from` 指向 valid-selected 的 best.pt**，否则 OOF 被污染。
- 不要用 `--extra-rows`。如果坚持用它，需要把列名改成 `Label` 并给 `FundusDS` 加跨 split 的路径回退，属于代码改动，不在本目录范围。

## 后续（训练侧，不在本目录）

- 每折从 ImageNet 初始化训 ConvNeXt-Base@384/cache512，16ep，lr 1e-4，head-lr 5e-4；
- held-out 1/5 单独 `predict_split` 评估（用 `last.pt`）；
- 汇总 OOF（57245 行，按原始 train+valid 顺序）、5 折 Spearman、fold 集成增益、test 概率（5855 行）。
