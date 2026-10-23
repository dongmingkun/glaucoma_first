# folds3/ — issue #118 Task 1：train+valid 分层 3 折切分（Swin-B 异构 fold 多样性）

> 关联工单：#118 [P1][训练] Swin-B@384 16ep × 3-fold on train+valid（异构 fold 多样性，不占 5060）
> 本目录仅交付**切分本身**（纯 CPU、零依赖、可复现，复用 #109 的 `folds/make_folds.py`）。实际 Swin-B 训练在云端 24GB+ 卡完成，不占队长本地 5060。

## 切分方案

- 合并 `glaucoma_first/train.csv` + `glaucoma_first/valid.csv` = **57245** 行（正例 **1848**，正例率 0.0323）。
- 按 `True_Label` **分层**切 3 折（`seed=42`，与 `folds/` 的 5 折同种子、同洗牌逻辑）：正例与负例各自洗牌后按 `index % 3` 轮转分配，保证每折正例比例≈全局。
- 对第 `k` 折：
  - `fold{k}_valid.csv` = held-out 的 1/3（**OOF 评估用，绝不参与训练**）；
  - `fold{k}_train.csv` = 其余 2/3（训练集）。
- `test.csv` 完全不触碰（顺序/内容不变）。

## 文件

| 文件 | 说明 |
|------|------|
| `fold{0..2}_train.csv` | 每折训练集（其余 2 折，约 38163–38164 行） |
| `fold{0..2}_valid.csv` | 每折 held-out 验证集（约 19081–19082 行） |
| `fold{k}_csvroot/train.csv`、`fold{k}_csvroot/valid.csv` | 训练侧直接可用的 csv-root（复制自上面两个 CSV，列名 `Image_name,Age,True_Label`） |
| `split_manifest.json` | 每折行数/正例数/比例 + 切分参数（可复现） |

> `make_folds.py` 本身在 `folds/make_folds.py`（#109 交付），本目录仅放切分结果与训练侧接入目录，避免重复维护脚本。

## 复现

```bash
# 在仓库根目录执行（脚本按 folds/ 目录解析 --out-dir，所以要写 ../folds3）
python folds/make_folds.py --n-folds 3 --out-dir ../folds3 --data-dir glaucoma_first --seed 42
```

## 校验结论（split_manifest.json，verified=true）

- 3 折 valid 并集 == 全部 57245 行，且无重复、无 train/valid 重叠（脚本内断言已通过）。
- 每折正例比例严格一致（0.032282–0.032283），分层均衡。
- 每折 valid 约 19081–19082 行（正例 616），train 约 38163–38164 行（正例 1232）。

| 折 | train 行 | train 正例 | valid 行 | valid 正例 |
|----|---------|-----------|---------|-----------|
| 0 | 38163 | 1232 | 19082 | 616 |
| 1 | 38163 | 1232 | 19082 | 616 |
| 2 | 38164 | 1232 | 19081 | 616 |

## 训练侧接入（重要：不要用 `--extra-rows`）

与 #109 的 `folds/README.md` 一致，`train.py --extra-rows` 要求列名 `Label` 且 `FundusDS` 只在 `cache_root/train` 下找图；本切分 CSV 列名是 `True_Label`，且混入了 `valid_*.jpg`（原图在 `cache_root/valid`）。直接传 `--extra-rows` 会 KeyError / FileNotFoundError。

正确做法（每折一个 csv-root + 一个合并 image cache，不需要改 `train.py`）：

```bash
# 1) 每折 csv-root（本目录已生成 fold{k}_csvroot/，可直接用）
#    fold{k}_csvroot/train.csv、fold{k}_csvroot/valid.csv 已就位

# 2) 合并 image cache（Linux 符号链接，不占磁盘）：
#    cache_all/train 和 cache_all/valid 都放 train+valid 的全部 57245 张
mkdir -p cache_all/train cache_all/valid
cp -rs /path/to/cache512/train/. cache_all/train/
cp -rs /path/to/cache512/valid/. cache_all/train/
cp -rs /path/to/cache512/train/. cache_all/valid/
cp -rs /path/to/cache512/valid/. cache_all/valid/

# 3) 训练第 k 折（Swin-B@384，从 ImageNet 初始化，16ep）
python train.py --arch swin_base --img-size 384 --cache-root cache_all \
  --csv-root folds3/fold${k}_csvroot --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 42 --tag swin_fold${k}
```

- `FundusDS` 用 `cache_all/train`、`cache_all/valid` 下的图；行由 csv-root 的 `train.csv`/`valid.csv` 决定，所以两张表可以混排原始 train/valid。
- **OOF 一律用 `last.pt`（固定 16ep 的最终 EMA），不要用 `best.pt`**：`train.py` 默认按 held-out valid AUROC 选 best，等于在 OOF 折上做模型选择，会高估 OOF。要么固定 16ep 用 `last.pt`，要么在训练折内部再切一个小验证集做 early stop。
- **不要用 `--init-from` 指向 valid-selected 的 best.pt**，否则 OOF 被污染。
- 16GB 卡时 Swin-B 用 `bs4×accum12`。

## 后续（训练侧，不在本目录）

- 每折从 ImageNet 初始化训 Swin-B@384，16ep，lr 1e-4，head-lr 5e-4；
- held-out 1/3 单独评估（用 `last.pt`）；
- 汇总 OOF（57245 行，按原始 train+valid 顺序）、3 折 Spearman、与 ens9 / Base fold 的 Spearman、test 概率（5855 行）；
- 交付 `docs/ISSUE_118_RESULTS.md` 与 3 个 checkpoint（Release/群文件）。
