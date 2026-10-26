# seed7 + seed42 model soup runbook（test-only，10 分钟内可出候选）

> 目的：在**权重空间**平均 #138 seed7 与 #145 seed42 两个同架构 Base@512 RA train+all
> checkpoint，得到一个新单成员，再与 ens10 融合。与 `tools/seed_average.py` 的**概率平均**
> 互补；两者都只能在线上回答是否提分（train+all 无干净 valid）。
>
> 工具：`tools/seed_soup.py`（CPU-only，自测 `--self-test` PASS）。不占本地 GPU。

## 前置

- 云端同时有：
  - `runs/cnxb_512_e16_ra_all/last.pt`（#138 seed7，Release `ra_all-1025`）
  - `runs/cnxb_512_e16_ra_all_s42/last.pt`（#145 seed42，Release `ra_all_s42`）
- `cache_all768`（test 5855；与训练同口径 768→512）；
- main 含本工具 + `tools/oof_predict.py` / `tools/fusion_ens10.py` / `check_train_on_all.py`。

## 1) soup（CPU，约 1–2 分钟）

```bash
python tools/seed_soup.py \
  --ckpts runs/cnxb_512_e16_ra_all/last.pt \
          runs/cnxb_512_e16_ra_all_s42/last.pt \
  --weights 0.5 0.5 \
  --out runs/cnxb_512_e16_ra_all_soup/last.pt
```

- 同时平均 `ema` 与 `model`；推理固定用 `ema`；
- 整数 buffer（`num_batches_tracked`）取 seed7 的值；
- 输出约 705MB；`auroc/auprc` 已置 -1，防止误报 valid。

## 2) test TTA4（4090 约 2–3 分钟）

```bash
python tools/oof_predict.py \
  --ckpt runs/cnxb_512_e16_ra_all_soup/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --use ema \
  --out oof/cnxb_512_e16_ra_all_soup_test_tta4.csv
```

硬校验：5855 行 / 官方顺序 / 无 NaN / [0,1]；mean 与 pos@0.5 应与 seed7/seed42 同量级
（约 0.05 / 4.7–4.8%）。若明显塌缩（如 mean <0.02 或 >0.15），直接判定 soup 失败。

## 3) check（可选，但建议）

```bash
python check_train_on_all.py \
  --ckpt runs/cnxb_512_e16_ra_all_soup/last.pt \
  --ref runs/cnxb_512_e16_ra/last.pt --tta 4 --cache-root cache_all768
```

- 对照 #128 train-only，看 Spearman / mean 漂移 / top200；REVIEW/PASS 如实记录。
- 该 check 不是提分门槛，只用于排除漂移型 soup。

## 4) 融合候选

```bash
python tools/fusion_ens10.py \
  --ens10 submission_20251018_ens10_cal.csv \
  --new oof/cnxb_512_e16_ra_all_soup_test_tta4.csv \
  --out-dir oof
```

生成 `oof/cnxb_512_e16_ra_all_soup_test_tta4_ens10_{logit,rank}_w{10,20,30}.csv`。

同时算与三个参照的排序一致性（CPU）：

```bash
python tools/oof_report.py metrics --names-only --labels glaucoma_first/test.csv \
  --probs oof/cnxb_512_e16_ra_all_soup_test_tta4.csv \
          cnxb_512_e16_ra_all_test_tta4.csv \
          cnxb_512_e16_ra_all_s42_test_tta4.csv
```

- soup vs seed7/seed42 的 Spearman：同配方应较高（>0.90 合理；<0.80 视为 soup 漂移，弃用）；
- soup vs 98.84（seed7 w10）：top200 重叠建议 ≥195/200；
- 候选不自动提交；线上失败不降低当前最高 98.84。

## 5) 判定与交付

| 结果 | 动作 |
|---|---|
| soup mean/pos 正常、check 不漂移、融合 w10 线上 ≥98.84 | 保留 soup 候选，考虑 w20 |
| soup 明显塌缩或 Spearman <0.80 | 弃用，保留 seed 概率平均 |
| 线上 <98.84 | 回退 98.84，不作为正式成员 |

交付物：
- `runs/cnxb_512_e16_ra_all_soup/last.pt`（705MB，Release/群文件）；
- `oof/cnxb_512_e16_ra_all_soup_test_tta4.csv`（5855）；
- `oof/..._ens10_{logit,rank}_w{10,20,30}.csv`（6 个候选）；
- 结果回填 `docs/ISSUE_RA_ALL_S42_RESULTS.md` 或本文件末尾。

## 时间

soup CPU 1–2 min + test TTA4 2–3 min + check/fusion CPU 1 min；**总计 <10 min**（4090）。
