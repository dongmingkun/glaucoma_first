# Issue #138 结果记录 — Base@512 16ep + RA 的 train+all 57245 家族成员（test-only 探索）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/138 。本文件为**预交付模板**，
> 云端跑完回填，未填处标 `待填`。
> **声明：本成员 train+all（57245 全量训练），没有干净 valid；只用 test 反馈
> （check_train_on_all + ens10 融合候选），不声称 clean valid 增益；
> 最终是否提分由一次线上提交回答，失败不降低 98.82。**

## 配置

| 项 | 值 |
|---|---|
| 命令 | `train.py --arch convnext_base --img-size 512 --cache-root cache_all768 --train-on-all --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 7 --tag cnxb_512_e16_ra_all` |
| 机器 / 耗时 | 待填（4090 24GB，工单估 3–4h） |
| 数据 | `cache_all768` 57245/57245/5855（#134 建好复用；train+valid 合并 seed 洗牌） |
| 对照 | `runs/cnxb_512_e16_ra/last.pt`（#128 train-only，valid 0.99284） |

## train+all 预检

- [ ] `--train-on-all` 日志：rows 51396 → 57245（+11.4%）（见 run138.log）；
- [ ] 无 valid 指标虚报（last.pt/best.pt 均为 EMA，val 指标无意义）。

## check_train_on_all（对照 #128 train-only）

```
待填（run138.log 中 check 段落：Spearman / mean 漂移 / top-k 重叠 / 判定 PASS/REVIEW）
```

| 项 | 值 | 判定参考 |
|---|---|---|
| Spearman(vs #128) | 待填 | <0.8 或塌缩 → 判负 |
| mean 漂移 | 待填 | 家族正常 13–16% |
| top200 重叠 | 待填 | |

## ens10 融合候选（rank/logit × w=0.1/0.2/0.3）

```bash
python tools/fusion_ens10.py   # 产出 oof/cnxb_512_e16_ra_all_test_tta4_ens10_{rank,logit}_w{10,20,30}.csv
```

| 项 | 值 |
|---|---|
| Spearman(RA_all, ens10) | 待填 |
| mean 漂移 / top200 重叠 | 待填 |
| 6 个候选 CSV | 待填（路径如上） |

## 验收（对照 #138）

- [ ] test 5855 行、官方顺序、无 NaN、[0,1]；
- [ ] train-on-all 预检 PASS；check 相关性合理；
- [ ] 明确 test-only 家族候选定位（本文件头部声明）。

## 交付物

- `runs/cnxb_512_e16_ra_all/last.pt` → Release（待挂）；
- `oof/cnxb_512_e16_ra_all_test_tta4.csv`（5855）；
- check 报告 + 6 个融合候选 CSV（不入库/入 PR 视大小，CSV 5855 行可入库）；
- 本文件回填。
