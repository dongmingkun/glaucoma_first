# #152 执行手册：seed123 第 3 个 train+all seed（3h 训练 + 3-seed 平均/soup）

> 目标：在 **2026-10-28 08:00（北京时间）** 前产出第 3 个 train+all seed 的候选，供队长选定**唯一**提交。
> 现状：线上最好 **98.84**（ID 20380 / 21547）——seed7 单 seed、seed7+seed42 概率平均（w10/w20）、seed7+seed42 soup 均停在 98.84，**单/双 seed 线已饱和**。
> 本单**不自动提交**；训练/推理需 **4090 24GB**（本地 5060 不硬跑 16ep）。

## 一、算力与前置
| 项 | 要求 |
|---|---|
| GPU | **4090 24GB**（约 3h 训练；实例不在需先租 4–6h） |
| 缓存 | `cache_all768`：train/valid 各 **57245**、test **5855**（软链合成即可，见下） |
| 权重 | seed7 `runs/cnxb_512_e16_ra_all/last.pt`（Release `ra_all-1025`）；seed42 `runs/cnxb_512_e16_ra_all_s42/last.pt`（Release `ra_all_s42`） |
| 基线 | `submission_20251018_ens10_cal.csv`（98.82）；98.84 参照 = `seed7_s42_ens10_logit_w20.csv`（ID 21547）/ `cnxb_512_e16_ra_all_ens10_logit_w10.csv`（ID 20380） |

> 下载踩坑（10-26 实测）：GitHub Release 直连仅 **28 KB/s**；`ghproxy.net`/`gh-proxy.com`/`ghproxy.cc` 均不通；**`ghfast.top` + `aria2c -x16 -s16 -k1M` ≈1 MB/s**，705MB 约 2 分钟。
> ```bash
> aria2c -x16 -s16 -k1M --file-allocation=none -o last.pt \
>   "https://ghfast.top/https://github.com/dongmingkun/glaucoma_first/releases/download/ra_all-1025/cnxb_512_e16_ra_all_last.pt"
> ```
> `cache_all768` 软链合成（不复制图像，2–3 秒）：
> ```bash
> python - <<'PY'
> import os
> src='cache768'; dst='cache_all768'
> for s,d in [('train','train'),('valid','train'),('train','valid'),('valid','valid'),('test','test')]:
>     dd=os.path.join(dst,d); os.makedirs(dd,exist_ok=True)
>     sd=os.path.join(src,s)
>     for fn in os.listdir(sd):
>         lp=os.path.join(dd,fn)
>         if not os.path.lexists(lp): os.symlink(os.path.join(sd,fn), lp)
> for d in ('train','valid','test'): print(d, len(os.listdir(os.path.join(dst,d))))
> PY
> ```

## 二、一键执行
```bash
PYTHON=/root/miniconda3/bin/python LOG=/root/run152.log nohup bash tools/pipeline_152.sh > /root/run152.nohup 2>&1 < /dev/null &
tail -f /root/run152.log
```
脚本步骤（fail-fast，可重入：训练/soup 已存在则跳过）：

| 步 | 内容 | 产出 |
|---|---|---|
| 0 | `git pull`（失败不阻塞） | — |
| 1 | `cache_all768` 计数校验 + 两个 seed ckpt 存在性 | — |
| 2 | **训练 seed123** 16ep（`--train-on-all --randaugment 2 9 --seed 123`） | `runs/cnxb_512_e16_ra_all_s123/{last,best}.pt` |
| 3 | seed123 test TTA4（`--use ema`） | `oof/cnxb_512_e16_ra_all_s123_test_tta4.csv` |
| 4 | **3-seed 概率平均**（等权 1:1:1） | `oof/seed7_s42_s123_test_tta4.csv` |
| 5 | **3-seed soup**（权重平均 1:1:1）→ TTA4 | `runs/cnxb_512_e16_ra_all_soup3/last.pt`、`oof/soup3_test_tta4.csv` |
| 6 | 与 ens10 融合 `w=0.10/0.15/0.20/0.25`（logit + rank） | `oof/*s123*_ens10_logit_w*.csv`、`oof/soup3_test_tta4_ens10_*.csv` |
| 7 | **强制校验**（见下）+ 任一候选不合规即 `exit 1` | 控制台表格 |

## 三、必须回填的校验（#152 正文要求，脚本自动打印）
1. **基础**：5855 行 / 官方顺序（逐行比对 `glaucoma_first/test.csv`）/ 无 NaN / `[0,1]` / mean / pos@0.5；
2. **成员一致性**：`s123`、`3seed_avg`、`soup3` 分别 vs seed7、seed42 的 **Spearman + top200**；
3. **对照 98.84**：每个候选 vs `seed7_s42_ens10_logit_w20.csv`（ID 21547）与 `cnxb_512_e16_ra_all_ens10_logit_w10.csv`（ID 20380）的 **Spearman + top200**；
4. **打包**：`package_submission.py --csv <最优> --name submission_20261027_ra_all_s123_wNN --out-root <盘上目录>`，记录 **md5**。

## 四、已知口径与坑
- **train+all 无干净 valid**：不报任何 valid AUROC/AUPRC，不得据 valid 调参；
- **权重平均会抬校准**：10-26 soup（seed7+seed42）mean 从 ~0.05 抬到 **0.0815**（接近 ens10 0.083）。AUROC 只看排序不受影响，但 0.5 阈值出 `Pred_Label` 必须重校准（候选的 Pred_Label 仅格式完整）；
- **`fusion_ens10.py` 权重**：本单已加 `--weights`（默认仍 0.1/0.2/0.3，向后兼容），#152 用 `--weights 0.10 0.15 0.20 0.25`；
- **`seed_average.py`** 需给 `--members` 按名字对齐，官方顺序由 `--test-csv` 保证；
- 云上非交互 PATH 无 `python`/`python3`，必须显式 `PYTHON=/root/miniconda3/bin/python`；
- 磁盘：`/root/autodl-tmp` 常被历史 run 占满 → 工作目录放 overlay（如 `/root/gf152`），权重 705MB×4 需 ≥4GB 余量。

## 五、风险与判负线
- 同配方换 seed 的多样性有限：**s123 很可能仍约 98.84**（#51 RA 三 seed 排名相关 0.9118–0.9404；seed7 vs seed42 实测 0.8868）；
- 若所有候选 vs 98.84 的 rho ≥0.995 且 top200 = 200/200，说明「几乎没动」，提分概率低，应**如实报告并建议不再投配额**；
- 训练/推理失败一律如实回填，**不交付半成品**。

## 六、交付
- `runs/cnxb_512_e16_ra_all_s123/{last,best}.pt` → 上传 Release（大文件不入库）
- `oof/cnxb_512_e16_ra_all_s123_test_tta4.csv`、`oof/seed7_s42_s123_test_tta4.csv`、`oof/soup3_test_tta4.csv`
- 融合候选 CSV（w=0.10/0.15/0.20/0.25）
- `docs/ISSUE_152_RESULTS.md`：训练日志/耗时、全部校验数字、**推荐哪一个候选 + 理由**
- **不要自动提交**

## 七、实战补充（2026-10-27 真机踩坑，务必先看）
1. **CSV 路径不是同一个目录**：`seed7` 的 test TTA4 在**仓库根**（`cnxb_512_e16_ra_all_test_tta4.csv`），而 `seed42` / `s123` 的产物在 **`oof/`** 下。本 PR 已修正 `pipeline_152.sh` 的 `--members` 路径（原先写成根目录 → `FileNotFoundError` → 3-seed 平均整段失败）。
2. **看门狗要有崩溃循环保护**：本次云端 watchdog 以「无进程且无 `pipeline DONE`」为续跑条件，遇到上面那个确定性失败 → **每 2 分钟重启一次、连续 4.5 小时空转**（每次重跑一遍 TTA4）。正确做法：记录重启次数（如 ≤3 次）或检测「上次退出码/最近 10 分钟无新增日志」就停止并写 `NEED_ATTENTION` 标记。
3. **训练是安全的**：训练完成 (`done. train-on-all 16 epochs`) 后 `last.pt` 已在盘，任何推理段失败都不会影响权重；重跑只需 `--ckpt` 指向它。
4. **权重上传走云端**：705MB 从云端 `uploads.github.com` 上传约 **60 s（11.7 MB/s）**，比本地代理快两个数量级；下载仍走 `ghfast.top` + `aria2c`。
