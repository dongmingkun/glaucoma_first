Issue #162 结果记录 — RA/Swin checkpoint soup + V5 候选
状态：部分完成（A 全套 ✅、V5a ✅；B 与 V5b/V5c 因 s2024/s7 ckpt 不可得如实跳过，未用替代品伪造）。 环境：本机 RTX 5060 Laptop 8GB（均衡电源）；RA ckpt 从 Release cnxb_512_e16_ra-1024 下载 （best/last 各 705225551 字节，与队长预期大小一致）。

A. RA soup ✅
python tools/seed_soup.py --ckpts <RA_best.pt> <RA_last.pt> --weights 0.5 0.5 \
  --out runs/cnxb_512_e16_ra_soup/last.pt
python tools/oof_predict.py --ckpt runs/cnxb_512_e16_ra_soup/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --use ema --out oof/cnxb_512_e16_ra_soup_test_tta4.csv
产物：oof/cnxb_512_e16_ra_soup_test_tta4.csv（5855 行 / 官方顺序 / 无 NaN）
cache_all768：本机用 cache768 硬链接构建（train/valid/test 各 63100，os.link，零额外磁盘）
B. Swin soup ❌ 未执行
工单指定的 swin_base_384_1021_124939_swinb_384_e16_s2024/last.pt 与 ..._s7/last.pt 不在本机、不在上游任何 Release（已核对全部 13 个 Release 资产清单）。weights-1016 只有 s7/stage1 的概率 CSV（已拉取），无 ckpt。按工单"缺 ckpt 就跳过，不要用 v3 pseudo"——跳过。

C. V5 候选（rank 融合，对照 V4 = 98.87 / ID 22205）
候选	配方	状态
V5a	0.40 ens10 + 0.35 RA_soup + 0.25 Swin_pair	✅ 已产出
V5b	0.40 ens10 + 0.35 RA_avg + 0.25 Swin_soup	❌ 跳过（Swin_soup 依赖 B）
V5c	0.40 ens10 + 0.35 RA_soup + 0.25 Swin_soup	❌ 跳过（同上）
Swin_pair 口径：使用官方 #160 文件 swin_pair_test_tta4.csv（md5 08250726，valid AUROC 0.99240292）， 与 V4/98.87 完全同口径；非近似、非 Swin3。

V5a 明细
文件：oof/V5_V5a.csv（5855 行 / 官方顺序 / 无 NaN / 范围 [0,1]）
md5：1eb4b12e586a825d1511317ca5aa4bb7（Swin_pair 已改用官方 #160 文件重算；旧近似版 9688e8be 作废）
对照 V4（rank 融合复现）：rho = 0.9872，top200 = 197/200（<0.995，非打平组合）
成员相关性：RA_soup 0.9083 / Swin_pair 0.9043 / ens10 0.9394 —— 三成员分布均衡，无单一主导
打包：submission_20261027_V5a_rasoup_swinpair_cal.zip（md5 9854c2084325eb734557e011ceba8af3；zip 内 submission.csv md5 = 1eb4b12e...，与外层一致，已验证）
合规
RA_best/last（train-only）、stage1/s7（train-only）均无测试伪标签；v3 pseudo 全程未使用；
V5a 为 test-only 探索候选，未自动提交，是否线上验证由队长决定。
复现
# A
python tools/seed_soup.py --ckpts <RA_best> <RA_last> --weights 0.5 0.5 --out runs/cnxb_512_e16_ra_soup/last.pt
python tools/oof_predict.py --ckpt runs/cnxb_512_e16_ra_soup/last.pt --csv glaucoma_first/test.csv \
  --cache-root cache_all768 --split-dir test --img-size 512 --tta 4 --use ema \
  --out oof/cnxb_512_e16_ra_soup_test_tta4.csv
# C
python tools/v5_fuse.py
执行与事故记录
ckpt 获取：Release cnxb_512_e16_ra-1024（asset id 672298145 / 672300665），API octet-stream 下载， 1.3GB 约 25 分钟（~1MB/s 单流）；
local_run139.sh 同源的 POSIX 路径坑再现（CK=/d/... → Windows python FileNotFoundError）， 已修为 D:/ 风格；
B 缺 ckpt 的替代方案（用 Swin3/概率平均替代 soup）均未采用，遵守工单禁止条款；
耗时：ckpt 下载 25min + soup/推理 12min + V5 组装 1min ≈ 40min（不含等待）。
