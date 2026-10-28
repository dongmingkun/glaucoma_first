# Issue #173 结果记录 — Large384 提交包审计 + 合规包 cost.py

#173 完成（A 审计 ✅ / B cost.py ✅）

═══ A. Large384 提交包审计 ═══

1. 前提确认：e3 提交包（md5 e3ddd565）内容与 oof/blend161_v4l_w35302015.csv **完全一致**（max_err=0.0，同一文件改名）——审计目标即复现 blend161。

2. 基线复算（台账配方 rank(0.35 ens10 + 0.30 RA_avg + 0.20 Swin_pair + 0.15 Large384)，成员全公开版）：**max_err = 0.1024，rho = 0.99747**——独立复现工单说的 0.103 偏差。

3. 4 维变体全排除（12 组合，max_err 0.1024~0.1031，无一为零）：
   - RA_avg 口径：rank 平均 / 概率平均 → 差 0.0007，非原因
   - Swin_pair 口径：#160 官方 / stage1+s7 概率平均 → 差 0.00003，非原因
   - rank 归一化：n+1/n-1/n/raw → 差 0.0003，非原因
   - 权重（和=1，融合内部归一等价）

【A 结论】e3 配方框架正确（rho 0.9975），但**至少一个成员用了未公开版本的概率文件**——最可能是 Large384（公开版 cnxl_384_..._s123_test_tta4.csv 与 blend161 构建时版本不同批次）。**L1（md5 e244a37e）不在任何 Release（15 个已核对）**，无法独立验证。**提交建议：L1 更安全（可复现/可审计）**；请上传 L1 或 Yulingling 补传构建时实际用的 Large384 概率文件，我立即完成比对。

═══ B. 合规包 cost.py ✅ ═══

环境重建：predict.py（submit_pkg）+ 3 ckpt 从 Release 下载（cb384 176.3MB / cs_hi512_all 199.7MB / cs_hi512_all_ra 99.8MB = 475.8MB，与队长数字一致）；100 张测试图目录 + test100.csv（含 Age）已备。

**F2 时延（100 张 / batch=1 / TTA1 / 3 模型）**：
- 3 轮：8.72s / 6.22s / 6.49s，**平均 7.14s**
- 均为 5060 reference（官方 T4 未实测，写 T4 待复测）

**输出一致性（cost 3 轮均值 vs predict.py 基准 baseline_submission100.csv）**：
- 二值（0.5 阈值）不一致数：**0 / 100** ✅
- 最大 |Δp|：**0.00005818**（平均 1.13e-6；fp16/非确定性算子的正常浮动）
- cost.py 位置：`compliant_pkg/cost.py`（import predict.py 同一 load_model/run_group；eval；固定种子；无随机增强）

**F3 大小**：cb384.pt 168.16 + cs_hi512_all.pt 190.41 + cs_hi512_all_ra.pt 95.21 + predict.py 0.01 = **453.79 MiB（=475.82 MB 十进制，与队长 475.82MB 一致）**。

【合规】三模型均 train-only/官方发布版；无 v3/pseudo；未自动提交。xlsx 回填数据已齐（F1 写 5060 reference/T4 待复测；F2/F3 见上），不要求现在上传。

【遗留】A 的成员版本澄清需队长上传 L1 或 Yulingling 补 Large384 构建版概率；B 的 5855 全量比对需 compliant_pkg 官方 submission.csv 就位（当前 100 张子集核验通过）。
