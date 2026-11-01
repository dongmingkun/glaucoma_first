Issue #195 结果 — Large384 RA all seed2024（Kaggle T4）+ 🚩 红旗报告
状态：训练 16/16 ep 完成、test CSV 已产出；但 CSV 有异常信号（红旗），三人汇总评估时请注意。 环境：Kaggle T4×2（自有账号 jiahaochen7711），Dataset jiahaochen7711/gloucoma（cache_all，私有）。 3 个会话（14:36→23:59 / 00:13→08:40 / 续跑至 16ep + 推理），每周配额消耗约 18h。

训练（convnext_large @384 + RA + train+all seed2024）
会话 v2：clone + smoke（batch8 通过）+ 正式训练，timeout 30600s 正常收尾；
会话 v3：--resume-from resume.pt 续跑（从 ep9 恢复，跳过 192 batch 正确），ep9→ep16 完成；
16/16 epochs 完成：log.csv ep9-16（每 epoch ~3760s T4），ep16 train_loss 0.0698、lr 衰减到 5e-07；
last.pt（1,576,423,375 B）/ best.pt / resume.pt 均在 kernel output。
交付 CSV
oof/cnxl_384_e16_ra_all_s2024_test_tta4.csv（infer.py，TTA4，use=ema，5855 行/官方顺序/无 NaN）
SHA256: b96fe9b2eeaebbe7570a8f36d96aabd79d2027bf9990f4d9b090e7ee798542d8
mean 0.024791 / pos(0.5) = 121 / 范围 [0.0000, 0.9594]
🚩 红旗（汇总评估前必读）
s2024 的 test 输出与其他 Large 成员严重偏离：

对照	Spearman
Large384_s2024 vs 98.87	0.5877
Large384_s2024 vs Large384_s123（同配方不同 seed）	0.5909（同配方应 0.9+）
Large384_s2024 vs Large512_s123	0.6805
mean 0.0248（s123 约 0.05，偏低一半）、max 仅 0.9594、pos(0.5) 121 张；
训练曲线本身正常（loss 0.124→0.070 正常下降、lr 正常衰减），嫌疑集中在 EMA-resume： v3 从 resume.pt 续跑时 EMA 状态恢复是否完整未经独立验证；也可能模型真的欠收敛；
建议：队长用 check_train_on_all.py --ckpt last.pt --ref <Large384_s123 last.pt> --tta 4 独立核验 s2024 质量；在核验通过前，不要把 s2024 计入 3-seed 平均提交。
候选（按工单：3-seed 概率平均后 0.05/0.10 掺入 98.87）
候选	文件	md5	vs 98.87
w=0.05	oof/cand195_seed3avg_w05.csv	0944ac864525608cacc08ea3f1f39445	rho 0.9994 / top200 198
w=0.10	oof/cand195_seed3avg_w10.csv	e9e6252f0c1be8a9e77c82d9c4619126	rho 0.9974 / top200 194
3-seed 平均 = 概率空间等权（L384s123 + L512s123 + L384s2024）/3，掺入后与 98.87 的 rho 被 s2024 异常拉低（3seedavg vs 98.87 仅 0.8424）；
红旗未澄清前，两个候选都不建议提交。
合规
train-on-all（57245 = train+valid），不含 test、无伪标签；v3 pseudo 未使用；TTA 允许；
未自动提交；每周 Kaggle 配额消耗约 18h/30h。
复现
训练/续跑/推理 kernel：kaggle.com/code/jiahaochen7711/cnxl384-ra-all-s2024（version 1-5 可溯）；
训练 log（ep9-16）与 CSV 已在 kernel output；本报告数据可由 kaggle kernels output 复算。
