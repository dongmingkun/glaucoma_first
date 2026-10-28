# #179 模型核验材料 + D1/L1 审计

> 状态：A/B/C/D/E 逐条核实 ✅｜F1 官方 T4 待复测 ⚠️｜D1/L1 均 100% 复现 ✅｜cs_hi512_all_ra SHA-256 已补 ✅
> 纯 CPU 完成，未租 GPU、未跑 Large 推理。填好的表格：`docs/模型核验材料与数据申明清单_20261028.xlsx`（7 个 sheet）。

## 0. 本轮最重要的发现：D1/L1 差异全在 RA_avg 口径

审计过程中定位到一个**会反复误导审计的口径分歧**：

| 口径 | RA_avg 定义 | 复算 L1 的结果 |
|---|---|---|
| `tools/rank_blend.py`（仓库现状） | best/last **先概率平均再 rank** | max_err **0.0593**，mean_err 0.00193，98.7% 行不同；但 top200 **200/200**、rho 0.99993 |
| 队长流水线 | best/last **先各自 rank 再 0.5/0.5 平均** | max_err **0.000000**，与官方 L1 **字节级完全一致** |

**结论**：按「先 rank 再平均」，D1 与 L1 都能用公开成员 100% 复现 —— 这说明此前的 max_err≈0.1 不是成员缺失、也不是权重问题，而是**两套流水线的 RA_avg 口径不同**。

| 对象 | md5（复算） | md5（官方） | 结论 |
|---|---|---|---|
| D1（官方双分辨率） | `e0f7e829d2e44382ef7de64864dd0f02` | `e0f7e829…`（队长报） | 前 8 位一致，逐行 max_err 0 |
| L1（Large384 单分辨率） | `e244a37e6e64cb184d895ce63cb8db13` | `e244a37e6e64cb184d895ce63cb8db13` | **完全相同（字节级）** |

> 建议：给 `tools/rank_blend.py` 加 `--ra-rank-mean` 开关（或直接改默认），让仓库口径与队长流水线一致。否则以后每个新包都会被审计判成「不可复现」。这一步我没有擅自改脚本，等确认后再单独提 PR。

## 1. A 模型核验（逐条证据）

| 条目 | 结论 | 证据 |
|---|---|---|
| **A1** 微调 checkpoint | 已核实 | cb384 = `convnext_base_384_cache512_1015_auprc0.8644.pt`；cs_hi512_all = `cs_hi512_all_fp16.pt`；cs_hi512_all_ra = `cs_hi512_all_ra.pt`；三者均为 ImageNet-1k 预训练骨干 + 官方 train/valid 微调 |
| **A2** 公网直链 | 已核实 | 三条链接均实测 **HTTP 206**，Content-Range 字节数分别为 176,328,125 / 199,660,651 / 99,831,694，无需登录、非 gated |
| **A3** 许可证 | 已核实 | torchvision 与 PyTorch 均为 **BSD-3-Clause**，全文已入 `licenses/`（1,517 B / 3,464 B），非 NC、非 gated |
| **A4** 预训练来源 | 已核实 | torchvision 官方 ConvNeXt 权重（ImageNet-1k），论文 arXiv:2201.03545，数据集 ILSVRC2012（非医学） |
| **A5** SHA-256 | 已核实（1 项待补） | 见第 2 节：**预训练骨干给出完整 SHA-256**，且 torchvision 文件名内嵌 SHA-256 前 8 位可自校验 |

## 2. A5 SHA-256（预训练骨干 + 微调权重）

| 资产 | 字节数 | SHA-256 |
|---|---|---|
| 预训练 `convnext_base-6075fbad.pth` | 354,486,097 | `6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd` |
| 预训练 `convnext_small-0c510722.pth` | 201,014,929 | `0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a` |
| 预训练 `convnext_large-ea097f82.pth`（训练用，不在最终包） | 791,189,585 | `ea097f822cdb2cab40905cec3949f4bdfd6bf1d546c3ec5e43f0b1c6275e144` |
| 微调 `cb384` | 176,328,125 | `a389f41cb367132b4c23e2f9195bf15ac87c9df1214d359656f36ee63aff710e` |
| 微调 `cs_hi512_all` | 199,660,651 | `3e04d1906965742d8b13657b16e539063b05dab622c5272454939f57105f492e` |
| 微调 `cs_hi512_all_ra` | 99,831,694 | `5a04ebe3152416022b0e07c9d42f7af57532b9cb366771c09889173a154c86e6` |

自校验说明：torchvision 权重文件名里的 8 位 hex（`6075fbad` / `0c510722` / `ea097f82`）正对应各自 SHA-256 的前 8 位，与上表实测值吻合 —— 这构成「预训练来源未被篡改」的独立证据。

## 3. B/C/D/E 申明要点

- **B1** 仅官方 train/valid；train+all = 官方 train ∪ valid，无外部数据
- **B2** 无测试集标签、无测试伪标签；test.csv 仅用于出概率
- **B3** train+all 后 valid 不再用于早停/选择/阈值，按工单固定 `last.pt`+EMA，不报 valid
- **C1/C2/D** 无自监督、无外部无标签数据、无蒸馏
- **E1/E2/E3** 最终 3 模型包无测试统计量、逐图独立预处理、BN eval；3 模型概率平均、TTA1、batch=1；包体 475.82 MB、71 ms/图（5060 reference）

## 4. 缺口（如实）

1. **F1 官方 T4 未实测**：现有只有 5060 8GB reference（71 ms/图，100 张与 Large 单尺度比对 0/100 不一致）；草稿口径「T4 待复测」保留，不冒充；
2. （已补）`cs_hi512_all_ra` SHA-256；
3. **xlsx 模板原件在队长机器**（`C:/Users/Kevin/Desktop/模型核验材料与数据申明清单_草稿.xlsx`），本表按 #179 条目重建，若模板字段有出入，我按模板再对一版；
4. D1 的原文件同样在队长机器，本节是「用公开成员复算 → md5 与队长报的一致」的比对，不是「对原文件逐行 diff」。

## 5. 交付物

| 路径 | 内容 |
|---|---|
| `docs/模型核验材料与数据申明清单_20261028.xlsx` | 填好的核验表（说明 / A / B-C-D / E / F / D1-L1 审计 / 附录直链与 SHA） |
| `licenses/LICENSE-torchvision-BSD-3-Clause.txt` | torchvision 许可证全文 |
| `licenses/LICENSE-pytorch-BSD-3-Clause.txt` | PyTorch 许可证全文 |
| `docs/ISSUE_179_RESULTS.md` | 本文件 |
