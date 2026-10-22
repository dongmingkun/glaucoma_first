# 文档索引（docs/）

> 本目录归档各工单的训练/评估结果、权重台账与交接文档。
> 工单权威内容以 GitHub Issues 评论为准；本索引仅做导航。

## 一、工单结果归档（ISSUE_*.md）

### 已达标 / 进集成
| 文档 | 工单 | 一句话结论 |
|---|---|---|
| [ISSUE_76.md](ISSUE_76.md) | #76 | Swin-B@384 16ep 阶段1 达标（valid TTA4 0.99233）；阶段2 train-on-all 验收 PASS（Spearman 0.9066） |
| [ISSUE_90.md](ISSUE_90.md) | #90 | Swin-B@384 16ep × seed7 复跑（valid TTA4 0.99189 ≥ 0.990，PASS）— 经 PR #99 合并后生效 |

### 负结果 / 不进集成（多样性 & 目标函数探索）
| 文档 | 工单 | 一句话结论 |
|---|---|---|
| [ISSUE_77.md](ISSUE_77.md) / [ISSUE_77_RESULTS.md](ISSUE_77_RESULTS.md) | #77 | ViT-B@384 8ep：valid TTA4 0.96918（<0.990），集成 w=0；16ep 按早停约定未跑 |
| [ISSUE_78.md](ISSUE_78.md) | #78 | Base@384 × Focal(γ=2.0)：valid 0.99054，ΔAUROC +0.00001（P=0.596）无增益 |
| [ISSUE_80.md](ISSUE_80.md) | #80 | Base@384 × RandAugment：valid 0.98953（<0.990），ΔAUROC +0.00001（P=0.602）无增益 |
| [ISSUE_82.md](ISSUE_82.md) | #82 | Base@384 × Mixup(0.2)：valid 0.98920（<0.990），w=0 零增益 |
| [ISSUE_79.md](ISSUE_79.md) | #79 | Base@448：valid 0.98786（<0.990）负结果 |
| [ISSUE_83.md](ISSUE_83.md) | #83 | Base@384 × ImageNet-22k：valid 0.98926（<0.990）负结果 |
| [ISSUE_84.md](ISSUE_84.md) | #84 | Base@384 × pos_weight=3：valid 0.98919（<0.990）负结果 |
| [ISSUE_71.md](ISSUE_71.md) | #71 | ConvNeXt-Large@384 阶段1 负结果 |
| [ISSUE_74.md](ISSUE_74.md) | #74 | ConvNeXt-Large@384 v2 温和优化负结果 |
| [ISSUE_95.md](ISSUE_95.md) | #95 | Swin-L@384 16ep 容量上探：valid TTA4 0.98732（<0.990）负结果 |
| [ISSUE_92.md](ISSUE_92.md) | #92 | Swin-B seed123：valid TTA4 0.98599（<0.990）；valid CSV 待补，3-Swin 增益待算 |
| [ISSUE_93.md](ISSUE_93.md) | #93 | Swin-B lr7e-5：valid TTA4 0.98535（<0.990）；概率已交付并复核 |

### 早期探索 / 其他
| 文档 | 工单 | 备注 |
|---|---|---|
| [ISSUE_1.md](ISSUE_1.md) | #1 | 阈值口径 A/B 验证 |
| [ISSUE_2.md](ISSUE_2.md) | #2 | — |
| [ISSUE_3.md](ISSUE_3.md) | #3 | — |
| [ISSUE_5.md](ISSUE_5.md) | #5 | — |
| [ISSUE_8.md](ISSUE_8.md) | #8 | Small@512 × RA（家族 check 最高成员） |
| [ISSUE_11.md](ISSUE_11.md) | #11 | — |
| [ISSUE_13.md](ISSUE_13.md) | #13 | — |
| [ISSUE_14.md](ISSUE_14.md) | #14 | — |
| [ISSUE_57.md](ISSUE_57.md) | #57 | — |
| [ISSUE_58.md](ISSUE_58.md) | #58 | — |
| [ISSUE_61.md](ISSUE_61.md) | #61 | — |
| [ISSUE_62.md](ISSUE_62.md) | #62 | — |
| [ISSUE_65.md](ISSUE_65.md) | #65 | — |
| [ISSUE_69.md](ISSUE_69.md) | #69 | — |
| [ISSUE_81.md](ISSUE_81.md) | #81 | 复赛包本地 PASS（5855 行 / 一致性 / 452.48MB）；5060 p50/p95 37.5–45.5 / 77.2–80.7ms；T4 待验证 |

## 二、流程与规范文档
| 文档 | 说明 |
|---|---|
| [WEIGHTS.md](WEIGHTS.md) | 权重台账（Release 资产索引 + 成员 AUROC/AUPRC） |
| [TRAIN_ON_ALL.md](TRAIN_ON_ALL.md) | train-on-all 协议（验收口径 Spearman ≥0.90） |
| [HOW_TO_PR.md](HOW_TO_PR.md) | PR 提交流程规范 |
| [TEAM_HANDOFF.md](TEAM_HANDOFF.md) | 团队交接文档 |

## 三、维护说明
- 新增工单归档：在 `docs/` 下新建 `ISSUE_<n>.md`，按「背景 / 配置 / 结果 / 结论 / 交付物」五段式；并在本索引对应表格登记一行。
- 训练产物（*.pt / *_probs*.csv）**不入库**，统一通过 GitHub Release `weights-1016` 共享；仅在归档文档的「交付物」节登记文件名与位置。
- 本索引由 #98 补齐早期缺口时建立；此后续档请同步维护。
