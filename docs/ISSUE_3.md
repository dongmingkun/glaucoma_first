# Issue #3: [P1][训练] 全量数据重训（train+valid，高分辨率配方）

> 状态：closed ｜ 指派人：Yulingling-sc
> 本文件是 GitHub issue #3 的离线快照（正文 + 全部评论），供队友的 AI 工具/离线环境直接读取。
> 权威内容以 GitHub issue 为准。

## 正文

## 背景（为什么要做）

valid 5849 张占全体 10.2%，train+valid 全量重训（`--train-on-all`）通常 +0.002~0.005 AUROC。代价：没有验证集，epoch 数只能相信已验证 run 的饱和点。

## 任务

1. 选定配置（等 Issue #2 / #4 结论出来再定：B3@384 / B3@448 / B4）。
2. epochs 取"还在涨但趋缓"的位置。参考 B3@384 曲线（EMA AUROC 0.9714→0.9849，共 10ep 仍在涨），建议 8~10 ep。
3. 运行（示例）：
   ```bash
   python train.py --arch efficientnet_b3 --img-size 448 --epochs 8 \
     --batch-size 12 --accum 3 --workers 3 \
     --train-on-all --tag b3_448_final --init-from <可选热启动权重>
   ```
4. 备份 final 权重（命名规范见 Issue #6），出一版提交与验证版模型做 LB A/B（只花 1 次提交配额）。

## 交付物

- [ ] 全量重训 final 权重（记录路径 + 大小）
- [ ] 一次 LB A/B 结果（vs 验证版模型）
- [ ] 结论：最终提交模型采用哪个

## 依赖

- 等 Issue #2（448+集成）与 Issue #1（评分口径）结论后再启动，避免白跑一遍。

## 评论

### @dongmingkun (2026-10-14 15:12)

**本工单负责人：@Yulingling-sc**（训练方向）。接受 GitHub 仓库协作邀请后即可查看/操作本仓库；如需调整分工，请在本工单回复。

### @dongmingkun (2026-10-15 14:49)

⏰ **今晚 24:00 前收尾提醒**

请给手上工单一个明确的收尾：**完成 → 贴结题评论并关闭；未完成 → 留言说明「做到哪一步 / 卡在哪 / 预计何时完成」**（附报错日志）。今晚 24:00 的状态决定明天的分工。

明天起所有改动走 **PR 合并请求**（`docs/HOW_TO_PR.md`：命令行 + 网页双流程，含首次认证配置）：
- 一次性配置：拉代码走代理或 SSH-over-443（`ssh.github.com` 的 443 端口可用）
- 实验类 PR 必填：指标表（含 **cache-root**、img_size）+ 复现命令 + 权重共享链接
- 权重（`*.pt`）绝不入库；由队长合并，不要自己点 Merge

新任务已派：见 **#11**（高分辨率模型矩阵）。

### @dongmingkun (2026-10-15 15:01)

## 配方更新（2026-10-15，方向已定）

原工单等 #2/#4 结论、按 B3/B4 配置——**现在方向明确，改为高分辨率配方**：

**推荐配置**（二选一，优先用已验证的强成员）：
```bash
# 方案 1：Small@512（768 缓存）全量重训 —— 当前最强单模配方
python train.py --arch convnext_small --img-size 512 --cache-root D:\glaucoma_work\cache768 \
  --epochs 6 --batch-size 8 --accum 6 --lr 5e-5 --head-lr 2.5e-4 --train-on-all \
  --init-from <cs_hi512/best.pt> --tag cs_hi512_all

# 方案 2：Base@384 全量重训（架构最强，成本更低）
python train.py --arch convnext_base --img-size 384 --epochs 5 --batch-size 8 --accum 6 \
  --lr 5e-5 --head-lr 2.5e-4 --train-on-all --init-from <cnxb/best.pt> --tag cb_all
```

**⚠️ 验证污染说明（重要）**：
`--train-on-all` 会把 valid 混进训练集，此后**该模型在 valid 上的指标不可信**（会偏高）。因此：
1. 该模型只能作为**集成成员或最终提交模型**，不要用它来选 epoch 或做模型对比；
2. 如果要保留一个"干净"的评估，建议 **train + 90% valid 训练、10% valid 持出**（约 585 张 / 19 阳性，只能看趋势不能看绝对值）；
3. 在 PR 里必须注明"该权重训练时见过 valid"，避免后续误用。

**交付**：final 权重（命名见 #13 规范）+ 一行训练配置说明 + 在台账登记是否已在线上验证。

### @dongmingkun (2026-10-16 04:38)

## 状态说明（2026-10-16，防重复执行）

- **全量重训尚未执行**（无 `--train-on-all` 的 run；配方见上一条评论）
- 现有高分辨率成员（hi512 / hi512_v2 / Base@512）均为**带验证集训练**的版本，与全量重训产物不可直接互比
- 建议：由持有训练机（D 盘环境）的一方执行，避免不同机器 / 缓存档位产生不可比的产物；执行前先确认当日提交配额安排（交付物含一次 LB A/B）

### @dongmingkun (2026-10-16 04:56)

## 关闭说明（工程侧已交付，实跑不再作为队员工单跟踪）

**已交付（PR #16，@JiahaoChen-star-JJ，已合并）**：
- `paths.py` 路径可移植（环境变量 → 仓库布局 → 旧路径三级回退）+ `train.py --csv-root/--cache-root`
- `--train-on-all` 闭环修复（不建 val DataLoader、写 best/last.pt、补 `use` / `train_on_all` 字段）
- 验收脚本 `check_train_on_all.py` + 手册 `docs/TRAIN_ON_ALL.md`

**未执行**：全量重训的实跑本身（无 `*_all*` run 目录；PROJECT_NOTES 亦记载"工程侧就绪…剩实跑"）。

**结论**：按"工程侧已交付"关闭本工单。实跑操作已完整固化在 `docs/TRAIN_ON_ALL.md`（前置条件 / epoch 换算 / 单向门风险 / 验收方式），改为**交付前 checklist**，不再占用队友工单——避免在不同机器上跑出不可比、且无法验证的产物。

> 如后续决定执行（例如最后交付前有 GPU 空档），请 reopen 或新开工单，并注明执行机器与缓存档位。