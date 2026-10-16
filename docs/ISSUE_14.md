# Issue #14: [P0][工程] 权重共享通道 + 复赛包体积纪律（≤500MB，fp16 瘦身）

> 状态：closed ｜ 指派人：JiahaoChen-star-JJ
> 本文件是 GitHub issue #14 的离线快照（正文 + 全部评论），供队友的 AI 工具/离线环境直接读取。
> 权威内容以 GitHub issue 为准。

## 正文

## 背景（两个刚需）

**① 跨机器集成缺权重通道**：目前所有 best.pt 只在队长 D 盘，队友拿不到，无法跨机器集成与复核。
| 模型 | best.pt（含 EMA 双份） | 单份权重 |
|---|---|---|
| ConvNeXt-Tiny | 226MB | ~113MB |
| ConvNeXt-Small | 399MB | ~200MB |
| ConvNeXt-Base | 705MB | ~356MB |

**② ⚠️ 复赛包体积红线**：官方限 **≤500MB（压缩前）**。当前最优集成（Small@512 + Base + B3×3）fp32 单份合计 **684MB → 超标**；转 fp16 后 **342MB ✓ 合规**。必须在打包前解决，否则复赛成绩作废。

## 任务 A（必做）建立权重共享通道
1. 创建 GitHub Release（tag 如 `weights-1016`），把各模型**去掉 EMA 的单份权重**作为附件上传
   - 命名规范：`<arch>_<img>_<cache>_<MMDD>_auprc<0.xxxx>.pt`
     例：`convnext_small_512_cache768_1016_auprc0.8726.pt`
2. 新建 `docs/WEIGHTS.md`：每个权重一行 —— 下载链接 / arch / img_size / **cache-root** / TTA4 指标 / 用途备注
3. 上传方法（网页打不开用 API，令牌与代理已配好；`release_id` 队长提供）：
   ```bash
   curl -x http://127.0.0.1:7897 -X POST \
     -H "Authorization: Bearer $(cat /d/glaucoma_work/.gh_token)" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @权重.pt \
     "https://uploads.github.com/repos/dongmingkun/glaucoma_first/releases/<release_id>/assets?name=<文件名>"
   ```

## 任务 B（必做）复赛包体积与格式纪律（与 #5 联动）
1. 写 `tools/slim_weights.py`：读 best.pt → 只留推理那一份 state_dict → 转 fp16 保存 → 打印"转换前后大小 + 加载自检（能否被 `FundusNet` 正常加载并跑通一次前向）"
2. 与 #5 联动：`predict.py` 支持 fp16 权重加载
3. 产出**体积预算表**（成员 × 单份 fp16 大小 = 总计 ≤500MB），回填到 #5

## 交付物
- [ ] `weights-1016` Release（至少含 Small@512 与 Base）+ `docs/WEIGHTS.md`
- [ ] `tools/slim_weights.py` + 体积预算表（回填 #5）
- [ ] `docs/TEAM_HANDOFF.md` 补一段"如何取用队友权重"（下载 + 放到 runs/ 目录约定）

## 评论

### @dongmingkun (2026-10-16 02:04)

## 队长侧完成记录（2026-10-16）

### ✅ 权重共享通道已建好
- Release **[weights-1016](https://github.com/dongmingkun/glaucoma_first/releases/tag/weights-1016)** 已上传 **8 个 fp16 瘦身权重**（合计 679MB）：
  `convnext_small_512_cache768`（AUPRC 0.8752）、`convnext_base_512_cache768`（0.8765，当前最强单模）、
  `convnext_base_384_cache512`（0.8644）、`convnext_small_384_cache512`（0.8637）、`convnext_tiny_384_cache512`（0.8558）、
  `effnet_b3_384/448/512_cache512`
- `docs/WEIGHTS.md` 已回填全部下载链接 + 实测体积预算表 + 热启动用法
- **#11 / #13 现在可以直接下载热启动**（⚠️ 瘦身版只有推理权重，不能续训；要续训找队长要原始 best.pt）

### ✅ fp16 ↔ fp32 等价性验证（队长实测）
`submit_pkg/predict.py`，同一模型（hi512_v2）同 20 张测试图，TTA1：
| 指标 | 结果 |
|---|---|
| 平均绝对差 | **0.000244** |
| 最大绝对差 | 0.001768（尾部极低概率图 0.0176 → 0.0194） |
| 相关系数 | 0.997174 |

**结论**：fp16 瘦身对排序无实质影响，可用于复赛包。注意偏差量级（~2e-4 均值）略大于"跨环境一致性 ±1e-4"的口径 —— 如果要求绝对严格，复赛包可以在加载时把 fp16 权重转回 fp32（`model.float()`），精度与原始完全一致、只多占内存不占磁盘。

### 剩余（#20 联动）
- [ ] 复赛包延迟优化（#20，已派给 @JiahaoChen-star-JJ / @Yulingling-sc）
- [ ] T4 实测（#5 剩余项）

### @dongmingkun (2026-10-16 04:13)

## 关闭说明（交付物已达成，2026-10-16）

- ✅ **权重共享通道**：Release **[weights-1016](https://github.com/dongmingkun/glaucoma_first/releases/tag/weights-1016)**（8 个 fp16 瘦身权重，合计 679MB）+ `docs/WEIGHTS.md`（下载链接 / cache-root / 指标 / 体积预算表 / 热启动用法）
- ✅ **瘦身工具与等价性验证**：`tools/slim_weights.py`（PR #19 + 输出目录自动创建修复）；fp16↔fp32 实测（20 张图，均差 2.4e-4 / 最大 1.8e-3 / 相关 0.9972，对排序无实质影响）
- ✅ **体积纪律**：当前最优集成 fp32 684MB → fp16 342MB ≤ 500MB 红线

联动项：复赛包延迟优化（#20）已交付并关闭；**官方镜像 T4 实测在 #5 中继续跟踪**（#5 保持 open）。