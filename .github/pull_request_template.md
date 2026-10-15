## 这个 PR 做了什么

<!-- 一句话说明；关联 issue 用 #编号（例：完成 #9 任务A） -->

## 类型

- [ ] 实验（训练结果 / 指标报告）
- [ ] 代码（脚本 / 工具 / 流水线）
- [ ] 文档
- [ ] 修复

## 交付物清单（逐项确认）

- [ ] 代码 / 文档已提交，**权重等大文件未入库**（`*.pt` 已被 .gitignore 排除，请放 GitHub Release 附件或群文件）
- [ ] 若有训练：`runs/<exp>/best.pt` 与 `runs/<exp>/probs_valid_full_tta4.npy` 已生成（后者用 `python evaluate.py --ckpt <best.pt> --tta 4`）
- [ ] 若改动了提交相关脚本：已跑通 `make_submission.py` 的格式校验（rows / names / prob range / label logic 全 OK）
- [ ] 若新增了训练模型：已在本 PR 说明它的 **cache-root 与 img_size**（集成时必须匹配）

## 实验结果（若有训练，必填）

| exp | arch | img_size | cache | AUROC (TTA4) | AUPRC (TTA4) | balmax | epoch 用时 | 机器 |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

> 参考基线（队长机器）：ConvNeXt-Small@384 = AUROC 0.98994 / AUPRC 0.86371；
> ConvNeXt-Small@512 on cache768 = 0.98994 / **0.87256**；当前集成 = 0.99110 / 0.87733

## 复现命令

```bash
# 完整、可直接复制执行的命令行（含 --cache-root / --init-from 等全部参数）
```

## 验证方式

<!-- 你做了哪些验证（跑通 / 对拍 / 小样本测试），reviewer 需要重点检查什么 -->

## 风险与注意

<!-- 是否影响既有流程？是否需要重新生成提交文件？是否与其他人正在跑的任务冲突？ -->