# Swin_pair 数据说明（98.87 候选实际使用的 Swin 组件）

- `swin_pair_test_tta4.csv`：5855 行，官方顺序；MD5 `0825072672651298f36e3e7991ab60ad`
- `swin_pair_valid_tta4.csv`：5849 行；MD5 `945d080da18f666544532dd30b860480`
- 构成：**Swin stage1（swinb_384_e16）与 stage1 seed7（swinb_384_e16_s7）各 0.5 概率平均**；两者均 train-only，无测试伪标签。
- valid TTA4 AUROC：**0.99240292**。
- 用途：98.87 候选（ID 22205）的 `Swin_pair` 成员；**不是** #139 的 Swin3 三折 OOF（0.9839）。
- #158 用 Swin3 近似 Swin_pair 的负结果不适用于本文件；请用本文件重跑。
