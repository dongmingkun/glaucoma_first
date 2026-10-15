# 如何提交合并请求（Pull Request）— 队员操作手册

> 目标：让你做的事（代码 / 实验 / 文档）安全地进入 `main`，队长审核后合并。
> 原则：**一个 PR = 一件事**；小步提交；**不要直接推 `main`**。
> 本文只讲 **Git 命令行（Git Bash）+ GitHub 网页**，不使用任何图形客户端。

## 0. 首次配置（只做一次）

### 0.1 安装 Git 并配置身份
```bash
git --version                                    # 确认已安装（Windows 装 Git for Windows 即可）
git config --global user.name  "你的名字"
git config --global user.email "你的GitHub邮箱"
```

### 0.2 推送认证（第一次 push 时会弹窗，按下面选一种）
- **方式一（推荐，最省事）**：保持默认，第一次 `git push` 会弹出浏览器窗口让你登录 GitHub 授权（Git for Windows 自带 Credential Manager），授权后凭据会被记住。
- **方式二（用 Token）**：GitHub → 右上头像 → Settings → Developer settings → Personal access tokens → Fine-grained tokens → 新建，权限勾 `Contents: Read and write` + `Pull requests: Read and write`。push 时用户名填 GitHub 用户名、**密码粘贴这个 token**。
- **方式三（SSH）**：`ssh-keygen -t ed25519 -C "你的邮箱"` 生成后把 `~/.ssh/id_ed25519.pub` 内容加到 GitHub → Settings → SSH keys；之后用 `git@github.com:dongmingkun/glaucoma_first.git` 这个地址克隆。

### 0.3 克隆仓库
```bash
cd /d/            # 或你放代码的盘
git clone https://github.com/dongmingkun/glaucoma_first.git
cd glaucoma_first
```
> 环境搭建（Python/依赖/数据/缓存）见 `docs/TEAM_HANDOFF.md`。

## 1. 标准流程（命令行，推荐）

```bash
# ① 每次开工前：同步 main
git checkout main
git pull origin main

# ② 新建分支（命名带工单号）
#    实验类 exp/9-cs-hi512 ；代码类 feat/11-multimodel ；修复类 fix/xx
git checkout -b exp/9-cs-hi512

# ③ 做事（写代码 / 跑训练），然后看改了什么
git status

# ④ 只添加你这次要提交的文件（不要用 git add -A，避免误提交）
git add train.py docs/xxx.md

# ⑤ 提交（信息规范：<类型>: <一句话>，可用中文）
git commit -m "exp: ConvNeXt-Small@512 on cache768 (AUPRC 0.8726)"

# ⑥ 推送分支（第一次会弹认证）
git push -u origin exp/9-cs-hi512
```

推送成功后终端会打印一个链接（形如 `Create a pull request for 'exp/9-cs-hi512' on GitHub by visiting: https://github.com/...`），**打开它**，或直接去仓库页面点黄色横幅 **Compare & pull request**。

### ⑦ 在网页上创建 PR
1. 标题：`[#9] ConvNeXt-Small@512 高分辨率实验（AUPRC 0.8726）`
2. 描述：**会自动带出模板**（见 `.github/pull_request_template.md`），逐项填写：
   - 关联 issue（写 `完成 #9 任务A`）
   - 实验结果表（TTA4 AUROC / AUPRC / balmax / epoch 用时）
   - 完整复现命令
   - 权重文件位置（Release 附件链接 / 群文件）
3. 右侧 **Reviewers** 选 `dongmingkun`；**Assignees** 选自己
4. 点 **Create pull request**

### ⑧ 审核与修改
- 队长在 PR 里留言后，**回到同一个分支继续改**：
  ```bash
  git add <改动的文件>
  git commit -m "fix: 按 review 意见修正 XXX"
  git push            # 分支已关联，直接 push 即可，PR 会自动更新
  ```
- **不要为了改问题新开 PR**。
- 队长点 **Merge** 后：
  ```bash
  git checkout main
  git pull origin main
  git branch -d exp/9-cs-hi512      # 可选：删掉本地分支
  ```

## 2. 纯网页流程（只改文档/小文件时用）

1. 打开 https://github.com/dongmingkun/glaucoma_first
2. 进入要改的文件 → 右上角铅笔图标（Edit）
3. 改完后页面下方选 **Create a new branch for this commit and start a pull request**，填分支名
4. 点 **Propose changes** → 下一页点 **Create pull request**，填模板内容
5. 之后流程同上（等队长合并）

> 新增文件：仓库首页 → **Add file** → **Create new file**，其余相同。

## 3. 本项目特有规矩（重要）

| 规矩 | 原因 |
|---|---|
| **权重（`*.pt`）绝不入库** | 仓库不存大文件（.gitignore 已排除）；用 GitHub Release 附件或群文件分享，PR 里贴链接 |
| **训练类 PR 必填指标表 + 复现命令** | 队长要靠它判断能否并入集成；缺了会被打回 |
| **写清 `cache-root` 和 `img_size`** | 不同分辨率的模型必须配对正确的缓存（512 缓存 / 768 缓存） |
| **一个 PR 只做一件事** | 方便回滚；训练结果与代码改动分开提 |
| **不要自己点 Merge** | 由队长合并（他负责集成与提交节奏） |
| **提 PR 后在对应 Issue 留言** | 写"已在 PR #x 提交"，方便追踪 |
| **改动提交流水线（make_submission.py 等）必须跑格式校验** | rows / names / prob range / label logic 四项全 OK 才能提 |

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| `! [rejected] ... fetch first` | 远端有新提交：先 `git pull --rebase origin main`，再 `git push` |
| `Authentication failed` | 见 0.2：改用浏览器授权 / Token / SSH |
| 忘了建分支，直接在 main 上 commit 了（未 push） | `git checkout -b exp/xxx`（改动会跟过去），main 自动恢复干净 |
| 已把 main push 上去了 | 别慌，在 issue 留言说明，队长处理；下次务必先建分支 |
| 误 add 了大文件 | 立刻 `git reset HEAD <大文件>`，别 commit；已在 .gitignore 里的不用担心 |
| 想撤销上一次 commit（还没 push） | `git reset --soft HEAD~1` |
| 不确定该开 PR 还是留言 | 只要**动了仓库文件**就开 PR；纯实验结果报告可直接在 issue 留言 |

## 5. 交付物与验收（训练类 PR 速查）

必须包含：
1. `runs/<exp>/best.pt`、`runs/<exp>/probs_valid_full_tta4.npy`（`python evaluate.py --ckpt <best.pt> --tta 4` 生成）
2. 指标表（一行）：`exp / arch / img_size / cache-root / AUROC / AUPRC / balmax / epoch用时 / 机器`
3. 复现命令（含 `--cache-root`、`--init-from` 等全部参数）
4. 权重下载链接（Release 附件或群文件）

参考基线（队长机器，TTA4）：
- ConvNeXt-Small@384（512缓存）：AUROC 0.98994 / AUPRC 0.86371
- ConvNeXt-Small@512（768缓存）：AUROC 0.98994 / AUPRC **0.87256**
- 当前最优集成：AUROC 0.99110 / AUPRC **0.87733**