# 如何提交合并请求（Pull Request）— 队员操作手册

> 目标：让你做的事（代码 / 实验 / 文档）安全地进入 `main`，队长审核后合并。
> 原则：**一个 PR = 一件事**；小步提交；不要直接推 `main`。

## 0. 前置条件（首次使用）

1. 接受仓库邀请（检查邮箱或 https://github.com/dongmingkun/glaucoma_first 的邀请提示）
2. 安装 GitHub Desktop（本机已有）或 Git，并登录你的 GitHub 账号
3. 克隆仓库并搭好环境（环境搭建步骤见 `docs/TEAM_HANDOFF.md`）

## 1. 标准流程（GitHub Desktop，推荐）

### 第 1 步：同步最新代码
打开 GitHub Desktop → 选中 `glaucoma_first` → 点顶部 **Fetch origin**，如果有更新再点 **Pull origin**。
> 每次开工前都要做这一步，避免基于旧代码改。

### 第 2 步：新建分支（不要直接在 main 上改）
1. 点顶部中间的 **Current Branch** → **New Branch**
2. 命名规范：
   - 实验类：`exp/<工单号>-<简述>`，例：`exp/9-cs-hi512`
   - 代码类：`feat/<工单号>-<简述>`，例：`feat/11-multimodel-predict`
   - 修复类：`fix/<工单号>-<简述>`
3. 确认基于 `main`，点 Create Branch

### 第 3 步：做事 + 提交（Commit）
1. 正常写代码 / 跑训练（注意：`*.pt`、`*.npy`、缓存、数据不会进 git，这是故意的）
2. 完成后回 GitHub Desktop，左侧会列出改动文件
3. **只勾选你这次要提交的文件**（右下角 Summary 写提交信息，规范：`<类型>: <一句话>`，例：
   `exp: ConvNeXt-Small@512 on cache768 (AUPRC 0.8726)`
   `feat: predict.py 支持多模型集成`）
4. 点 **Commit to exp/9-cs-hi512**

> 如果训练产物里有想给团队看的东西（例如指标 csv），放到 `runs_reports/<exp>.md` 或写进 PR 描述，不要提交权重。

### 第 4 步：推送分支
点顶部 **Publish branch**（第一次）或 **Push origin**（后续）。

### 第 5 步：创建合并请求（PR）
1. GitHub Desktop 会提示 **Create Pull Request**，点它（或去网页：
   https://github.com/dongmingkun/glaucoma_first → 会出现黄色横幅 "Compare & pull request"）
2. 标题写清楚：`[#9] ConvNeXt-Small@512 高分辨率实验（AUPRC 0.8726）`
3. 描述会自动带出模板，**逐项填**：
   - 关联 issue（`完成 #9 任务A`）
   - 实验指标表（TTA4 AUROC / AUPRC / balmax / epoch 用时）
   - 复现命令（完整可直接执行）
   - 权重文件在哪（Release 附件链接 / 群文件位置）
4. 右侧 **Reviewers** 选 `dongmingkun`（队长），Assignees 选自己
5. 点 **Create pull request**

### 第 6 步：等审核 / 改问题
- 队长会在 PR 里留言。要改动时：**回到同一个分支继续 commit + push**，PR 会自动更新（不要再开新 PR）
- 队长合并后：回到 GitHub Desktop，切回 `main` → **Pull origin** → 删除本地分支（Branch → Delete）

## 2. 命令行等价流程（熟悉 git 的话）

```bash
git checkout main && git pull origin main          # 同步
git checkout -b exp/9-cs-hi512                     # 建分支
# ... 做事 ...
git add <你改的文件>                                # 别用 git add -A，避免误提交
git commit -m "exp: ConvNeXt-Small@512 on cache768 (AUPRC 0.8726)"
git push -u origin exp/9-cs-hi512
# 然后到网页点 "Compare & pull request"
```

## 3. 本项目特有的规矩（重要）

| 规矩 | 原因 |
|---|---|
| **权重（*.pt）绝不入库** | 仓库不存大文件；用 GitHub Release 附件或群文件分享 |
| **训练类 PR 必须写指标表 + 复现命令** | 队长要判断能否并入集成，指标和命令缺一不可 |
| **写清 cache-root 和 img_size** | 不同分辨率的模型要配对正确的缓存，集成时才知道怎么加载 |
| **一个 PR 只做一件事** | 方便回滚；训练结果和代码改动分开提 |
| **不要自己点 Merge** | 由队长合并（他负责集成与提交节奏） |
| **提交前跑一遍校验** | 改动提交流水线的，务必跑 `make_submission.py` 的格式校验 |
| **提 PR 后在对应 Issue 留言** | 说明"已在 PR #x 提交"，让队长在 issue 里追踪 |

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| 推不上去 / 提示 `rejected` | 远端有新提交：`git pull --rebase origin main` 后再 push |
| 忘了建分支，直接在 main 上改了 | `git checkout -b <新分支名>`（改动会跟着过去），main 会恢复干净 |
| 误提交了大文件 | 立刻在 issue 里说明，不要继续 push 更大的版本；队长会帮你处理 |
| 想撤销上一次 commit（还没 push） | `git reset --soft HEAD~1` |
| PR 里有不想合的文件 | 在 PR 界面逐文件检查；必要时新提交把它删掉 |
| 不确定该开 PR 还是直接留言 | 只要动了仓库里的文件，就开 PR；纯实验结果报告可以直接在 issue 留言 |

## 5. PR 描述模板（系统已自动带出，这里备一份）

见 `.github/pull_request_template.md`：包含「做了什么 / 类型 / 交付物清单 / 实验结果表 / 复现命令 / 验证方式 / 风险与注意」。
**实验类 PR 的实验结果表和复现命令是必填项**，缺了会被打回。