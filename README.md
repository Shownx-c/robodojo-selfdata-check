# RoboDojo 自采数据检查 Skill

面向 Codex 的 RoboDojo 数据验收 skill，整理自 VPP2 项目的实际检查和转换经验。

覆盖数据数量与结构、三视角 JPEG、黑帧及前后帧证据、四元数连续性、时间偏移、颜色通道、动作语义、成功标签、重复轨迹，以及已授权转换后的加载验收。

## 安装与使用

将本仓库放到 Codex 的 skills 目录。目标目录尚不存在时：

```bash
git clone https://github.com/Shownx-c/robodojo-selfdata-check.git ~/.codex/skills/robodojo-selfdata-check
```

使用自定义 CODEX_HOME 时，安装到其 skills 子目录。私有仓库需要当前机器已登录有权限的 GitHub 账号。

在 Codex 中调用：

```text
$robodojo-selfdata-check 检查 /数据目录，按之前的标准输出统计、问题清单和证据图。
```

技能入口：[SKILL.md](SKILL.md)。直接执行工具的输入格式与命令见 [检查器使用说明](references/checker-usage.md)。

## 运行环境

Python 3.10 或更高版本；基础扫描需要 numpy、h5py、opencv-python，导图需要 Pillow，可选 FK 检查需要 scipy。优先复用现有机器人训练环境。

```bash
python tests/test_checkers.py
```

基础扫描工具只读原始 HDF5，输出逐轨迹报告、逐任务统计、重复组和图像候选问题。扫描完成不自动代表训练就绪；偏移、颜色、动作语义与成功标签需要对应批次的证据。不能把旧批次规则直接用于所有新数据。

references/project-reuse.md 保留了原 VPP2 工作区的案例和路径，换机器后按实际位置替换；独立扫描与导图脚本不依赖该仓库的 Python 模块。

## 验证范围

已通过 5 项合成数据行为测试，并用 6 条真实轨迹验证三种原始格式；扫描 7,773 张缩小解码图像和 144 次原尺寸复核，复现已知样本右腕连续 53 帧黑帧。此验证未重放仿真，也不代表对任何新批次的成功或物理对齐认证。

仓库不包含数据集、图像样本、模型权重或认证信息。
