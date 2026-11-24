# 本机 VPP2 复用入口

2026-11-24 的证据索引；这些路径可替换，不是新批次参数。

仓库 `/mnt/project_modelware/hxguan/yzy/workspace/video-prediction-policy-2`；数据 `/mnt/project_modelware/hxguan/yzy/dataset/wuwen_selfrobodojo`。Python 优先用 `/mnt/project_modelware/hxguan/yzy/envs/vpp2-physbrain/bin/python`，不为检查重装训练环境。

## 只读案例（仓库相对路径）

- `runs/wuwen_selfrobodojo_audit_20261112_113205/audit.py`：初批三类 HDF5、数值/索引/JPEG。相邻 `check_dark_frames.py`、`export_black_frame_review.py`：全帧暗/纯色复核及邻帧证据。主函数绑定旧数据根，先读函数，不直接重跑。
- 同目录 `alignment_review/`、`show_alignment_example_zh.py`：图像偏移诊断与局限，旧名单不能套新批。
- `runs/wuwen_newdata_audit_20261113_142812/audit.py`：新增1969，含 Parquet、dense control deposit_coin、measured-state swap_T。
- `runs/wuwen_new8_audit_20261123_1905/audit_batch.py`：八目录2600，全行FK、标定、EE16归一化。
- 同目录 `check_support.py`、`finalize.py`、`report_zh.txt`、`READY.json`：附件/成功证据绑定、交付副本、用户确认、最终覆盖范围。

适配旧代码时，把必要函数放进新的参数化脚本。旧主函数常硬编码路径、任务数、成功/offset策略和输出目录，不应直接用于新批或覆盖旧报告。输入不相容则补适配，不跳过检查。

## 转换/验收（已授权时）

`scripts/robodojo/`：

- `align_wuwen_hdf5.py`：旧用户规则的VDS，依赖原文件，防二次移帧；不复用旧 ZERO_TASKS 决定新批。
- `prepare_wuwen_training.py`：X5 FK/标定、连续四元数、夹爪映射、原控制旁路留存。
- `prepare_wuwen_new1969.py`：目标/实测标签保留、逐批颜色合同、完整视频验证。
- `prepare_wuwen_new2600.py`、`validate_wuwen_new2600.py`：最新offset0批次，数值→媒体→loader→小批模型验收。不是任意批次通用主函数。
- `validate_wuwen_training.py`、`validate_wuwen_new1969.py`：独立源数据、索引、颜色和归一化对照。

已验证 X5 标定：
`/mnt/project_modelware/hxguan/yzy/dataset/wuwen_selfrobodojo_vpp2_training_20261112/conversion_assets/x5_kinematics.json`

官方 EE16 统计：`weights/robodojo_release/checkpoints/joint2b_s100000/dataset_stats.json`。先确认机器人关节/基座/EE link，和统计维度；不是Joint14统计。

## 不能外推的旧决定

- 初批4414：用户确认成功、重复动作保留、pickup evalgroup纳入训练；4条黑帧整条排除，最终4410。不是所有未来批次的成功授权。
- 新2600：明确offset0、RGB-direct；2000供方证据，600 align_blocks_v2/plug_in_charger由用户确认；480 entries文件为已核对SHA的副本。1425条使用官方评测布局，仍是self来源。
- 实验9/10“每任务300采样名额、不足重复补齐”仅在训练manifest生效。原始检查/转换保留所有真实primary；不复制视频补样本。
