# 基础检查器使用

依赖 Python>=3.10、numpy、h5py、opencv-python；导图另需 Pillow，可选 FK 需 scipy。不依赖 VPP2 repo import，不需 GPU。

先建立明确的 primary 文件清单 `inputs.jsonl`，一行一个 JSON 对象。path 用绝对路径，task 是本轮统计名称，不自动由目录猜：

```json
{"path":"/absolute/task/hdf5/episode_0000000.hdf5","task":"task_name"}
```

可选 expected_sha256（供方完整HDF5 SHA）；success 必须为 JSON boolean 且 success_evidence 非空，仅在文件外已找到有效证据/用户已确认时填。例如：

```json
{"path":"/absolute/task/hdf5/episode_0000001.hdf5","task":"task_name","success":true,"success_evidence":"User explicitly confirmed this batch on YYYY-MM-DD; preserve original label absence"}
```

其他输入字段（source_directory、canonical_task、variant、layout_origin、合同证据等）原样留存在 input_evidence，检查器不把它们自动解释为真实证据。不能为了填示例而编造日期/成功。先明确 hdf5/ 与 entries/ 等重复交付口径，不要把所有递归找到的文件都当独立轨迹。

执行示例，替换绝对路径；`SKILL_DIR` 为此技能目录，`PYTHON_BIN` 为已有可用环境：

```bash
SKILL_DIR=/root/.codex/skills/robodojo-selfdata-check
PYTHON_BIN=/mnt/project_modelware/hxguan/yzy/envs/vpp2-physbrain/bin/python
"$PYTHON_BIN" "$SKILL_DIR/scripts/check_hdf5.py" \
  --inputs /absolute/report-inputs.jsonl \
  --out /absolute/new-audit-directory \
  --images all --workers 2
```

已核实当前机器人/坐标系相容时可增加 `--kinematics-json /absolute/x5_kinematics.json`；核实官方 EE16 统计口径后可增加 `--stats /absolute/dataset_stats.json`。无对应适配表示时 FK/统计明确标记待补；不自动把关节数组解释为 EE16 或生成训练标签。

`--images all`：每个 JPEG 1/8 解码，首/1⁄4/中/3⁄4/尾及黑/纯色候选原尺寸解码。`sample`：仅这5个索引；`none`：不查图像。三者均全读数值并算完整文件SHA；抽图不代表轻量读盘。sample/none 不输出整段图像流指纹。

输出新目录，不覆盖已有报告，原 HDF5 只读打开：

- inputs.json、check_config.json：本次输入、参数、标定与统计文件SHA。
- episodes.jsonl、progress.json：逐条结果和完成数量。控制台每25条打印进度。
- task_summary.csv：逐任务数量、成功标签、错误/待复核、解码/FK/统计覆盖。
- duplicate_groups.json：本输入集合内各类精确重复，含 action-only 指纹；不自动删除。
- image_flags.json：解码失败与原尺寸确认的黑/纯色候选，0起始RGB帧号。
- report.json：状态始终为基础扫描完成，training_ready=false。exit 2 表示存在结构/数值/解码错误；exit 0 也可能有语义、标签等待复核项。

仍需补做：交付附件/成功阈值验证、相机标定、任务归并/Open分类、cross-dataset 重复/布局比较、offset与颜色证据确认、适配器专属字段。该脚本不重放仿真、不独立证明物理对齐、不转换数据。完整任务由 SKILL.md 的工作流完成。

## 黑帧与邻帧

```bash
"$PYTHON_BIN" "$SKILL_DIR/scripts/review_frames.py" \
  --hdf5 /absolute/episode.hdf5 --frame 69 --radius 1 \
  --color rgb-direct --out /absolute/report/episode_rgb69.png
```

--frame 为RGB序列索引；radius=1展示前/当前/后。表明各相机及 rgb_step_index 对应控制行；越界邻帧显示缺失，不伪造。`--color standard-jpeg`表示标准JPEG，`unknown`展示两种候选。颜色确定依赖证据，不依赖程序选项本身。

## 自测

```bash
"$PYTHON_BIN" "$SKILL_DIR/tests/test_checkers.py"
```

在临时目录验证正常/损坏HDF5、四元数符号、布尔标签、索引、重复轨迹、FK、统计及导图；不读取或修改训练数据。
