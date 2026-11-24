# 检查合同与证据口径

## 交付、数量、来源

先确定 primary 轨迹，再统计文件数。核对压缩包清单、manifest 和实际文件的缺失/多余、大小、哈希、目录别名、任务变体和编号。编号不连续未必缺文件；没有上游清单时，只能证明收到的文件可读，不能证明计划交付完整。

分别报告轨迹数、RGB 时刻数、三路图像数、控制行数、频率、时长和字节数。250 Hz 控制行不是 25 Hz 视频帧；terminal_rgb 单列，不拼入正常训练时序。扫描前后核对源 size/mtime，有供方 SHA256 则逐个比对。

`hdf5/` 和 `entries/` 的同一文件可能只是交付副本，不是额外示范；不同视觉下相同动作可能有意保留。完整文件 SHA、数值 SHA、图像流 SHA、EE16 action SHA、state+action SHA 分开描述。精确无匹配不表示没有近重复或相同布局。跨旧数据比较必须列出参考集合数量、格式和方法。

目录名未必是 canonical task。历史项目有 fold_clothes_v2/fold_clothes_random 归 fold_clothes，tic_tac_toe 归 play_tic_tac_toe，但不能机械删除所有 `_v2`。结合指令、属性和清单确认，保留原目录和变体。

## 格式与数值

| 字段 | 识别重点 |
|---|---|
| `state/left_ee_poses`、`action/*`、`vision/cam_*/colors` | legacy EE / joint25。pose 通常 `[T,7]` xyz+wxyz；有 state pose 不等于有 action pose。 |
| `observations/ee`、`observations/joints`、`actions/left_arm_joint_state`、`rgb/*` | native joint25，issued joint target 需 FK 才能转为末端目标。 |
| `observations/qpos`、`actions/joint_position`、`rgb_step_index`、`rgb/*` | dense control + RGB。16 维通常是关节/指关节，不是 EE16。 |
| `data.parquet` + `video.mp4` + source mapping | 已转换数据，核对列、源索引、PTS/FPS、转换合同及真实 loader。 |

频率由本批证据读取，25/250 不是固定真理。全部数值检查 dtype/shape/长度/有限值；必需字段检查缺失。时间索引检查维度、整数性、递增、越界及图像数量一致。终止标志、next observation 连续性和尾部残缺控制周期按源 schema 判断，不强行补帧。

项目 EE16 为 `[左xyz, 左wxyz, 左gripper, 右xyz, 右wxyz, 右gripper]`。核对单位、坐标系、EE link、关节顺序、夹爪正负号/范围。FK 使用匹配 URDF、EE link 和基座标定；不能凭外观复用。旧 X5 适配器使用位置误差 0.5 mm、旋转 0.002 rad 的一致性阈值；新硬件另定。

四元数分别报告范数、零/非有限值、相邻 dot<0 数量和帧号，真实角度是 `2*acos(abs(dot(q0,q1)))`（先单位化）。dot<0 不是 180° 转动。动作极值/突变及归一化范围给出分位数和位置，大 z-score 不自动判损坏。

## 时间对齐

分开记录：声明（属性/采集合同/本批用户指示）、结构（索引/长度/时间戳/已应用偏移）、独立物理验证（视觉特征与状态对应或重放）。没做的层次要写明。

2026-11-12 批次曾按用户规则对部分任务不移帧、其余按旧官方图像慢一帧处理；这不是新数据默认名单。后续多批明确 offset=0，必须同帧。

确认 +1 时，25 Hz 输出 `new_image[t]=source_image[t+1]`，state/action 保留 t，有效长度 T-1。dense control 的 +1 是 RGB 帧，不是一条控制记录；保留 RGB→控制映射并核对终止边界。检查 alignment_view_version、image_shift_applications、source mapping，禁止二次偏移。

`action[t]≈state[t+1]` 可能来自动作定义，不是图像对齐证据。光流/几何诊断图需解释点来自哪些像素、线由哪个状态/标定/投影生成、比较哪些 lag；给出相机/帧号/单位/误差/置信度。静止、遮挡、慢运动或标定不准允许“不确定”。

## 动作语义

区分 issued_joint_target、issued_ee_target、post_execution_measured_state。action 字段名不足以区分；结合采集代码、供方定义、source mapping 和数值关系。

历史 swap_T 交付记录执行后实测状态，其他部分自采数据记录发出的目标。二者可在明确监督定义后用于训练，但不能声称含义相同，不能悄悄改标签。官方数据也应检查实际版本与转换代码。

旧 250→25 Hz 适配曾取 `state=qpos[s[j]]`、`action=issued_target[s[j+1]-1]`，末段按真实剩余控制范围处理，表示视觉周期末发出的目标，并保留原高频 position/velocity。不是随意每10行取第一行，也不是换成下一帧实测状态。新控制合同先核实再复用。

## 颜色与图像

| 约定 | JPEG/视频还原为实际 RGB |
|---|---|
| 标准 JPEG：RGB 已正确变成 OpenCV BGR 后编码 | cv2.imdecode 后 BGR→RGB；PIL RGB 通常直接正确 |
| RGB-direct JPEG：原 RGB 数组直接交给 OpenCV 编码 | cv2.imdecode 数组直接作为实际 RGB；PIL RGB 与实际场景红蓝相反 |
| 标准 MP4 | PyAV rgb24 直接用；OpenCV VideoCapture 按其 BGR 接口转换 |

两个解码器颜色相反只能说明表示不同，不能单独确定真实颜色。核对编码代码、codec 属性、校验收据及已知颜色物体/任务。逐批记录；正确转 MP4 后不二次交换。

完整 JPEG 1/8 解码不等于原尺寸全检。曾使用每条每相机首/1⁄4/中/3⁄4/尾原尺寸抽查，候选原尺寸复核。筛查为空间标准差<1.2 或近黑比例>=0.99，复核为标准差<1 或近黑比例>=0.995（全部通道<3）。这是候选阈值，不能凭此判原因；白墙或遮挡也可纯色。

坏帧图包含前/当前/后一帧及三相机。连续段另取段前、首/中/尾、段后。标明0起始RGB索引、控制索引与不存在的邻帧。解码失败、黑帧、纯色、尺寸变化、相邻重复分开统计。

## 成功、用途

成功依据分为文件标签、供方验收/重放、用户确认、本机实际重放。严格解析布尔值，`bool("false")` 是错误做法。不同任务 score 的成功阈值不同，不能统一为1或100；附件要绑定路径或SHA。

缺 success 不是失败。先查随包附件；已有明确用户确认则写旁路标签和来源，不重问、不冒充原存标签。本批无确认又无证据，只问这部分，继续其他检查。

evalgroup 名字、官方评测布局、自采来源是不同维度。继承已确认用途；未确认不擅自划入训练。Open/非Open 按任务定义/官方任务集合，不能看字符串是否有 open。保留 parent layout、layout SHA、动作变体等来源以区分已见/未见布局。

## 输出和完成标准

- task_summary.csv：目录/canonical task/变体、primary数、RGB/控制数、图像覆盖、成功证据分类、异常轨迹数、重复统计。
- episodes.jsonl：路径/大小/mtime/SHA、schema、数值/相机统计、RGB索引、offset/颜色/action/state语义及证据、异常。
- episode_annotations.csv：success（可空）、success_evidence、是否原存标签、offset_declared/applied、color_decoding、include_in_training（未决定为空）、exclusion_reason、布局来源。
- duplicate_groups.json、坏帧清单/PNG、待补信息。区分文件损坏、表示可修正、语义待核实、供方追溯缺项。

只有基础扫描则写“基础扫描完成，专项待复核”；部分样本写“抽查”。覆盖本次全部 primary 且列明未做验证后才能称“本次检查完成”。检查完成不自动等于训练就绪。
