# 已授权转换后的验收

用于用户要求“处理/转换数据”，不是普通检查的默认副作用。已有本批授权直接执行，不追加审批。

在独立输出目录保存 processing policy、源索引、成功证据、排除清单，保留原文件。沿用有证据的动作/状态、offset、颜色、任务归并和用途；未确认决定留待解决。检查已对齐标记，偏移仅应用一次。VDS依赖源文件，不能称独立可搬运；最终导出不再依赖HDF5才是standalone。

EE16为未归一化float32绝对xyz+wxyz+gripper×2。可接受四元数先单位化，再分别沿时间连续，动作首帧锚定状态，记录修改数；坏范数不静默修正。保留commanded/measured区别、高频控制和速度前馈；夹爪映射用本机器人配置，容许的小幅实测越界也要记录。

旧项目沿用joint100k EE16统计，self的norm_clip_value为null，官方保持原配置。不因self大z-score改统计或截断动作，不用Joint14统计替代。

视频常为880×480（左640×480头部，右上下各240×240腕部），25FPS；具体服从loader合同。实际RGB正确编码标准MP4，PyAV rgb24不再换通道。Parquet保存action、observation.state、frame_index、timestamp、instruction及源映射、成功证据。frame_index/fps只是派生时间。terminal_rgb不补尾帧。episode ID不与现有数据冲突。

完整提示词完全匹配才复用T5 embedding，不只匹配任务名。critical index、有效窗口、历史帧按真实训练配置生成；VAE缓存/固定随机窗口不是原始转换的默认步骤。

验收层次：

1. 所有轨迹逐项对照源state/action、索引、符号等价、标签依据及源哈希，数量/排除可对账。
2. 全部MP4帧实际解码，核对帧数/FPS/PTS；每条多帧与源实际RGB做MAE/PSNR及三视角目视对照。有损视频不要求byte-equal。
3. 真实loader逐格式/任务/变体取首中尾有效窗口及随机窗口，检查动作长度、图像形状、历史条件、mask、归一化和来源。短轨迹按loader规则处理，不硬编码必须65帧。
4. 需要模型兼容验证且有资源时，小批前后向检查有限loss/梯度，不执行optimizer step或改权重；普通文件检查不必加载大模型。
5. 完成后生成READY及关键SHA，明确未做的重放/物理对齐。只有pilot不能将整批置ready。

截取/补齐、混合比例、任务权重和训练步数是实验配置任务，不并入数据转换。
