# UniAD 与对象—空间对应候选的直接重合核对

2026-09-16。结论：**UniAD 是强直接先行工作。“共享对象状态、对象 query 向 BEV 注入、当前与未来位置先验、实例 attention、联合训练”均不能单独作为本候选的新颖性。** 本核对只补充先行工作边界，不改原简报、正式 V/G 结论或 N/P/Z 诊断。

读原论文 [Planning-oriented Autonomous Driving，arXiv v2，§2.2–2.5](https://arxiv.org/html/2212.10156v2)，并核 [CVPR 2023 正式条目](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)。源码只读作者仓库三个直接文件，固定 commit `609ee083ea51c3521c323f1279dfc4cee0e60467`，三者均与本次读取的远端 raw 内容逐字节相同。该 commit 来自此前固定的 **v2.0** 源码，不冒充 2023 发表时的确切版本；以下分别注明原文和实际代码依据。完整 URL、SHA 和范围见 [来源清单](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/research_notes/uniad_motion_occupancy_overlap_audit_v1_sources.json)。

**实际连接比“共享 query”更具体。** MotionHead 输出逐 agent、多模态的 motion decoder 特征，同时保留 track query 与由当前检测中心产生的位置 embedding。OccHead 取最后一层 motion 特征，先在模式轴 max-pool，**保留 agent 轴**，再把它与 track query、当前位置 embedding 拼接融合。每个时域的 temporal MLP 从同一个融合 instance query 生成相应状态；递推的是稠密 BEV state，不能把此代码说成 instance query 也逐时递推。BEV 像素作为 Q，时域 instance query 作为 K/V，最后仍输出逐实例占据 logits。[MotionHead 当前位置与返回值](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py#L243-L350)、[OccHead 时域交互及融合](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L198-L283)。

**它已有空间对应约束，但不是框内硬 ROI。** 每个 instance embedding 与当前层 BEV state 点积产生预测 mask，sigmoid 后阈值化为 cross-attention mask；布尔 mask detach，全部 key 均被屏蔽的像素则解禁全部 key。浮点 mask logits 另受实例 mask/dice 辅助监督。因此这是“预测的实例占据限制 pixel→agent attention”，不是固定距离邻域或直接把 GT mask 送进前向；也不能把其局部性解释成严格几何局部。训练复用上游匹配的 instance identity，构造各 query 的未来实例 mask 目标。[mask 构造](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L173-L196)、[实例目标及辅助损失](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L331-L404)。

**当前/未来地址与训练反馈也并非空白。** 原文 §2.2 的 motion 位置先验含当前点和上层预测终点，并在预测终点附近取 BEV 证据；固定源码的调用端确实传入当前中心 embedding、场景/agent 锚点终点 embedding、轨迹参考点和回归分支。本次未额外读 motionformer 内部，故不把内部实现也列作已核。OccHead 消费的是 motion 特征，不直接读 `all_traj_preds` 数值后做轨迹 raster/splat。总模型将同一次 motion 输出交给 occupancy 训练，没有在这条边整体 detach；OccHead 的可选 detach 只针对当前位置 embedding，融合中的 motion/track 特征仍有结构上的梯度路径。布尔 attention mask detach 不等于整个连接断梯度；本次也未执行 VJP 来证明实际梯度效用。[调用与位置先验](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py#L245-L322)、[联合训练调用](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py#L184-L214)。未读运行 config，不将构造器的阈值或 detach 默认值冒充实际论文配置。

与[现有候选简报](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/research_notes/object_to_bev_correspondence_candidates_v1.md)相比：

| 候选表述 | UniAD 已有重合及剩余边界 |
|---|---|
| 同一对象动态状态供 motion 与 occupancy 使用 | 直接重合：融合 motion/track/position 且保留实例轴。移除 32 scene latents、改直接 object attention 本身不构成方法差异。 |
| 保留当前和未来地址，解决对象到空间的关系 | 直接重合到当前点/预测终点先验及 instance-mask attention。仅称“双地址”不足；需具体说明两种地址对应什么预测量和监督。 |
| 占据在未来 x 查询，物理在源 p0 查询同一对象状态 | 尚可检验的窄区别：本项目要求任意 t0 材料点的 forward 位移 `p_h−p0`；所读 UniAD 分支是 agent 轨迹与未来实例占据，未见这种源材料点读出接口。这里不能推成“UniAD 没有源—目的概念”，也未完成全领域新颖性证明。 |
| 不借 GT 关联，保留全框、O 稠密回退 | 数据/架构约束不同：本候选 identity 只是预测框 index，GT 只作既定 loss/eval；UniAD 训练复用匹配及实例目标。所读 OccHead 无 query 时占据输出零，而本候选保留 O 背景/漏检路径。这些约束值得明确，但不是贡献证明，更不能仅凭不同传感器、类别或任务声称创新。 |

**主候选应被定位为结构检验，尚非可主张的新方法。** 真正要证的是：在同一预测对象假设中，区分“源材料点如何访问状态”与“未来空间如何访问状态”，能否在相同输入、容量、监督、预算和完整支持下提供可辨别增量，同时改善原 GMO 与原 forward 物理目标。原简报只切换 motion 键锚点 `c0/c_h` 的控制仍有明确判别意义，但仅能支持该新结构中的地址效用，不能倒推旧 conditioner 必然错位，也不能用赢错配控制代替联合目标成功。D 与 CRN-CV 的完整强对照仍须保留；若引入直接 CV 基底，必须另有同基底控制。

UniAD 的实例辅助监督已经是一种学习空间归属的办法。若以后采用这类监督，必须先确认本项目原始标签/关联确实可用，不能把缺失实例补全或 GT ownership 偷渡为推理输入；本次不建议新增监督或调整冻结实验。我们候选保留 O 稠密覆盖的要求也延续 ImplicitO 已指出的检测漏检限制，并非新发现。

操作范围：只读论文网页、三个 UniAD 文件及已有候选；对固定 commit 的同三文件做远端内容比对。未下载权重、未导入或运行模型、未执行 Torch/GPU/训练/评分；只新建本札记和来源清单，没有修改原简报或当前诊断。未以 UniAD 论文成绩代替本项目实测。
