# 动静可靠性与稠密／对象运动互补：先行工作边界

2026-09-16，只读原文与作者源码；本轮集中审查一篇直接相关工作，不拼接泛文献列表。未运行外部代码、下载数据或权重、调用 SSH、训练或改变当前实验。源码原字节、URL、逐文件 SHA 见 [manifest](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/research_notes/motion_reliability_prior_art_sources_v1/sources.json)。

**Let Occ Flow 已明确覆盖“运动／静止分解、静止零流约束、动静不平衡处理”；给 D 补一个动态门控不能单独成为新颖性论据。** 其任务是从历史至当前图像预测当前占据及当前网格 flow，原文 §3.4 用光流与静态重投影的不一致识别动态区域。它不是我们四时域材料点 forward 位移的同协议基线。[原文 v2，§3.1、3.4](https://arxiv.org/html/2407.07587v2)

官方主仓库固定为 `eliliu2233/occ-flow@c782571ecc4556cb5a72c2d4be3ba40a2e9e839b`（2026-09-13）；安装文档明确要求作者修改的 `sdfstudio_occ`，其当前 commit 固定为 `b1915c4daa7dfab17568b125b04dbaf309b34f79`（2025-04-14）。以下不是作者历史实验的实际运行证明。

| 直接问题 | 已核发布实现与边界 |
|---|---|
| 动态掩码怎么算？ | `FlowLoss` 在 `no_grad` 内采样当前→下一帧光流伪标签，以 **像素 L2 残差 >8 px**、光流两分量绝对值均 <2000，再与 2D movable 语义掩码相交。静态流来自模型渲染的 surface point 经真实目标相机变换投影，未加物体位移。它是依赖深度、位姿、伪光流误差的训练启发式，不是已校准的“运动可信概率”，也不是 8 m/s 阈值。[flow_loss.py:88–107](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/loss/flow_loss.py#L88)、[neus_head.py:569–585](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/model/head/neus_head/neus_head.py#L569) |
| 静止区域怎么抑制误运动？ | `~dynamic_mask` 上对渲染的水平场位移取绝对值均值；低渲染权重 `<1e-3` 的样本另加 `0.2` 倍流幅正则。渲染 scene flow 的权重被 detach。配置外乘 `FlowRegLoss2d.weight=0.1`。这里不是推理时把静止输出硬归零。[flow_reg_loss_2d.py:57–74](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/loss/flow_reg_loss_2d.py#L57)、[config:213–224](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/config/nuscenes/nuscenes_occ_flow_voxelaffm.py#L213) |
| 动静不平衡的代码是否就是论文公式？ | 通用 loss 有动态权重 `min(保留项数/动态项数,20)`，但所审 nuScenes flow 配置明确 `with_static=False`，也未传 backward flow；静态项先置零，再由 `loss>0` 排除。因此这个配置的剩余有效项均为动态时，该比值为 1。不能拿通用分支或论文公式声称发布配置始终做全静动样本的逆频率重权。另有渲染权重、投影有效性和异常误差筛选，分母不是我们完整物理对象支持。[config:171–191](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/config/nuscenes/nuscenes_occ_flow_voxelaffm.py#L171)、[flow_loss.py:223–274](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/loss/flow_loss.py#L223) |
| 推理时什么可得？ | 配置 `queue_length=3`，dataset 取当前和两个过去 keyframe 图像，利用过去→当前 ego 变换；nuScenes 训练读离线 CoTracker 光流与 movable 语义，loss 顺序把动态 mask 传给后续正则。官方 `occ_only` 路径直接导出 SDF 和二维 `fwd_flow`，不调用上述动态 mask。作者 field 中时间拼接被注释，实际是 `flow_net(tpv)`，没有四个未来 h 的接口。loader 仍可能准备未来图像／标定用于训练和渲染，不能宣称整个 loader 从不读取未来；所核占据导出预测路径并不以未来伪光流做推理门控。[dataset:380–413,428–463](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/dataset/dataset_sweeps_dist_lidar.py#L380)、[head:215–243](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/model/head/neus_head/neus_head.py#L215)、[作者 field:430–470](https://github.com/eliliu2233/sdfstudio_occ/blob/b1915c4daa7dfab17568b125b04dbaf309b34f79/nerfstudio/fields/sdf_custom_field.py#L430) |

对我们的直接含义：已经完成的固定权重诊断显示，在原全部支持上，用 D 填 CRN-CV 未覆盖点，2 s moving XY EPE 比 CV 降 `0.5621 m`，但 stationary 增 `0.0549 m`、ambiguous 增 `0.0258 m`。这证明现有两个预测有可利用的分区互补，同时存在代价；**尚未证明当前输入能识别哪个未覆盖点适合 D，也没有占据收益结论**。来源仅为 [固定 D 分析](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/fixed_D_coverage_complementarity_analysis_v1.json)，SHA `b91a892b34bf2fea74c226971972e4cee9d53d0b20e11e359beaea6730532b54`。

下一步值得区分的问题是：**当前可得的信息能否识别 D 的有效补覆盖与静止漂移，而不只是按位移幅度收缩所有输出？** 若研究四时域 D 预测对近似常速模型的残差，应把它作为待验证的可靠性线索：恒定的错误漂移也可以有零常速残差，真实加速／转弯却可能残差大；D 四头又共享输入／参数，跨时域一致性不是独立观测证据。Let Occ Flow 的“光流−静态重投影”使用相邻图像的外部对应证据，不能等同于对 D 自身四个输出做自洽检查。

建议后续候选仅围绕这一判别展开：在相同全支持、监督与预算下，比较只看幅度／当前覆盖的信息，与加入合法历史观测或时域一致性的信息；对照需能判定改善是否仅来自整体收缩。若新增信息不能在保留 moving 补覆盖的同时解释／降低 stationary 与 ambiguous 代价，或与相同容量的幅度控制相当，就不把结果归因于新的运动可靠性机制。未来 GT 动静组只可用于训练目标和固定评价，不能成为推理门；本笔记不冻结路线、不选择阈值或批准新训练。

可声称的差异限于：我们的证据来自冻结稠密 D 与检测 CV 的实际覆盖互补及其全分母代价；所读 Let Occ Flow 并未实现这对预测器的覆盖混合。它已经占据动静分解／静止正则的通用思想，传感器变化或添加一个 sigmoid 都不足以推出新颖性。公开访问无失败；未核历史论文权重对应的精确源码版本、伪标签准确率和本环境复现，亦未对该方法作四 h EPE／未来 GMO 比较。
