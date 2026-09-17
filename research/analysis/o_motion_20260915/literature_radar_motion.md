# 从 O 出发的雷达运动与占据预测文献核查

核查日期：2026-09-15。只查公开论文、作者项目/仓库，并只读本地资料；未连接训练服务器、未运行模型、未修改旧项目。代码结论绑定下表 commit；没有把论文宣称等同本地复现。

**建议先做“有监督的 source displacement → 真正连接预测的 feature transport”，以 aux-only、zero-flow、错配 flow 为对照。** 下一优先级是保留过去帧关联的 camera–radar 局部匹配，补充径向以外的运动证据。继续增加小 gate、把径向速度直接当完整二维速度、或者仅扩 memory，并不能解决已经暴露的问题。

本轮读取的[完整共同验证](../m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1/report.md)给出 O 的 future macro GMO 14.7478%，相对 M0_fp32 +0.8029 pp、相对同预算 C/native1 +1.0201 pp。O 只改变训练目标；GMO 是**可移动语义类别**，不是实际运动标签。旧 M1/M3/S3 未带来收益是父任务已审计的上下文，本轮没有重跑。因而这些结果既不证明雷达运动无信息，也不证明 O 的物理运动已经更准。

## 最相关的一手工作

| 工作与核实出处 | 实际机制、监督与推理边界 | 对本项目的含义 |
|---|---|---|
| **CRT-Fusion: Camera, Radar, Temporal Fusion Using Motion Information for 3D Object Detection**，NeurIPS 2024。[论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/c49a28241640407b23bba8f2495f4bc9-Paper-Conference.pdf) · [作者代码](https://github.com/mjseong0414/CRT-Fusion) | 相机/雷达融合后，两个小 CNN 预测 BEV 的二维速度及 object-presence，再用速度补偿历史特征。训练把标注框覆盖区域赋予框速度；推理用预测速度。正文 §3.2–3.3，代码 motion loss 与 history fusion 已读。 | 最直接的可移植骨架：**显式速度监督 + 运动传播**，而非只改融合权重。其 presence 是框投影，不是我们的完整三维 GMO；检测 mAVE/NDS 不能代替 future occupancy 收益。 |
| **CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception**，ICCV 2023。[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_CRN_Camera_Radar_Net_for_Accurate_Robust_Efficient_3D_Perception_ICCV_2023_paper.html) · [代码](https://github.com/youngskkim/CRN) | 雷达辅助图像到 BEV 的深度/视图变换，multimodal deformable attention 处理跨模态偏差；原任务为检测、跟踪、BEV 分割。查阅正文、README 与模块入口。 | 可借鉴融合发生在几何信息丢失之前这一点；没有显式场景流目标，单独移植不能宣称运动识别改善。 |
| **Self-Supervised Scene Flow Estimation with 4-D Automotive Radar (RaFlow)**，RA-L / IROS 2022。[论文](https://arxiv.org/abs/2203.01137) · [代码](https://github.com/Toytiny/RaFlow) | 两帧雷达点云关联；径向位移、soft Chamfer、邻域平滑三种自监督约束，加静态刚体运动修正。已读 loss、两帧 correlation 与 forward。 | 径向约束只限制一个方向，Chamfer 还会受稀疏、遮挡和错误对应影响。应借鉴**多时刻 correspondence**，不能把径向损失本身当切向监督。 |
| **Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision (CMFlow)**，CVPR 2023。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_Hidden_Gems_4D_Radar_Scene_Flow_Learning_Using_Cross-Modal_Supervision_CVPR_2023_paper.html) · [代码](https://github.com/Toytiny/CMFlow) | 训练用里程计、LiDAR tracking、光流伪监督；光流约束为 warped 3D point 到相机射线的距离。推理只需两帧雷达；静态 mask 由模型预测，训练可用伪标签。CMFlow-T 用 GRU 更新全局点特征。方法及实际 loss/forward 已读。 | 与旧 radial-only 路线实质不同：**训练阶段引入额外运动约束**，部署不增加 LiDAR。代价是同步、遮挡、伪标签误差和 VoD→nuScenes 域差异；作者权重不能直接当我们的运动真值。 |
| **TARS: Traffic-Aware Radar Scene Flow Estimation**，ICCV 2025。[论文](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_TARS_Traffic-Aware_Radar_Scene_Flow_Estimation_ICCV_2025_paper.html) · [作者页](https://osvia.org/TARS/) | 分层点匹配之外，引入由检测特征、点流特征及 GRU/attention 构成的粗 Traffic Vector Field；点再读取局部 traffic context。§3、补充 ego 分支已读。两帧输入中的目标帧用于当前两帧流估计；VoD ego 版本沿用跨模态监督。 | 稀疏雷达不宜仅在少数点内强求刚体拟合，可利用较广的视觉语义/道路背景。目标帧必须是过去→当前，不能在预测未来时偷读未来传感器。未找到可公开读取的作者实现，暂作概念参照。 |
| **RaLiFlow: Scene Flow Estimation with 4D Radar and LiDAR Point Clouds**，AAAI 2026，40(5):4012–4021。[出版页](https://ojs.aaai.org/index.php/AAAI/article/view/37404) · [正文](https://arxiv.org/html/2512.10376v1) · [代码](https://github.com/FuJingyun/RaLiFlow) | 雷达与 LiDAR 两帧 pillar 特征，局部双向跨注意力以径向运动热图调制，迭代流预测；重做 VoD 雷达 flow 标签，处理落在框外的动态回波。正文数据、DBCF、监督与训练设置已读。 | **推理需要 LiDAR**，不能作为同模态替代模型。可借局部匹配/实例一致性，但其新标签、过滤和传感器条件不能移来改变我们的 GT。低径向速度并不等于静止。 |
| **TEOcc: Radar-camera Multi-modal Occupancy Prediction via Temporal Enhancement**，ECAI 2024（作者仓库标注）。[正文](https://arxiv.org/html/2410.11228v1) · [代码](https://github.com/VDIGPKU/TEOcc) | 随机遮去一个历史视觉体素特征；长/短期 3D 卷积结合雷达重建历史，再由共享 occupancy head 监督。§3 与实际两个 detector 类已读。 | 可借**过去可观察状态的辅助重建**，使时序表示获得可检验目标；它不是未来 flow。公开默认 radar config 与论文辅助分支存在接线差距，不能直接说开箱可复现（见下）。 |
| **Let Occ Flow: Self-Supervised 3D Occupancy Flow Prediction**，CoRL 2024，PMLR 270 (2025)。[出版页](https://proceedings.mlr.press/v270/liu25e.html) · [正文](https://arxiv.org/html/2407.07587v2) · [代码](https://github.com/eliliu2233/occ-flow) | 历史相机预测当前 occupancy 与水平 flow；TPV、双向时间聚合、3D refine，加光流/深度渲染监督。先静态几何再联合 flow；用光流减 ego-induced flow，并结合可移动 mask 区分运动。§3、附录与 flow loss/config 已读。 | 直接提醒几何与运动可相互补偿、静态样本会淹没运动。可借可见点光流约束；整套 SDF/renderer 重写成本高，且当前 flow 不等于 0.5–2 s 未来 occupancy。 |
| **OccFlowNet: Occupancy Estimation via Differentiable Rendering and Occupancy Flow**，WACV 2025。[论文](https://openaccess.thecvf.com/content/WACV2025/html/Boeder_OccFlowNet_Occupancy_Estimation_via_Differentiable_Rendering_and_Occupancy_Flow_WACV_2025_paper.html) · [代码](https://github.com/boschresearch/OccFlowNet) | 原代码由相同 instance 的跨帧标注框生成刚体 voxel flow/transform，供训练 renderer 使用；推理 decoder 返回 density/semantics，没有预测 flow 输出。已读 `create_flow_data.py` 和 detector。 | 是 annotation-derived source displacement 的工程参考，也是重要反例：**名称含 flow 不等于模型学会预测 flow**。其 label-derived flow 不能成为我们正式推理输入。 |

### 2026 年最近邻：CRISP

**CRISP: A Spatiotemporal Camera-Radar Backbone for Driving via Forecasting-Based World-Model Pretraining**，arXiv:2607.04541v1，2026-07-05；本次未核实会议接收。它用历史相机+雷达预测未来 LiDAR，保留 ViDAR 的未来 decoder；通过雷达初始化 query、时间门和模态 innovation gate 改 observer。论文 §III–V 明确：200×200×256 BEV、6 sweeps、3 层未来 decoder、24 epochs / 8 A100；未来预测使用同协议给定 ego trajectory。[正文](https://arxiv.org/html/2607.04541v1)

这已覆盖“雷达参加未来预测预训练”的宽泛主张，但没有由所读公式给出完整速度可辨识性保证。我们的机会只能来自更具体、可反驳的**运动监督—传播连接**及相对 O 的证据。作者[项目页](https://umfieldrobotics.github.io/CRISP/)仍写 Code Coming Soon；其所链 `umfieldrobotics/CRISP` 仓库和 GitHub API 本次均返回 404，未核到公开代码/权重，commit/license 记 null，不把网站存在当实现可用。

另两类不宜直接移植：**RadarOcc**（NeurIPS 2024）从原始 range–azimuth–elevation–Doppler 张量提取信息，而我们当前 nuScenes 点表没有同等原始谱；**MetaOcc**（2025 预印本 v2）做当前 occupancy，其作者仓库本次只有 README、LICENSE 和三张图片。两者均不构成可直接加载的本任务未来运动分支。[RadarOcc 论文/代码](https://github.com/Toytiny/RadarOcc) · [MetaOcc 正文](https://arxiv.org/html/2501.15384v2) · [仓库](https://github.com/LucasYang567/MetaOcc)

## 代码可移植性与实际缺口

| 仓库（固定 commit 前缀） | 许可证与入口 | 依赖/成本/适用限制 |
|---|---|---|
| CRT-Fusion `2fa611a5c894` | Apache-2.0；`mmdet3d/models/detectors/crtfusion.py` 的 `fuse_history`、`motion_net`，`dense_heads/crtfusion_head.py:446–489` 的 motion/occupancy loss | MMDetection3D/BEVDepth 系列自定义算子、两阶段训练。直接借鉴小 head 很轻；整个预训练检测器不是 O 的同预算对照。 |
| CRN `5e9d2fa2f91c` | MIT；`layers/backbones/rvt_lss_fpn.py`，`layers/modules/multimodal_deformable_cross_attention.py` | 作者旧 Torch 1.9/MMCV 1.6/Lightning 环境及自定义算子。改 view transform 意味着重算 observer，既有冻结 latent cache 不足。 |
| RaFlow `c01897faeb5d` / CMFlow `16a095a25045` | MIT；`models/raflow_vod.py`、`models/cmflow.py:171–197`、`models/cmflow_t.py:64–107`、`losses/radar_loss.py:100–290` | 点对相关与 PointNet++ 扩展；需真实 per-sweep 时间/pose/对应监督。CMFlow 提供模型权重，适用 VoD 分布，未经 nuScenes 验证。 |
| RaLiFlow `1abb92e93952` | 未核到仓库许可证；作者训练/eval/预处理脚本存在 | 论文 150 epochs、双传感器流网络；不是低成本同模态 baseline。仅依公式独立实现或先确认许可。 |
| TEOcc `7b4b8a022518` | 未核到仓库许可证；`bevdet_occ.py:204–238` 历史重建，`temporal_backbone.py` | `teocc_rc.py` 指向的 `BEVStereo4DOCCRC.forward_train` 仅 current occupancy/depth，且 Collect3D 的 hop GT 被注释；需补清真实 active config，不能把另一类代码存在当默认运行了该机制。 |
| Let Occ Flow `c782571ecc45` | 未核到仓库许可证；`model/neck/temporal_aggregation.py:268–344`、`loss/flow_loss.py` | 需要 2D teacher、相机射线和 3D 表征/渲染依赖；不建议本轮整体移植。 |
| OccFlowNet `27e102e467b7` | AGPL-3.0，另有第三方条款；`tools/create_flow_data.py:136–169`、`occflownet.py:101–140` | 可参考 box pose→source grid 的算法；原 flow 由 GT 生成，借鉴方法与直接复制发行代码须分别处理。 |
| RadarOcc `dbaa20d29909` / MetaOcc `713a9b1c8ff0` | 均 Apache-2.0 | 前者有实现但传感器数据类型不匹配；后者没有模型实现。TARS、CRISP 本次未核到可用作者源码/权重。 |

上述 license 为对应仓库本次 API/文件核查，**null 表示未确认授权，不表示禁止引用论文或算法不能独立实现**。完整 SHA、URL、阅读深度及文件摘要在相邻 sources JSON。工作区原有 CRT-Fusion checkout 有本地修改，因此官方实现结论来自 pinned raw，而非直接把 dirty checkout 归于作者。

三个必须避开的复制陷阱：

1. CRT-Fusion 官方 `fuse_history` 中 `grid[moving_occ_mask][:, 0:2] = ...` 使用链式高级索引；按 PyTorch 语义第一步返回副本，后续赋值不回写原 grid。这是**该公开字节的静态问题**，不是对论文成绩的复现反证。移植须先验证非零速度确实改变采样位置，并验证重复落点累积。[固定源码](https://github.com/mjseong0414/CRT-Fusion/blob/2fa611a5c8944c487e6fe78adbef63b3d0237fec/mmdet3d/models/detectors/crtfusion.py) · [PyTorch 索引说明](https://docs.pytorch.org/docs/main/tensor_view.html)
2. CMFlow `RadialDisplacementLoss` 虽接收 interval 参数，内部固定 `self.interval=0.1`；这适配作者设置，不能覆盖我们各返回真实 lag。相对 Doppler 和 ego-compensated velocity 也不得混用或重复减 ego。[固定 loss](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/losses/radar_loss.py)
3. 由框标签得到的是**实例刚体位移代理**；框外雷达反射、遮挡、肢体运动、转弯旋转和出生/消失并非精确点对应。不要强行赋框外点静态标签，也不要用更换 GT 的方式获得 motion 指标改善。

## 三条可检验路线，先执行第一条

**A：source displacement supervision + connected transport。** 从 train scene 的同一 instance 在 source 和各 horizon 的标注 pose 得到刚体变换，用 source 支持区域构造位移及有效 mask；优先直接监督各 horizon 位移，避免把恒速外推假设当真实 2 秒运动。O 初始化、原 occupancy loss/GT/坐标/ego/action/ROI 不变；预测位移由截至 t0 的观测得到，再用于可微传输当前特征，原 O 输出保留残差通路。先核监督量覆盖及运动可达性，再谈训练。

固定 seed11 下最关键比较是 **O、同头 aux-only、aux+transport**；无训练的 zero-flow 应保持定义的静态/identity 行为，错配 flow 只作干预诊断。另用 GT flow 作明确标记的 oracle 工程阳性上界，不能列为正式推理成绩。若 oracle 都不能改善或不能改变输出，应先排查坐标、支持/空洞与 readout，不继续大规模学流；若学得流误差下降但同协议 GMO 不升，则监督—预测连接假设未获支持。这借鉴 CRT-Fusion / OccFlowNet，**新颖性不能是“给 flow 加监督并 warp”本身**。

**B：保留过去帧的局部关联，再预测完整运动。** 在原本汇聚为均值 BEV 之前，保留点的 timestamp/source sensor、XY 与有效 Doppler，关联相邻过去帧与视觉 BEV；借 CMFlow 的 point correspondence/point-to-ray 及 TARS 的局部场景语义，预测二维运动。光流、轨迹或 LiDAR 仅作 train-time teacher；正式输入仍相机+雷达。相对 A 的可检验增量是：在相同监督/transport/预算下，多时刻关联是否降低**切向误差**，尤其低径向高切向实例；只降低径向残差不足。随机错配帧/flow 可能造成 OOD，不能自动称条件独立负控。

**C（备选）：共享 readout 的 masked-history 状态重建。** 按 TEOcc，用完全位于 t0 以前的相邻状态重建一个已观测历史状态；以实际历史 GT 或明确冻结 teacher 监督，减少只靠未来稀疏/不平衡 occupancy 目标学习动态的困难。需要相同容量、相同训练暴露对照，teacher 坐标必须独立核实。若只提高静态几何或 t0 IoU，不算运动成功。这是辅助表征路线，不与 A/B 一次叠加，且不重复已经失败的“多一槽 memory”实验。

所有路线先按原同协议衡量 **future macro GMO 相对固定 O**，同时报告各 horizon、静态/动态代理分层；运动侧单列 source-instance 位移误差、径向/切向分量与有效支持覆盖。评估目标来自未用于该轮选择的预定数据才可称确认；已经暴露的 full/dev 只能叫回溯验证。单 seed11 不提供训练随机性稳健性结论。当前没有可从论文直接换算的本地 GPU 成本；A 的小头/warp 应先 profile，B 要重算 observer/读取历史，C 的 3D 重建也有额外内存，均未授权本报告自行启动。

## forward splat 与 backward warp：首选及假收益风险

标注派生位移天然是 **source-centric**：每个 source voxel/point 有自己的轨迹。因此首选双线性 **forward splat**，先把 source 中心加位移，再映射到目标 query 网格，显式累计 numerator、mass、coverage。Backward `grid_sample` 需要 destination→source 的逆映射；一般并不能用 `x-u(x)` 代替它，特别是遮挡、碰撞和多对一运动。

但 feature splat 的归一化均值不是 occupancy 概率的物理合并规则；source 静态残差、动态覆盖、空洞和落点冲突必须分开定义，避免当前物体与被搬运物体同时留下“重影”。没有运动监督支持的区域不得默认为可靠静止。正式 inference 的 mask/velocity 均须预测，不能使用 GT foreground mask 选择只搬对的位置。

对 O 还须先核 token→物理坐标契约，不能把网络命名为 future state 就当它一定在 future ego 系。保持 O 原坐标路由，在接口处统一 source→world/reference→target-query 变换；ego 补偿与物体位移各做一次。单点非对称轴、ego 平移、90° yaw、恒速含 lag、零速度/identity、重复落点、越界和 source/target 支持计数是必要 phantom。它们证明实现的几何与梯度，不证明预测真实运动准确。

最容易造成假收益的是：偷用未来 flow/GT mask；把 source flow 当 backward flow；重复 ego 补偿或加错 lag；交换 XY/yx；移动 logits 却使用另一 frame 的标签；丢掉困难 ROI/invalid 而改变分母；把低 Doppler 当静止；只报告检测 mAVE 或可移动类 IoU；把额外 teacher/更深 observer 的收益全归因雷达。这些都应在结果解释之前排除。
