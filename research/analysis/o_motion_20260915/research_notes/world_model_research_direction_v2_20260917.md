# 世界模型调研后的研究方向：从运动旁路转向可验证的未来场景状态

2026-09-17。本轮使用两位 **GPT-5.6-Luna** 子 agent 分别调查状态更新和几何运动，根 agent 阅读未来视觉特征路线、核对关键源码并综合判断。沿用此前 17 篇主会论文的调查，新增筛选 13 篇主会论文，合计覆盖 30 篇；另列 1 篇 TMLR 文章和 4 篇预印本。包含解决邻近感知/跟踪问题的论文，不能把全部 30 篇都叫开环未来世界模型，也不把筛选、全文阅读、核心代码审查和成功复现混为一谈。

新增主会清单：COMBO、PreWorld、World4Drive、PIGDreamer、PERSIST、TAWM；GenFlow3D、ICP-Flow、MemFlow、Chrono、ForeSight、DifFlow3D；DINO-Foresight。TMLR 补充为 JEPA-WMs 系统研究；预印本为 DINO-WM 及用户给出的三篇。正式状态按来源核实，未将 DINO-WM 的历史 ICLR 投稿当作录用。

**结论：最值得投入的是“历史观测如何形成可靠的场景状态，以及这份状态如何产生可被占据任务使用的未来”。当前证据不足以选定 Gaussian、slot、扩散或某种 gate。建议把观测归属与未来状态学习作为竞争假设，先定位主要限制，再训练一个完整的新原型。**

本轮是文献和仓库研究，没有新候选模型训练，也没有产生超过 O 的结果。保留 O/M0、单 seed11、S3 停止和外部 baseline 仅评测官方权重的约束。

## 1. 目前证据真正要求我们解决什么

以下引用的是工作区已完成并审计的结果，不是本轮重新运行：

| 已有事实 | 可以支持的判断 | 不能推出的结论 |
|---|---|---|
| O 在原 full5119 上 future GMO IoU 14.7478%，M0 13.9449% | O 是保留的占据参考 | 不是语义 mIoU；没有 O 原生物理 flow，不能编造 O 的 EPE |
| V 在同一最终权重下交换或置零对象速度，终端指标几乎不变 | 输入提供了速度，但当前终点几乎没有获得实用速度收益 | 不能说雷达速度本身无信息，或结构上绝对没有依赖 |
| Cpl/Fix 已将共享刚体状态连入运动和占据，IoU 同涨约 0.314 pp，学习地址无实质优势，高速召回下降 | 再次接入 warp 或增加容量不足以说明运动机制成功 | 不能把共同增益归因为运动；也不能由这次预算否定所有共享状态方法 |
| 固定 D 补充 CRN 未覆盖区域，移动误差下降、静止误差上升；简单速度路由已形成强对照 | 新方案必须同时考察 moving 收益与 stationary 代价 | 只胜过零运动或 CV 不够；不能事后选择最优 dev 阈值 |
| 虚拟材料点投影的 RAFT 2 s AUC 0.5096，D 自身速度 0.6008 | 当前评分不足以作全局路由；高速窄组有条件线索 | 不是相机没有信息，也没有唯一定位到几何、遮挡或时间模型 |

证据：[V 速度分配](../object_state_velocity_assignment_结果与决策.md)、[Cpl/Fix](../shared_rigid_fixed_dev200_结果与决策.md)、[速度路由及独立复算](../fixed_speed_routing_dev_shared_20260917_independent_results_v1.md)、[RAFT 完整结论](../history_frozen_flow_shared_v2_结果与决策.md)。原调查中“RAFT 仍运行”“速度路由未独立重算”等描述是当时快照，已由这些终点报告替代。

这些结果共同指向三个待区分问题：**观测是否对应真实表面/实体；可用历史信息是否在状态中保留；状态中的运动是否改善未来占据。** 三者不能靠一个总体 IoU 或一个 flow loss 代替。

## 2. 最值得借鉴的机制与边界

| 参考工作 | 对我们的具体作用 | 必须保留的边界 |
|---|---|---|
| [DINO-Foresight，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/efaca9631eb6bd5df8f6761c5d9a9fe3-Abstract-Conference.html)；[代码](https://github.com/Sta8is/DINO-Foresight) | 以冻结视觉特征为预测目标，任务头实际读取预测未来特征；是显式对象运输路线的有力竞争方案 | 2D 未来理解不是 metric 3D 运动；直接复现的训练成本高，不建议先重训整个模型 |
| [PreWorld，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/9c979c7a7bcc6a791c3b492697f97e1e-Paper-Conference.pdf)；[代码](https://github.com/getterupper/PreWorld) | 递归 volume feature 接未来 occupancy 与渲染监督，是与本项目更直接的表示先例 | 递归和未来监督已有先行工作；渲染密度不等于我们二元 GMO 概率 |
| [PERSIST，ICML 2026](https://arxiv.org/abs/2603.03482)；[代码](https://github.com/francelico/PERSIST) | 将三维环境状态、相机与渲染分开，持久状态值得借鉴 | 生成环境的内部持久性不等于真实相机/雷达的观测更新；不建议移植整个 voxel diffusion |
| [GenFlow3D，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_GenFlow3D_Generative_Scene_Flow_Estimation_and_Prediction_on_Point_Cloud_ICCV_2025_paper.pdf)；[代码](https://github.com/ustc-hlli/GenFlow3D) | 点的历史状态、对应与未来位置递推连接明确；可研究源状态在未来地址的持续性 | LiDAR 点云条件比相机+雷达强，不是 occupancy predictor；不直接搬入它的 diffusion |
| [MemFlow，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_MemFlow_Optical_Flow_Estimation_and_Prediction_with_Memory_CVPR_2024_paper.html)；[代码](https://github.com/DQiaole/MemFlow) | 历史对应记忆可以区别于当前一对图像的 RAFT 分数 | 在相同错误虚拟点上换网络不解决三维绑定；只用历史前缀，不能把光流当 3D 速度 |
| [GaussianWorld，CVPR 2025](https://arxiv.org/html/2412.10373)；[代码](https://github.com/zuosc19/GaussianWorld) | 区分状态传播、ego 对齐和新观测更新 | 每步有新图像的 streaming 感知，不是无未来观测的预测；GMO 不能当 opacity |
| [DIO，CVPR 2025](https://research-assets.waabi.ai/DIO/paper.pdf) | 以 source 与时空 query 联合描述占据/运动，为统一查询语义提供先例 | 训练一致性不能证明推理必须消费 flow；本轮没有认证官方实现 |
| [RaCFormer，CVPR 2025](https://arxiv.org/html/2412.12725)；[代码](https://github.com/cxmomo/RaCFormer) | 依据历史时间和速度查询原生视图及 BEV，强调融合地址 | 检测不等于未来占据，采样路径速度 detach；检测覆盖外的人口仍需处理 |
| [JEPA-WMs 系统研究，TMLR 2026](https://arxiv.org/abs/2512.24497)；[代码](https://github.com/facebookresearch/jepa-wms) | 比较表征、条件输入、训练 rollout 和最终任务；提醒我们单步 loss 不能代替长期效用 | 期刊工作，非 ICLR 主会；不同任务的最佳配置不能直接移植成我们的超参 |

新增细节见 [部分观测与状态更新调查](world_model_followup_state_20260917.md)、[几何、scene flow 与跟踪调查](world_model_followup_geometry_20260917.md) 和 [冻结视觉特征调查](world_model_followup_features_20260917.md)。此前对象/粒子线路包括 Dyn-O、LPWM；详细核查见 [第一轮综合调查](world_model_topconf_synthesis_20260917.md)。

根 agent 抽查的两个具体实现事实：GenFlow3D 在未来步骤将目标点云/特征设为空，并在继续预测时构造 `pseudo_pc = source + predicted_flow`；ICP-Flow 在 ICP 配准误差不优于初始化时保留初始化。前者是预测接口，后者是两帧估计的质量检查，不能混称为未来预测器。ForeSight 虽已发表于 ICCV 2025，其 [作者仓库](https://github.com/TRAILab/ForeSight) 本轮仍为 `Code coming soon`，不能承诺直接部署。

## 3. 我建议的主问题与候选方法

**研究问题：部分观测条件下，怎样把可靠的历史运动证据写入持续场景状态，并使这份状态改善完整未来空间预测？**

这比“雷达与图像怎样融合”更具体：同一观测必须有空间位置、时间和来源；同一状态必须在观测缺失时仍有定义；未来解码需要能区分运动带来的变化和补全先验。它也比“添加显式对象状态”更可证伪，因为我们已有相关失败对照。

候选方法分成三个有清晰接口的部分，尚未实现：

1. **观测到状态。** 只用当前/历史相机与雷达形成有几何归属的观测。表面对应、预测深度、雷达径向运动各自保留有效性与来源。没有观测不自动等于物体静止或不存在。当前框内虚拟材料点只是评分坐标，不继续把它们冒充相机表面点。
2. **状态到未来。** 在可用历史内进行观测更新；到预测区间后只做状态转移。实体或空间 query 保留位置、内容及时间信息，关联可以是软的，不能假定一个 slot 就是一个真实物体。运动状态负责相应的空间变化，补全保留独立来源。
3. **未来到任务。** 未来场景状态应是占据 head 实际读取的输入，并用独立物理指标及固定权重干预验证它是否有用。可借鉴 DINO-Foresight 学习具有任务信息的未来特征，但要与几何状态路线比较；不能以低特征 loss 代替未来 GMO 或移动召回。

补充的 [PIGDreamer（ICML 2025）](https://proceedings.mlr.press/v267/huang25ai.html) 提供训练期特权状态与部署期部分观测分工的先例；[TAWM（ICML 2025）](https://proceedings.mlr.press/v267/nhu25a.html) 提醒我们显式处理时间间隔。前者可以启发 teacher/student 状态学习，后者可以作为时序接口；二者都不是本项目尚未验证的运动机制替代品，也不证明某种状态分解是数学必要条件。

这里不是宣布三部分都应叠加。首先比较两个竞争解释：**主要限制是观测归属/持续性，还是未来表征的学习目标/读出？** 如果密集特征预测已经达到相同效果，就没有理由把对象状态的复杂度包装成必需；如果可靠对应的增量只存在于极少数已覆盖对象，也不能宣称解决了全场运动预测。

## 4. 让文献指导实验，而不是继续堆诊断

下一阶段应有两个明确的决策产物，随后才进入一个统一原型；不再无限延伸相似分数的 AUC 检查。

**决策一：观测能否给未覆盖区域增加有效信息？** 在现有输入边界内，先实现并检查合法相机 query 与实体/表面的对应；不能只在 CRN 框或同格雷达范围内工作。可以使用公开模型的历史前缀推理，但预测深度与置信度都不能视为真值。保留所有原始评分人口，缺观测仍记账。若只研究少量可观测源，结论必须明确限于局部。现成 ZNCC/RAFT 弱评分不直接拿去拟合全局 gate。

**决策二：当前表征能否承载我们需要的未来？** 在相同读出与坐标语义下，比较现有未来状态、当前状态的简单外推，以及来自未来观测的 teacher 状态。teacher 仅作特权诊断或训练目标，不能进入正常预测。该接口需实际实现与核验，不能将 O 的当前 BEV 与任意模型的 future embedding 直接替换。若 teacher 本身也无法被同一 head 正确读取，优先修表征/读出；若 teacher 好而预测状态差，才有针对性地学习未来状态。

这两项不是宣称已运行，也不是要求一个未训练的模块先超过 O。它们用于确定新原型需要解决什么。已有 GT 位移 oracle、wrong-horizon、速度置换直接作为先验事实复用；新工作必须增加 **合法源支持或未来状态信息**，不能把旧控制换名字重做。

**原型学习与晋级。** 冻结已认证基底，从能检验上述选择的原型开始，先测资源与拟合能力，再锁定正式预算；保持 seed11。必要竞争对照应具有相同输入与相近容量，区分历史状态/未来特征学习和一般容量收益。最终固定权重再测试匹配支持的错关联、错时间以及无更新，观察目标和非目标区域的终端变化。真实驾驶数据没有对应的真实反事实未来，因此这些是模型机制干预，不称反事实因果真值评测。

正式晋级同时要求：原协议未来 GMO 改善；移动占据召回有实用收益；原始材料点支持上的 moving EPE 改善且 stationary 代价受控；新运动策略不被现有八阈值速度策略的收益—代价边界支配。不是要求每项指标都逐一胜过每个阈值端点。dev200 已曝光，下一稳定候选还需在未参与设计的 scene 集上验证；先查使用历史再锁定，不事后称 sealed。单 seed 的限制应如实报告，scene bootstrap 不能代替训练随机性。

新原型的 metric 运动读出必须单独实现并认证单位、坐标、时刻和源点归属。只有 latent/占据输出的 adapter 不能直接拥有 EPE；若继续用冻结 D/CV 评分，那仍是 D/CV 的物理结果，不是新 adapter 已改善运动的证明。

如果联合收益仍不成立，而历史状态在受控传感器缺失下稳定优于 O 和简单回退，再以证据决定是否转向退化可靠性。现在不预写全天候泛化结论。

## 5. 与三篇用户提供文章的关系

- [Drive-HWM](https://arxiv.org/abs/2609.03572)：与“预测动态表征，再由下游任务消费”相关；光流监督、FiLM 和规划提升不自动证明三维物体运动更准确。
- [Low-Rank Dynamics-Effective Latent Carriers](https://arxiv.org/abs/2608.15156)：最有用的是固定权重、正确对象与错误对象等特异性控制；其精确配对反事实环境在我们数据里不存在，不能直接做一个差分 SVD 就宣称因果载体。
- [DriveCache](https://arxiv.org/abs/2608.16354)：强调局部计算改动对终端输出的影响，适合借鉴评价设计；缓存扩散去噪步不是我们的未来物理时域，当前优先级较低。

三篇仍按预印本单列。更广泛的正式论文与代码告诉我们：**模块组合已经很多，真正值得研究的是信息在哪个环节丢失、什么状态足以解决任务，以及运动收益为何没有转化为未来占据收益。** 当前报告给出可检验路线，不给未经实验支持的性能承诺。
