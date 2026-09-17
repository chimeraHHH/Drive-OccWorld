# 世界模型后续调研：可部署三维对应、scene flow 与持久源状态

日期：2026-09-17。目标是从 CVPR/ICCV/ECCV/ICLR/NeurIPS 2024–2026 的新增工作中，筛选能帮助“仅用过去/当前相机+雷达，预测未来空间占据和运动”的几何模块。本文只做文献和官方仓库审计，没有改模型、没有训练。新增筛选为 6 篇；GenFlow3D、ICP-Flow 两篇作机制深读，并额外检查 MemFlow、Chrono 官方仓库的关键实现。

## 先给判断

最有价值的共同线索不是再加一个 flow head，而是把“空间对应是否成立”“运动属于哪个实体”“这次观测是否足以更新状态”显式分开。GenFlow3D 证明了多帧历史可以递推到无目标点云的未来 scene flow；ICP-Flow 证明了多体刚性、对象匹配和可回退的几何初始化在稀疏 LiDAR 上可以低成本工作；MemFlow/Chrono 则说明具有记忆的对应关系可以提高短期运动读出，但二维图像对应不能直接替代 metric 3D 占据地址。

这条线必须与现有 Cpl/Fix 区分。Cpl/Fix 已经把同一刚体状态接入运动和占据，IoU 有小幅收益，但 moving recall 下降，Cpl 相对 CRN-CV 的 moving EPE 只改善约 0.0018 m；现有速度置换对终点几乎没有效应。因此下一步不应再次证明“共享状态能进入两个 head”，而应检验：只有得到观测支持、时间正确、实体对应可信的源状态，才允许它改变未来占据；未支持区域继续由 O/D 基底处理。

## 新增工作筛选

|工作（正式会议）|推理时输入与训练监督边界|具体机制|对本项目的价值与错配|官方实现、权重、部署证据|
|---|---|---|---|---|
|[GenFlow3D: Generative Scene Flow Estimation and Prediction on Point Cloud Sequences，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_GenFlow3D_Generative_Scene_Flow_Estimation_and_Prediction_on_Point_Cloud_ICCV_2025_paper.pdf)|`train_cfg.yaml` 设 `length_pre=3, length_fut=1`；推理用历史 LiDAR 点云/点特征，不给未来点云，未来 flow 的目标只在训练/评价存在。nuScenes/Argoverse 2 的相邻点云 flow 标签由动态实例姿态和 ego 变换生成。没有相机或雷达通道。|递归点集编码器加多尺度相关体；`SetUpdate_rec2_flow` 用 GRU 保存跨时刻状态；解码器对当前 flow 残差做 diffusion refinement。历史估计的 flow 被用于 warp、插值和下一步状态；到未来时用 `pseudo_pc = last_pc + flow_pred0` 递推，而非读取未来点云。|最直接的“表面/点轨迹→未来空间”先例：历史运动被消费在下一时刻的 metric 点地址上，并能处理加速度而不只是一对帧。错配是它预测点云 scene flow，不生成体素占据；动态标签、完整 LiDAR 和点级稠密几何比我们的相机+稀疏雷达更强；diffusion/PointNet2 多尺度成本高。不能把其 EPE 当 GMO 改善。|[官方仓库](https://github.com/ustc-hlli/GenFlow3D)，固定浅克隆 commit `a19e3790beb079cc87c40d890f696dc4e916b8d1`；仓库含 `pretrain/GenFlow3d_nuscenes.pth`，故公开权重为“有”，但本轮未加载或运行。作者给出测试命令，不等于本项目已部署。|
|[ICP-Flow: LiDAR Scene Flow Estimation with ICP，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html)|原任务是两帧附近 LiDAR 的 flow 估计：源帧和待估区间终点帧都作为观测输入；无人工 scene-flow 监督，并另给出用 ICP 伪标签训练实时 FNN 的路线。它不是从 t0 直接预测未来的模型。对本项目只能在两帧均属于 `<=t0` 的历史窗口时调用，不能把论文的终点帧当作部署时未来输入。|先地面分割/聚类，再用直方图找平移初始化，按候选对象对做 ICP，得到每个局部刚体变换，最后恢复点级 flow；ICP 变差时回退到初始化变换。对象匹配和刚性是结构约束，不是后处理的可选 loss。|给出可部署、可解释的对应质量信号：簇内 ICP RMSE、inlier ratio、前后向误差和刚体残差可成为未来地址更新资格的候选特征。它比学习一个无来源的 confidence 更可审计。错配是依赖两帧稠密 LiDAR、地面分割与聚类；当前稀疏雷达通常不够形成稳定簇，不能假称可直接得到每个占据 voxel 的 flow。|[官方仓库](https://github.com/yanconglin/ICP-Flow)，固定 commit `c7b0b83f77db562b66b75eec26208c75a785521a`；代码公开，FNN/结果有 Google Drive 链接但本轮未下载权重。仓库有 demo，未在本项目环境成功部署。|
|[MemFlow: Optical Flow Estimation and Prediction with Memory，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_MemFlow_Optical_Flow_Estimation_and_Prediction_with_Memory_CVPR_2024_paper.html)|历史图像和当前图像；训练用 FlyingThings3D 等光流数据，未来 flow 预测训练只用过去 flow，不能用未来帧作为当前估计输入。|`KeyValueMemoryStore` 保存 working memory；查询特征与历史 key 做相似度读出 value，读出注入 update block。MemFlow-P 可把过去估计 flow forward-warp 作为 future-flow baseline/附加输入，再预测下一步二维 flow。|适合做相机侧“短期对应记忆”的低成本 probe：按时间压缩 memory、观察 memory readout 对 moving/static 的贡献。错配是像素 flow 没有深度、遮挡后的实体身份和体素地址；forward warp 会把错误对应传播，不能直接用于未来 GMO。其历史记忆也不是持久 3D 场。|[官方仓库](https://github.com/DQiaole/MemFlow)，固定 commit `5486fb88a55b04c2e2b292b3abaa7641d9988831`；README 提供多组公开权重文件名/下载入口，本轮未下载。代码和 demo 公开，未在本项目部署。|
|[Chrono: Exploring Temporally-Aware Features for Point Tracking，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Exploring_Temporally-Aware_Features_for_Point_Tracking_CVPR_2025_paper.html)|视频片段和查询点；监督为查询点跨帧位置、遮挡与距离误差。若输入完整片段，未来帧本身会成为特征输入，所以不能直接作为无未来预测器；只能对历史前缀做对应诊断，或作为在线观测 encoder。|DINOv2 中插入 temporal adapter：空间下采样后按每个空间位置沿时间做局部 attention，再零初始化残差恢复原通道。之后以查询特征和每帧 feature grid 的 einsum cost volume 取轨迹，并输出 occlusion/expected distance。|有用的不是 DINOv2 名字，而是“显式遮挡/对应置信度 + 长时特征”接口；可用于判断相机历史点是否持续属于同一表面，再决定是否更新 3D 源状态。错配是二维、无 metric depth，遮挡轨迹不等于可运动物体轨迹；需要完整视频时会越过本项目未来输入边界，不能直接接未来占据 head。|[官方仓库](https://github.com/cvlab-kaist/Chrono)，固定 commit `153b2f2a4ebd4fb9be82ca68e35355c5c05f150b`；README/代码公开，ViT-S/B 权重有 Google Drive 入口，本轮未下载。未在本项目部署。|
|[ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Papais_ForeSight_Multi-View_Streaming_Joint_Object_Detection_and_Trajectory_Forecasting_ICCV_2025_paper.html)|多视角当前图像、历史检测与预测记忆；轨迹监督来自 nuScenes。推理强调 streaming，不读未来图像。|forecast-aware detection transformer 与 joint streaming forecast transformer 共享双向 query memory；历史 forecast 作为检测先验，检测结果再生成 forecast query，作者称不需要显式 tracking association。|提醒我们可让“未来预测帮助当前状态维护”，但它的终端是对象中心轨迹和检测 mAP/EPA，没有表面、体积占据或 metric scene flow；共享 query 仍不证明实体绑定物理正确。可借鉴 memory queue 的时序接口，不能把 trajectory gain 写成 GMO gain。|[项目页](https://foresight-iccv.github.io/) 已核正式 ICCV 2025；[官方仓库](https://github.com/TRAILab/ForeSight) 只有 README，明确写着 `Code coming soon`，无可审计核心实现、公开权重或本地部署证据。|
|[DifFlow3D: Toward Robust Uncertainty-Aware Scene Flow Estimation with Iterative Diffusion-Based Refinement，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_DifFlow3D_Toward_Robust_Uncertainty-Aware_Scene_Flow_Estimation_with_Iterative_Diffusion-Based_CVPR_2024_paper.html)|原任务是两帧点云、法向和 scene-flow 标签之间的估计；第二帧是待估时间区间的观测终点，不是无未来输入的 t0→未来预测。对本项目只能在两帧都处于 `<=t0` 历史窗口时使用；FlyingThings3D/KITTI 训练标签不改变这个输入边界。|扩散式 flow residual refinement，并输出不确定性；用多种 flow-related features 作为条件，应对局部相关错误、重复纹理与噪声。作者强调可作为已有 scene-flow 网络的 plug-in。|可提供“多个 flow 假设/不确定性”作为拒绝更新的诊断，尤其适合测对应 ambiguity；但它仍是瞬时两帧、非持久对象状态，diffusion 采样与旧版 PointNet2 成本较大。论文效果是 EPE，不是未来占据。|[官方仓库](https://github.com/IRMVLab/DifFlow3D)，固定 commit `f93e3e166341249878def599f79456ad3e29cdf3`；README 声称公开 checkpoint，代码公开；本轮未加载、未运行。|

上表的“公开权重”只表示作者仓库/链接可取得；没有把文件下载、权重成功加载和在本项目成功部署混为一谈。六篇均是正式主会论文，ForeSight 的正式性由 ICCV Open Access 页面核验，Chrono 的正式性由 CVPR Open Access 页面核验。

## 两篇深读与四个官方仓库的核心实现事实

### GenFlow3D：递推的是点状态，未来查询不依赖目标点云

论文的关键不是 diffusion 本身，而是顺序输入带来的高阶运动状态。官方 `models.py::GenFlow_rec_seq.forward` 将历史点云拼接编码，然后逐步调用 `GenFlow_unit.decode`；`units.py::GenFlow_unit.decode` 在相邻历史帧之间用 `PointWarping` 和 `UpsampleFlow` 对齐，并将 `state0..3` 传入 `SetUpdate_rec2_flow`。该 GRU 以 `corr` 和已有 flow feature 更新空间状态，不是把历史 flow 当一个全局标量。

到未来步时，代码将 `xyzs` 置空，`DiffusionLayer_rec2_transformer` 只在 `xyz1 + flow_new` 上用当前点特征、上一层 flow 和扩散时间特征估计残差；若还要继续滚动，则显式构造 `pseudo_pc = (pcs[idx_pc-1] + flow_pred0).detach()` 再编码。训练的 `flow_gt` 只作为目标/残差噪声源，未来推理没有目标点云。这个接口对我们有启发：未来占据若要消费运动，应读取同一 metric 源地址递推后的状态；仅并列输出 flow 与 occupancy 不够。但 GenFlow3D 本身是点云序列任务，不能替代相机+雷达的观测支持问题。

但它也给出反证边界。该实现的输入是可采样的 LiDAR 点及其点特征，编码器用 PointNet2 的固定半径、多尺度邻域；代码没有相机投影、雷达稀疏性、体素出生/消失或占据 decoder。把它搬进 O 会同时引入新表示、训练目标和计算开销，不能称为轻量修补。更适合先抽取“递推源地址 + 对应置信度”的诊断接口。

### ICP-Flow：把对应质量变成可解释的几何证据

官方 `utils_match.py::match_pcds` 先把同标签簇作为静态候选，再对未匹配簇构造动态候选；`utils_hist.py::estimate_init_pose` 用三维平移直方图产生多个初始化，`utils_icp.py::apply_icp` 调用 PyTorch3D ICP，并在 ICP 误差不优于初始误差时保留初始化。`utils_track.py::track` 只返回簇对和 4×4 局部变换。也就是说，系统的核心产物不是任意点的自由 flow，而是“哪一簇对应哪一簇、该刚体变换是否通过几何检验”。

这正好补足当前路线缺的中间证据：对于每个候选源实体，可记录 inlier ratio、双向最近邻误差、初始化到 ICP 的改变量、簇大小和跨时间稳定性；只有证据足够时才让运动状态改动未来占据地址。若雷达回波不足以形成簇，就明确标记 unsupported，让 O/D 继续承担补全，而不是把零速度或低置信度误写成静止。

ICP-Flow 不能直接解决本项目：其地面分割、聚类和两帧配准依赖较密 LiDAR，当前 radar 的覆盖率和形状都更弱。最合理的迁移是把它当作一个可回退的对应质量定义，或者在有足够 radar return 的小区域上做诊断；不能声称它能给全体 GMO voxel 提供真实 scene flow。

### MemFlow：二维记忆读出可以做相机证据 probe，但不能做 3D 地址

在固定 commit `5486fb8`，`inference/memory_manager_skflow.py::MemoryManager.match_memory` 将 query/key/value 展平，在 working memory 上做相似度 softmax 读出；`add_memory` 按帧加入 value，达到上限后压缩。`core/Networks/MemFlowNet/MemFlow_P.py::predict_flow` 将历史 key/value 的读出加入 `motion_prompt` 和 update block；预测版本还可拼接 `forward_warp_flow`。

这条路径的可用诊断是：如果历史图像特征只在正确时间/正确像素对应上提高 flow 置信度，而错时/错对象读出明显退化，它可以作为相机侧观测支持。其局限也很清楚：key/value 没有深度或实体 id，forward warp 的误差会被继续写入 memory；这与“未来空间中哪个 voxel 被占据”不是同一个 query 语义。

### Chrono：遮挡和 expected distance 是对应置信度，不是物体运动

固定 commit `153b2f2` 的 `models/locotrack_model.py::TemporalAdapter.forward` 先在空间下采样，再将每个空间位置的特征沿时间排列为 `(B*H*W,T,C)` 做局部 attention，最后以零初始化卷积做 residual。`tracks_from_cost_volume` 用 `einsum('bnc,bthwc->tbnhw')` 建立 query-to-frame 的 cost volume，输出轨迹、遮挡 logit 和 expected distance。

可迁移的最小模块是“对应的置信度和不可见状态”，不是把二维 tracks 当 3D flow。使用时必须只输入历史前缀，并用 metric/radar 几何把高置信 pixel 对应绑定到源点；任何使用未来图像计算当前对应的实验都应排除在部署结论之外。

## 表示比较：表面轨迹、metric 3D flow、图像 flow 与体积状态

|表示|天然查询语义|能直接支持的终端|主要失败方式|适合当前问题的角色|
|---|---|---|---|---|
|表面/点轨迹（GenFlow3D 类）|源点 `p0` 经过递推得到 `p_h`；身份和地址相对明确|moving 点 EPE、点云/表面未来位置|遮挡、点消失、雷达稀疏和非表面虚拟点；不能覆盖新生体积|作为动态实体的 source-address 状态，给占据 head 提供局部、可审计的运动证据|
|局部刚体/metric 3D 变换（ICP-Flow 类）|簇/对象级 `T_j`，点 flow 由变换恢复|刚体对象未来中心、局部表面地址和对应质量|聚类/匹配错误，簇太小，非刚体或遮挡；不能凭空覆盖全场|低成本 correspondence gate、回退规则和反事实错配控制|
|图像 flow/长时 point track（MemFlow/Chrono 类）|像素 query 在图像平面随时间移动，并有遮挡/距离置信度|相机运动证据、可见性和投影一致性|深度歧义、遮挡、物体身份切换；未来帧作为输入会违反预测边界|相机观测支持 probe；需与 radar/深度几何绑定后再影响 3D 状态|
|体积/BEV occupancy state（O 及现有 future head）|未来空间 query 直接读体素/BEV 状态|GMO IoU、FP/FN、未覆盖区域补全|运动可能只作为旁路信号，地址和源实体未显式绑定|保留为主干和完整场覆盖；只接受通过对应 gate 的局部运动修正|

表面轨迹与体积状态不能互相替代：前者有实体归属但会漏掉出生/消失和空体积，后者覆盖完整空间但容易把运动证据平均化。最稳妥的连接是保留体积主干，由 metric 源状态在未来地址产生受限残差，并记录 unsupported 区域的来源标记。

### 覆盖边界：camera query 不能用投影本身冒充三维对应

若只在预测框或雷达回波内改动，确实会留下 CRN 未覆盖区域的主要问题；当前 owner 划分下未覆盖区域约占合法 moving 点的四分之一左右，却贡献了超过一半的 CV 误差。这个约四分之一是预测框覆盖的下游统计，不能与传感器层面约 95% 没有同格雷达回波混为一谈；后者仍是更大的覆盖瓶颈。一个合法的相机增覆盖路线应从当前图像像素或 patch 出发，沿标定 camera ray 结合当前可得的深度分布/深度候选，把它们 lift 到 t0 的 metric BEV/voxel query，再由当前图像的遮挡和匹配置信度决定是否形成源状态。它不能把未来 GT 表面点、未来帧光流或任意框内虚拟材料点投影后当成已观测表面。

目前 O/本候选尚未暴露一个经过验证的“相机 ray + metric depth → source query”接口；因此 camera lift 只能作为待实现的新增 adapter，不能写成现有覆盖已增加。若当前系统没有可用深度候选，先做只读 coverage probe，比较 radar-only、camera-ray×深度候选、camera+radar 交集的合法 metric 候选覆盖率和深度残差；已有 wrong-horizon/object-velocity 的错配结果只作背景，不重复包装为本 probe 的新独立证据。若 camera-only lift 的深度不确定性没有可分辨的真实对应，就把本路线明确限定为局部支持区域，不能宣称解决完整 CRN 未覆盖人口。即使新 query 增加覆盖，也必须让未支持区域保留 O/D 主干，避免用扩大候选数换取虚假动态。

## 推荐方向：观测支持的持久 metric 源状态，而不是再次共享 state

推荐研究问题是：**在历史/当前相机+雷达只支持部分实体和部分表面时，能否用可审计的对应证据选择性更新未来占据，且把未支持区域留给原 O/D 补全？** 高价值点在“更新资格”和“空间地址”同时可证伪，而不是增大 latent 或增加一项 flow loss。

一个最小原型可以这样定义：

1. 对当前预测框/雷达回波形成候选源片段，保留 `p0`、预测速度或局部刚体变换、时间戳、覆盖率和对应置信度。置信度由历史/当前观测的一致性组成，例如双向最近邻误差、簇内刚体残差、相机 track 的遮挡/expected-distance；没有足够回波时置信度为 unsupported。
2. 预测未来源地址 `p_h = T_j(h)p0`，但不把它直接散射成完整占据。未来空间 query 同时看到 `(x-p_h)`、源局部坐标 `(p0)`、动态状态和置信度，只生成 O/D 的局部残差；新生区域和未支持区域仍走原稠密路径。
3. 把对应质量作为可替换控制。已有 wrong-horizon/object-velocity 置换继续作为背景控制，不重新包装成新结论；新增实验只应检验 support metadata、合法 camera query 与 source/destination 地址接口的作用。所有原始材料点和未覆盖人口都保留在分母，不能按支持区域重算一个有利的 EPE。

它与 Cpl/Fix 的差别是可操作的：Cpl/Fix 验证了同一刚体状态接入两个终端，但没有证明状态由哪些观测支持，也没有把 unsupported/错误对应从未来地址更新资格中排除；推荐原型先固定 O/D 与动态状态，只切换 correspondence gate 和 source/destination metric address，才能分辨“状态有用”与“错误状态被阻断”。这也避免把 CRN-CV 已有的对象框速度收益误报成新学习增益。

### 可证伪实验

先做信息可用性筛选，不启动新模型训练，也不假定已有一个可直接调用的 gate 接口：在已有 dev200 和完整 16,074 对象×时域支持上，从当前/历史相机和雷达的合法输入中离线提取候选证据统计，记录可见覆盖、时间差、匹配残差、预测速度和静止/移动分组。先只问真实证据是否比距离、速度、覆盖率匹配的随机/错配证据具有特异性；这一步不要求超过 O 或任何 speed frontier。已有 wrong-horizon/object-velocity 置换是背景和复用控制，不把它们算作本阶段新的独立结论。待筛选支持新增接口后，再实现 source-state、support metadata 和 future-address 的 adapter，训练前向与门控路径并不存在于当前 O 中，不能把“固定组合”写成现成结果。

若信息筛选表明真实对应在匹配支持和静止代价上没有特异优势，则停止依赖该几何证据的训练路线。若只在 GT 可见性/GT 定位或未来帧条件下有效，结论是部署几何仍未解决。若筛选通过，才实现并冻结新增 adapter/gate，在同参数、同数据、同预算 train512 上检验联合效用；训练后的晋级才要求同时降低 stationary 误动、保留 moving recall，并在完整未来 GMO 上优于 O 与全部速度阈值的收益—代价曲线。这个联合门槛属于训练后证据，不是训练前对不存在接口的要求。

晋级需报告：四个 horizon 的 future GMO IoU、moving/static XY/XYZ EPE、moving voxel recall、FP/FN、按覆盖/置信度/距离/速度分层的曲线，以及目标和非目标区域响应。失败条件应预先写明：

- 只提高点/框 EPE，不提高未来 GMO 或移动占据召回：只能称对应改善；
- 只提高 IoU，但 moving recall 和 stationary 代价恶化：只能称占据偏置，不能称运动改善；
- 错对象/错时间与真实对应不可分：停止“实体持久状态”主张；
- 只优于 CRN-CV 或 zero，不超过 O 和八个 D-speed 对照：不进入复杂部署方案。

该方向的成本低于重训 diffusion/视频 DiT：第一阶段是已有输出上的几何和置信度审计；只有出现特异性，才在 train512 加小型 adapter。它仍有明确风险：雷达覆盖不足可能使大量区域长期 unsupported，Cpl/Fix 的负结果也可能意味着地址连接比对应质量更深地受限；因此不存在先验保证，实验可以合理停止。

## 来源与代码核验记录

- 正式会议核验：[GenFlow3D ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Li_GenFlow3D_Generative_Scene_Flow_Estimation_and_Prediction_on_Point_Cloud_ICCV_2025_paper.pdf)、[ICP-Flow CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html)、[MemFlow CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Dong_MemFlow_Optical_Flow_Estimation_and_Prediction_with_Memory_CVPR_2024_paper.html)、[Chrono CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Exploring_Temporally-Aware_Features_for_Point_Tracking_CVPR_2025_paper.html)、[ForeSight ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Papais_ForeSight_Multi-View_Streaming_Joint_Object_Detection_and_Trajectory_Forecasting_ICCV_2025_paper.html)、[DifFlow3D CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_DifFlow3D_Toward_Robust_Uncertainty-Aware_Scene_Flow_Estimation_with_Iterative_Diffusion-Based_CVPR_2024_paper.html)。
- 实际检查的官方源码快照路径和 commit：`/tmp/genflow3d_scout` → GenFlow3D `a19e3790beb079cc87c40d890f696dc4e916b8d1`；`/tmp/icpflow_scout` → ICP-Flow `c7b0b83f77db562b66b75eec26208c75a785521a`；`/tmp/memflow_scout` → MemFlow `5486fb88a55b04c2e2b292b3abaa7641d9988831`；`/tmp/chrono_scout` → Chrono `153b2f2a4ebd4fb9be82ca68e35355c5c05f150b`；另筛查 `/tmp/difflow3d_scout` → DifFlow3D `f93e3e166341249878def599f79456ad3e29cdf3`。上述快照只作源码阅读，未加载权重、未运行训练或评价。
- 权重状态单独核验：`/tmp/genflow3d_scout/pretrain/GenFlow3d_nuscenes.pth` 是实际 7.9 MB 文件，`file`/文件头显示 PyTorch zip 序列化格式，不是几 KB 的 Git-LFS 指针；`/tmp/difflow3d_scout/pretrain_weights/` 中两个文件各约 14 MB，同样是实际 zip 序列化文件。它们是仓库随浅克隆带入的 tracked checkpoint，本轮没有 `torch.load`、没有加载到任何模型，也没有成功部署结果；MemFlow/Chrono/ICP 的权重未下载。
- 代码快照的核心位置：[GenFlow3D `models.py::GenFlow_rec_seq.forward`](https://github.com/ustc-hlli/GenFlow3D/blob/a19e3790beb079cc87c40d890f696dc4e916b8d1/models.py)、[`units.py::GenFlow_unit.decode`](https://github.com/ustc-hlli/GenFlow3D/blob/a19e3790beb079cc87c40d890f696dc4e916b8d1/units.py)、[`layers.py::SetUpdate_rec2_flow`/`DiffusionLayer_rec2_transformer`](https://github.com/ustc-hlli/GenFlow3D/blob/a19e3790beb079cc87c40d890f696dc4e916b8d1/layers.py)；[ICP-Flow `utils_match.py::match_pcds`](https://github.com/yanconglin/ICP-Flow/blob/c7b0b83f77db562b66b75eec26208c75a785521a/utils_match.py)、[`utils_hist.py::estimate_init_pose`](https://github.com/yanconglin/ICP-Flow/blob/c7b0b83f77db562b66b75eec26208c75a785521a/utils_hist.py)、[`utils_icp.py::apply_icp`](https://github.com/yanconglin/ICP-Flow/blob/c7b0b83f77db562b66b75eec26208c75a785521a/utils_icp.py)；[MemFlow `inference/memory_manager_skflow.py::MemoryManager.match_memory`](https://github.com/DQiaole/MemFlow/blob/5486fb88a55b04c2e2b292b3abaa7641d9988831/inference/memory_manager_skflow.py)、[`MemFlow_P.py::predict_flow`](https://github.com/DQiaole/MemFlow/blob/5486fb88a55b04c2e2b292b3abaa7641d9988831/core/Networks/MemFlowNet/MemFlow_P.py)；[Chrono `models/locotrack_model.py::TemporalAdapter.forward`/`tracks_from_cost_volume`](https://github.com/cvlab-kaist/Chrono/blob/153b2f2a4ebd4fb9be82ca68e35355c5c05f150b/models/locotrack_model.py)。
