# 超过 O 且改善真实运动：文献与可执行路线审计

2026-09-15。仅原论文、作者页面和作者仓库联网核验；未安装依赖、运行模型或训练，也未修改既有源码/协议。代码逐文件 SHA、固定 commit 和本次未成功的取读均记录在 `literature_motion_sources.json`，摘取的原文件保存在 `motion_source_reads/`。

**建议首选“有真实位移监督、且进入未来状态计算的传输机制”，而不是再给 O 加一个不参与预测的 flow head。** 已有结果支持 O 的未来占据 IoU 优于 M0/C，并且优势主要随未来特征来源改变；FP 减少同时 FN 增加仍不能证明运动更准确。下一步必须同时闭合运动标签、传输计算、独立运动评价三条链。这里的三个路线是可区分的机制假说，不是已验证的新方法。

本次在 `literature_temporal.md`、`2026_literature_update.md`、`新增文献_直接预测与监督尺度.md` 的基础上增加实际 motion/flow 源码核验。persistent2/rolling2 和 reference-frame A 的阴性结果保留；不重新提名双记忆或纯坐标干预。GEM、ForecastOcc、OccProphet 等已读工作仅补充其与新问题的边界，不重复列摘要当作新发现。

## 1. 先区分三个常被混用的“flow”

| 对象 | 准确定义 | 能证明什么、不能证明什么 |
|---|---|---|
| 物理位移/速度 | 同一物体材料点在统一世界/参考坐标中的位置变化；速度另除真实时间差 | 可直接评价运动。由 3D 框刚体变换生成的标签是 **box-derived rigid proxy**，不是精确逐点真值；行人形变、遮挡、框抖动仍有误差。 |
| backward centripetal flow | 时刻 h 的每个前景体素指向 h−1 时刻同实例质心 | 同时编码中心偏移和关联；静止车内部通常也非零。其 EPE 不能直接解释成速度误差。 |
| latent feature flow | 神经网络预测的采样/传输偏移 | 若只受占据损失，它可以承担补偿、重采样和形变；除非有对应监督及独立验证，不能说它就是物理运动。 |

这是实际代码差别：Cam4DOcc `loading_instance.py:198–213` 用 **上一帧实例占据网格质心（round 后）减当前网格坐标**，且 `:224–227` 在缩小四倍的网格上存储。PowerBEV 的定义一致。[Cam4DOcc 原码](https://github.com/haomo-ai/Cam4DOcc/blob/542f14a9d9e142d9faf9044df47c8488a7c3166a/projects/occ_plugin/datasets/pipelines/loading_instance.py#L173)、[PowerBEV 原论文 §3.3](https://www.ijcai.org/proceedings/2023/0120.pdf)。因此不能把现成 Cam4DOcc `flow` 文件改个名字就作为米/秒监督。

## 2. 三条有区分力的路线

### A. 显式位移监督的源状态传输——首选

**文献事实。** EfficientOCF（CVPR 2025）将 BEV 分割、height、backward centripetal flow 分开，以实例关联细化未来占据；其原任务还定义了 C-IoU。[原论文](https://arxiv.org/html/2411.14169v1)、[作者仓库](https://github.com/BIT-XJY/EfficientOCF)。PowerBEV（IJCAI 2023）展示了减少冗余输出任务的可能性，但并不是删任意辅助损失都会受益的普遍结论。其 `utils/instance.py:137–151` 把 flow detach 后做实例 ID warp；EfficientOCF `efficientocf.py:1006–1045` 也在实例后处理里 detach。这两段 **不能直接充当可微未来特征传输**。[PowerBEV 原码](https://github.com/EdwardLeeLPZ/PowerBEV/blob/7cdce93e76f3a0e91fcefaf0ab7bfe33f7c53321/powerbev/utils/instance.py#L137)、[EfficientOCF 原码](https://github.com/BIT-XJY/EfficientOCF/blob/bc2289fafe38d1d6befa88c32cb48e30eb632e5c/projects/occ_plugin/occupancy/detectors/efficientocf.py#L1006)。

更接近计算路径的是 BEVerse：`ResFuturePrediction.forward` 预测 offset，经 `warp_with_flow` 搬运状态，再经 GRU/空间卷积生成未来特征。但默认 `detach_state=True`，其 warp 还原地修改 flow，不能直接照搬到我们完整 BPTT 和 motion-audit 接口。其代码检查到可微 **backward gather**，并非源端 forward splat。[论文](https://arxiv.org/abs/2205.09743)、[原码 :198–227 / :434–449](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/models/motion_modules.py#L198)。

**建议的新计算路径（我们的设计，未实现/未验证）。** 在冻结 O 的当前观测 BEV `z0` 上预测 source-centric 的四个时域位移，并使用显式时间输入；源点运动不是 h 次重复同一个速度。用 nuScenes 同实例跨时刻框位姿构造标签，预测时只读截至 t0 的传感器状态及原协议允许的 ego/action 条件。将 z0 通过可微 forward splat 传到未来查询网格，使用碰撞归一化/支持度，以及独立的未知或新出现区域残差；结果参与 O 未来特征计算。必须保留原 3D decoder、完整 GT、argmax 和 native GMO 评价，不能用 height 填柱或 C-IoU 换掉任务。

用列向量表示，若物体局部点 u 在当前/未来世界中的框变换是 `B0/Bh`，当前 LiDAR→世界为 `G0`，源点 `p0=G0⁻¹B0u`，则物理位移标签可定义为

`d_h^R(p0) = G0⁻¹ Bh B0⁻¹ G0 p0 − p0`。

以上是参考系 R 内的米制位移。给未来 **内部查询网格** 送值时再用 `G_h⁻¹G0(p0+d_h^R)`，不能把 ego 位移误当物体速度。BEV 搬运的 SE(2)/高度近似必须明确；不能把高度压缩特征称为完整刚体 SE(3) 表面运输。框外、身份缺失、出界、框重叠/形变只控制辅助运动监督的有效掩码，原任务 GT/eval mask 一律不变。

**与旧负路线的区别。** 双 memory 改的是可访问状态数量；A 改的是几何采样坐标。这里增加有标签的 **物体相对运动** 并让其改变 value 的空间落点，保留 O 原生主干。重要限制：原 GT 在固定 t0 LiDAR R，而 O 显式 query 路由采用未来局部约定；O 学习特征是否已补偿这种差异未经证明。因此“splat 到未来局部后逐 token 加 O 特征”是需实测的兼容性假说，不是数学上已保证与 GT 对齐，不能借新模块重写 A 的阴性结论。[本项目坐标审计](../m0_improvement_20260915/coordinate_contract_audit.md)。

**确切复用和成本边界。** 可参考 MIT 的 Cam4DOcc `LoadInstanceWithFlow.get_label/generate_flow` 做身份/网格审计，替换其监督语义；PowerBEV `MultiBranchSTconv` 可作为预测不同 horizon 的卷积结构参考，MIT（LICENSE 包含 PowerBEV/FIERY 两段 MIT）。EfficientOCF 是 MIT。BEVerse 根目录/API 未识别独立许可证，本次只借原理，不承诺可直接复制分发。它们的整体 checkpoint 都不能 strict-load 成 O 的新位移模块。首选本地原生小型传输模块，冻结 O 观测编码避免重新提图像；真实运行时间、显存上限需两锚工程测量，不引用作者端到端毫秒作为我们的预算。

**可证伪的首实验。** 先在固定训练锚做无学习几何预检：源实例刚体标签与独立未来框位置相符；只改变 ego 时参考系物体位移为零；静止/平移/旋转、米↔格、碰撞与出界均有已知答案。随后首轮 seed11 对照固定数据、初始化、优化步数和运动头监督：**接入 transport** 与 **auxiliary-only（不把 transport 加入未来特征）** 两个条件。两者运动头本身都训练，不能把“有无运动 loss”混进通路对比。对接入模型再做只读的置零/时间错配 transport 反事实，原预测位移和所有权重保持不变。若接入没有超过 aux-only 和 O 的共同运动/占据指标，或切断通路几乎不影响预测，应否定“运动通过该传输帮助预测”的本轮主张；切断下降本身仍可能有分布失配，不能单独代替训练对照。

### B. 视觉对应监督的运动状态——当框刚体标签不足时

**文献事实。** Let Occ Flow（CoRL 2024，PMLR 2025）从过去/当前图像预测 **当前** 占据和水平二维流，训练用光流与投影一致性，将静态 ego 流从光流中扣出以识别运动。`flow_loss.py:93–107` 明确以两者差和语义 mask 构造 dynamic rays，`:187–224` 把预测流加入空间点后投影比较；不是给未来 BEV 做不区分坐标的 L2。[原论文 §3.1/3.3/3.4](https://arxiv.org/html/2407.07587v2)、[实际 loss](https://github.com/eliliu2233/occ-flow/blob/c782571ecc4556cb5a72c2d4be3ba40a2e9e839b/loss/flow_loss.py#L187)。

2026 的 SelfOccFlow（RA-L）进一步利用邻帧动态特征相似度产生 flow 监督，并通过 flow 变换动态场；仍是当前运动估计，动态语义组还包含停放车辆。其 mAVE 只在 RayIoU true positives 上计算，这提醒我们不能只在候选自己检测对的点上谈“运动变好”。本次未找到可核验的作者公开实现，不承诺复制其模块。[作者 arXiv 接收版](https://arxiv.org/html/2602.23894v1)、[出版 DOI](https://doi.org/10.1109/LRA.2026.3665447)。

**可复用接口与缺口。** Let Occ Flow `loss/flow_loss.py`、`loss/reproj_loss_flow.py` 是可读的投影/动态监督接口；`model/neck/temporal_aggregation.py` 是历史体素融合，不是现成四时域未来 transition。当前仓库未发现 LICENSE，直接代码复用前须另解决许可；可用论文公式独立实现。MIT 的 STCOcc（CVPR 2025）`OccFlowHead` 接 `voxel_feats,pred_voxel_semantic`，用 3D block 和 MLP 输出每 voxel 两通道 flow；但它是当前 flow，且源码 `:59` 无条件 `gumbel_softmax` 引入随机性，不能当作我们 deterministic eval 的可直接替换件。[STCOcc 原论文](https://arxiv.org/html/2504.19749v1)、[作者 head](https://github.com/lzzzzzm/STCOcc/blob/23a4a346a30ab1242ef99a54bac14d4c4c67e08c/mmdet3d/models/stcocc/heads/occ_flow_head.py#L57)。

**我们的设计。** 将可靠的像素/点对应压缩成 train-only 的空间运动约束，指导 O 的未来 transition 或路线 A 的位移；不把未来图像放进测试输入。需要未来图像/标定/pose、对应伪标签、遮挡及有效深度，当前 native z0+GT cache 不包含完整这些资产。现成光流工具的生成成本、域差和模型版本需独立计入；这条路线比 A 重，不建议先重训 TPV/SDF 整网。它不同于双记忆在于监督明确指向同一场景点；不同于坐标 A 在于约束扣掉 ego 后的 object motion。

**证伪条件。** teacher 与独立框运动在静止/转弯/遮挡分层先核方向和量纲；若运动伪标签不能超过零速度/恒速的独立物体位移误差，先否定它作为运动 teacher 的可靠性，不训练大模型。后续即便 2D flow loss 下降，若独立 3D 轨迹误差或遗漏率不改善，只能说拟合了 teacher，不能说预测更准。

### C. 实例轨迹查询驱动占据——结构改造更大，但运动与任务连接最明确

**文献事实。** UniAD（CVPR 2023；作者仓库已有 2.0 更新）将 tracking/map queries 送入 MotionFormer，运动 query 再参与 OccFormer 的未来状态更新。它预测 BEV 实例占据，不能与我们的 3D GMO 数字横比；其当前官方代码是可复核的机制先例，而非本工作的直接 checkpoint。[原论文 §2.2–2.3](https://arxiv.org/pdf/2212.10156)、[作者仓库](https://github.com/OpenDriveLab/UniAD)。

**确切接口。** Apache-2.0，commit `609ee083…`：`motion_head.py:88–137` 接 `bev_embed`、tracking/map 输出及 `gt_fut_traj/mask`；`:383–440` 对多层轨迹监督。`occ_head.py:271–283` 把 `traj_query`、`track_query`、`track_query_pos` 合并，`:215–267` 经每时域 MLP、pixel↔agent attention 更新未来 dense feature 并输出实例 mask。这是 readout 之前真正使用 motion 的路径。[MotionHead](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py#L88)、[OccHead](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L198)。

**我们的设计。** 保留 O 编码器，以可学习 object queries 从 z0 读取当前实例状态，预测四时域位置/朝向及可见性；把轨迹条件化的 query 送到原生 3D 未来 decoder，而不是输出一个断开的辅助框。所需标签是实例身份、当前框/类别、四未来位姿及有效性；可选地图会新增输入条件，首轮不默认加入。原 UniAD `loss/traj_loss.py` 的多模态/匹配代码可读，但 minADE best-of-K、非线性 GT smoother 和 NaN→0 处理不能原样变成我们的单轨迹真实误差指标。[轨迹损失源码](https://github.com/OpenDriveLab/UniAD/blob/609ee083ea51c3521c323f1279dfc4cee0e60467/projects/mmdet3d_plugin/losses/traj_loss.py)。

这是身份持久的稀疏对象状态，区别于把同形状 BEV memory 多放一个槽。代价是检测/关联错误会传到未来；不能以测试时 GT 框/ID 初始化对象，再与原 O 称公平。缺失实例必须计入，而非仅报匹配上的 minADE。源状态缓存仍可复用，但目前没有准备好的、与 O 对齐的 tracking queries，整个 UniAD 权重和训练栈不是即插即用。首个可证伪实验是同一 query/trajectory supervision 下，motion query 接入/切断占据 transition 的配对检验；若只见 box ADE 改善、完整 GMO/实例遗漏不改善，说明任务连接尚未成功。

## 3. 近年论文为何不直接变成第四条主路线

| 工作 | 本次核验后的边界 |
|---|---|
| [GEM，2026](https://arxiv.org/abs/2605.17682) | 显式连续时间 4D Gaussian dynamics 是有意义的结构先例；本次未核实作者可用代码，不能把同名仓库当此模型，也不能把换成 Gaussian 本身当运动证据。 |
| [GaussianWorld，CVPR 2025](https://github.com/zuosc19/GaussianWorld) | 主任务是 streaming current occupancy，接受新时刻观测；不是只给过去图像就输出我们四时域 future GMO。 |
| [ConsistentCity，ICCV 2025](https://benjin.me/publication/iccv25_consistentcity/) | sequential BEV semantic maps→occupancy/video synthesis 的 semantic-flow conditioning 含序列条件；不能把已知未来地图藏成预测输入。作者页 Code 指向 `V2AI/CC`，本次实取 404，尚不可承诺复用。 |
| ForecastOcc / HERMES++（已读 2026 文献） | 将未来观测当训练 teacher 值得研究，但 feature/Gram 对齐不自动等于 motion correspondence。先明确坐标与独立运动评价，不能同时替换感知、监督和输出后声称单一原因。 |
| OccFlowNet，WACV 2025 | 作者实现 `tools/create_flow_data.py` 由框位姿生成 flow 用于时序 rendering，属于监督/几何生成参考，不是可以直接拿来做因果四时域预测的未来 flow checkpoint。[作者仓库](https://github.com/boschresearch/OccFlowNet)。 |

## 4. 收敛后的首轮：先在输出参考空间检验传输上限

根代理选择先做 **固定 t0 LiDAR R 空间的源占据传输**：使用当前 O 占据与真实框运动的 oracle，判断正确物体位移是否能改善未来占据；之后才考虑学习 source displacement 与门控融合。此选择优于首轮直接把 latent splat 加到 O 内部状态，因为它避开尚未证明的 latent 坐标语义，不改变 O 递推、future ego 条件或旧 A 的结论。若以后搬运特征，也只考虑原 readout 对应的 R 索引空间，并声明该空间里的 feature 可运输性仍是假说。

| 首轮选项 | 能更干净地检验什么 | 必须承认的代价 |
|---|---|---|
| 源 occupancy 在 R 中传输 | 真实运动是否能把已有几何送到正确未来位置；物理位移及输出坐标可直接核验 | 当前遗漏/噪声、刚体近似、源 ROI 缺失会限制效果；纯搬运不能创造新出现实例、恢复未来可见表面。概率聚合/碰撞规则还会影响 FP/FN。 |
| 源 latent feature splat | 可携带超出当前前景概率的外观/形状信息，由 decoder 产生新几何 | 更难证明特征的几何对应；融合分布、readout 共适应、高度折叠会与运动效果混杂，首轮代价更高。 |

**oracle 是诊断，不是候选模型或可部署成绩。** 必须把“用 O 当前预测作源”与“用 GT 当前 occupancy 作源”的条件分开：前者检验现有证据可运输性，后者含更强感知 oracle，不能作为新方法优于 O 的证据。固定同一源、同一 splat/融合规则，比较真实框位移、零物体位移（R 系静态复制），并与原 O 四未来输出对照；保持完整原 GT。真实位移无收益会否定“简单运输现有源形状已足够”的假说，未必否定所有带生成残差的运动方法。只有 oracle 有明确空间且覆盖/错配审计通过，才有理由花预算学习位移。

门控融合需防止绕过机制：门把 branch 全关仍可能得到 O，门只抑制前景也可能提高 IoU。因而学习后仍保留 **相同运动监督的 aux-only 控制**，以及正确/置零/时间错配位移的只读输出干预，报告 gate 激活、有效 source 支持、每类 motion EPE、FP/FN 与共同轨迹评价。不能仅凭融合 IoU 增加，就称实际利用了正确运动。这条路线是有物理标签的、可证伪任务连接研究；尚不声明它比现有 flow-guided occupancy 文献更新颖。

## 5. 共同评价与首轮建议边界

首轮只优先 A；B/C 是当身份标签覆盖或对应监督出现明确失败时的备选，不同时堆入三套模块。单训练 seed11，保留冻结 M0/O，既有 S3、双 memory、坐标 A 不复活。实际优化预算、loss 权重、motion threshold 尚未冻结，本文件不伪造工程通过或预注册状态。

必须有两个共同端点：①相同完整原生 GT 上超过 O 的未来四时域 GMO，同时报 t0、FP/FN；②独立来源的运动误差/轨迹指标改善，且不能通过只保留少数 true positives 得到。建议由 GT 位姿定义真正 moving / stationary / turning 分层，阈值在看候选前固定；所有方法共用同一源实例与有效未来定义，报告人数、遗漏和越界，不能把 GMO 语义类别直接当 moving mask。

O 没有独立 flow 输出，不能给新模型直接位移 EPE、给 O 一个随意光流算法后宣称公平击败。应先冻结 **从所有模型最终占据输出读取位置/匹配轨迹的同一评价流程**，报告检测覆盖和 miss-aware 指标；新 motion head 的 EPE 作为单列诊断。若另用 GT 当前实例 ROI 作受控 probe，必须标明条件化且所有模型同用，绝不能把未来 GT 中心作为预测输入或通过只在未来 GT 框内取 centroid 忽略漏预测。

最有辨别力的证据链是：标签对应正确 → 运动预测误差下降 → motion 接入确实改变未来状态 → 相同输出评价下运动与 GMO 同时改善 → 同预算 aux-only 不足以解释收益。即使链条成立，flow transport、object queries 都有清晰先例；ICLR 新意还需回答在何种可观测性、遮挡和监督条件下该连接成立或失效，不能把一次单 seed 提升包装成通用运动学习理论。
