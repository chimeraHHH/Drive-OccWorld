# M0 改进：多模态、运动连接与蒸馏文献分支

核查日期：2026-09-15。独立阅读原论文与作者仓库，没有使用 skills、连接服务器、安装依赖或训练。下文的外部成绩都是作者在其原任务上的报告，不是本项目实测；候选假设和成功门是研究建议。

**建议保留两个候选：有监督的历史运动对齐；跨时域的占据表征蒸馏。优先解决特征如何承载未来状态，不再在冻结未来特征后追加一个稀疏点分类残差。** 这两项都有强先验工作，单纯移植不足以成为 ICLR 创新；真正待证的是在相同预测任务、相同容量和风险目标下，改变时序对应关系或监督表征能否带来可重复的机制增量。

## 1. 由 R0–R4 限定问题，而非为旧叙事找文献

- R0：点表是同源雷达的不同压缩，不是新增传感器。由补偿速度投影得到的“径向速度”不能包装为独立原始 Doppler。新方法须保留原始时间戳，不能跨越输入截止时刻。
- R2：点分支有梯度，人工阳性对照可以学习，但真实任务没有稳定点增量。因此“梯度接通”不是任务信息被有效利用的证据。
- R3：分层查询保持自然风险时没有挽救 IoU；类平衡可以改善排序，却使默认操作点失配。新增损失、蒸馏区域或辅助任务必须区分**新监督信息**和**重新分配前景权重**。
- R4：B_P0 的新评价 AP 13.805% 高于 M0 12.443%；共同经验预算下 GMO 13.314% 对 13.064%，差值区间跨零；B_P2 不稳定优于 B_P0。P0 已有汇聚雷达，不能称为纯视觉。当前证据不支持继续为点分支加长短训，也不证明所有雷达机制无效。

证据：[R0](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/research_execution_20260914/R0实测结论.md)、[R2](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/r2_execution_20260914/R2结果与推进决策.md)、[R3](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/r3_execution_20260914/R3结果与推进决策.md)、[R4](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/r4_execution_20260914/R4结果与研究决策.md)。这里没有把历史 development 场景重新称作盲测。

## 2. 七项有实际作者代码的核心文献

前六项仓库均读到明确 MIT/Apache-2.0 许可证；第七项有公开源码，但未发现仓库许可证。许可证是代码文件的声明，不自动覆盖数据、权重和依赖。已读模块、源码 SHA256、仓库 commit、依赖与检索缺口保存在独立 JSON。未运行这些仓库，以下不构成复现认证。

| 原论文 | 机制与作者的实验证据 | 原任务及迁移边界 |
|---|---|---|
| **CRT-Fusion，NeurIPS 2024** | 从融合 BEV 预测网格速度与前景，补偿历史对象运动后再聚合。消融中加入 MFE/MGTF：NDS 56.1→57.2、mAP 48.9→50.0；但仅相机模型加入同模块两项均下降 0.5。它提供“运动估计必须足够准”的反证，而不是 warp 必然有效的证明。[论文](https://arxiv.org/html/2411.03013v1) | nuScenes **当前 3D 检测**。框级速度/前景辅助监督有现成来源；不能直接把检测头或其 NDS 当未来 GMO。公开实现与论文模型稍有差异，README 明示权重成绩变化。 |
| **RaCFormer，CVPR 2025** | 雷达深度/RCS 改善视角变换；查询从图像与 BEV 取信息；雷达时序编码。单帧消融：BEV 拼接检测头 35.9 mAP，查询 BEV 40.5，图像+BEV 查询 43.6；同时加入雷达深度和 RCS 为 44.7。[论文](https://arxiv.org/html/2412.12725v1) | nuScenes/VoD **检测**。支持“在几何变换/解码之前融合”的假设；查询数量、头结构、容量都改变，不能把全部收益归因于新雷达信息。移植到密集未来占据需要新的查询/输出契约。 |
| **Doracamom，TCSVT 2026；2025 预印本** | 雷达几何与图像语义初始化体素查询，BEV/voxel 双时序分支，辅助占据与前景监督。v3 表 VIII 的两帧对照：普通 ConvBlock mIoU 20.15，DTE 21.44；四帧两者 21.51/21.81，增益并不随历史长度扩大。[论文 v3](https://arxiv.org/html/2501.15394v3) | OmniHD-Scenes 等 **4D 雷达+相机当前检测/占据**，非未来预测。其高程与更密点云不可由 nuScenes 点凭空恢复。可借鉴任务监督与双域时间对应，不能照搬输入统计。 |
| **RadarOcc，NeurIPS 2024** | 直接处理原始 4D 雷达张量，保留球坐标、抑制旁瓣，再向 Cartesian 聚合。K-Radar 51.2 m 上几何 IoU 30.4；去 Doppler descriptor 为 30.0，mIoU 均为 22.6，不能声称收益主要来自速度。[论文](https://arxiv.org/html/2405.14014v1) | **单帧当前占据**，两种非空语义。恶劣天气主要为定性比较；缺可靠天气 GT 是作者说明的边界。我们的补偿点没有原始 tensor，不能直接移植；作为输入信息与表示保真度的上界参照。 |
| **CRKD，CVPR 2024** | LiDAR-camera 教师到 radar-camera 学生，跨阶段目标性、空间关系和响应蒸馏。43.2→46.7 mAP 中，门控本身已到 44.9，故同门控下 KD 净增 **1.8 pp**；其 mAVE 0.304→0.331 反而变差。[论文](https://arxiv.org/html/2403.19104v1) | nuScenes **检测**。蒸馏可改善语义/几何，不保证运动质量；不能以总 mAP 上升推断预测状态更好。教师框热图须换成与本任务一致的占据/状态监督。 |
| **EFFOcc，2024 预印本、2025 修订** | 轻量 LiDAR-camera 当前占据教师，蒸馏 BEV 与 3D 特征。表 VI 无 KD 32.08，BEV MSE+3D cosine 33.93，BEV MSE+KL 31.63；带 FG/BG 的组合 33.72/33.75，并非最佳。[论文 v2](https://arxiv.org/html/2406.07042v2) | nuScenes/Waymo **当前语义占据与少标签学习**。这支持比较表征目标，不能替前景重权背书，也不证明未来预测可直接受益。不同表有 24/48 epoch、检测预训练等差别，不混算增益。 |
| **LiCROcc，RA-L 2025，2024 在线发表** | nuScenes 补偿雷达点与相机，LiDAR-camera 教师传递多尺度残差、BEV 关系及预测分布。正式版报告 KD 比同 RC-Fusion 基座增加 1.5 mIoU、0.8 IoU；远距与恶劣天气教师优势减弱。[正式论文](https://april.zju.edu.cn/wp-content/papercite-data/pdf/ma2025locro.pdf) | **当前 SSC**；其 xyz/RCS/补偿 vx,vy 与我们的输入最接近。不能把论文“15.5%”相对提升写成 +15.5 pp，也不能把其 17 类 0.2 m 全场景 SSC 与未来 GMO 比较。源码可读但无许可证，不能直接按许可明确的开源模块处理。 |

### 仓库实际检查

| 仓库与冻结 commit 前缀 | 已读真实模块 | 许可证、依赖和可执行性边界 |
|---|---|---|
| [CRT-Fusion](https://github.com/mjseong0414/CRT-Fusion) `2fa611a5c894` | `mmdet3d/models/detectors/crtfusion.py`：`motion_net`、`occupancy_net`、`fuse_history`、`reverse_bilinear`；`necks/view_transformer_crtfusion.py`：`RCA_Attention` | Apache-2.0。安装文档 Torch 1.10.1/CUDA 11.3、MMCV 1.3.16、MMDet 2.14、MMDet3D 0.17.2；自定义 BEV 算子。历史缓存显式 detach，不能不加区别地描述为跨全部历史反传。 |
| [RaCFormer](https://github.com/cxmomo/RaCFormer) `4d4c534f84ac` | `models/racformer_transformer.py`：`BEVSampling`、`RadarBEVTemporalEncoder`、`ConvGRU`；`racformer.py` 有深度监督 | MIT。Torch 2.0/CUDA 11.8、MMCV 1.6、MMDet 2.28.2、MMDet3D 1.0.0rc6，自定义 MSMV CUDA；不宜把整套框架覆盖到现有旧 MMDet 环境。 |
| [Doracamom](https://github.com/TJRadarLab/Doracamom) `d3238c714ee2` | `projects/configs/Doracamom/Doracamom.py` 连接 `RCDetSOC`/`RCDetSOCHead`；`rc_detsoc/modules/occ_temporal_encoder.py` 同时处理 voxel 与 radar BEV | Apache-2.0。Torch 1.9.1/CUDA 11.1、MMCV 1.4、旧 MMDet、Detectron2、torch_scatter、3D deformable attention；数据/标签、几何范围与高程输入需适配。 |
| [RadarOcc](https://github.com/Toytiny/RadarOcc) `dbaa20d29909` | 实际配置在 `projects/configs/baselines/RadarOcc_self.py`，不是 README 示例中的 `projects/baselines/...`；`occupancy/detectors/radarocc_self.py` 接收稀疏 tensor 与球坐标索引 | Apache-2.0。Torch 1.10.1/CUDA 11.3、MMCV 1.4、VoxFormer 3D deformable 算子。作者 2025 修复注明使用 `generate_4d_polar_doppler.py`，不能用 percentile 旧预处理。 |
| [CRKD](https://github.com/Song-Jingyu/CRKD) `c7e6893fd008` | `mmdet3d/models/fusion_models/feature_response_distiller.py`，配置 `configs/nuscenes/distill/feature_response_distill/...yaml` 明确 teacher/student 与 KD 路径 | Apache-2.0。Python 3.8、Torch 1.9–1.10.2、MMCV 1.4、MMDet 2.20、OpenMPI/mpi4py/torchpack、自定义 CUDA。教师权重链接可见，未下载/验证权重文件。 |
| [EFFOcc](https://github.com/synsin0/EFFOcc) `13aeb78c2774` | `mmdet3d/models/detectors/distill_occ2d.py` 与 `configs/effocc_distillocc/fgbg_distill_flashocc-r50.py`；多种 KD 类不能仅凭一个类名推断实际运行路径 | Apache-2.0。有 MMDet3D/BEV-pool CUDA 构建，无完整环境锁。该示例配置指向的 `configs/flashocc/flashocc-fusion-r18_pretrain.py` **当前树不存在**，需选实际 model-zoo 配置重新核对；不能宣称开箱可跑。 |
| [LiCROcc](https://github.com/HR-zju/LiCROcc) `0c1423be2c3f` | `ssc_rs/detectors/ssc_rs.py` 的三种 Distill 开关；`ssc_rs/modules/bev_net.py` 的 `distill_mlp`；`ssc_rs_base_nuscenes_LC2LR123.py` | 未发现 LICENSE，GitHub API license=null。requirements 同列 Torch 1.9.1+cu111、spconv-cu111 及 CUDA 12 包；README 的 Torch URL 缺 h。适合作为只读机制参照，依赖不能照单安装。 |

## 3. 不能遗漏的相邻工作与新颖性压力

[RCBEVDet，CVPR 2024](https://arxiv.org/abs/2403.16440) 的 RCS-aware scatter、双流点编码与 deformable BEV 对齐是应有的结构对照；[作者仓库](https://github.com/VDIGPKU/RCBEVDet) 当前主要以 zip 归档发布，未核对归档内部代码/许可证，不计入上面六项明确许可的代码核验。[RadOcc，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28533) 的 Rad 是 **Rendering**，通过渲染的深度终止分布和语义关系蒸馏；本次未找到可核实作者实现，作为理论动机来源而不是承诺即用的模块。

[RCTDistill，ICCV 2025](https://arxiv.org/html/2509.17712v1) 已把距离-方位不确定性、历史对齐和关系蒸馏结合用于检测；本次未核实作者代码入口。因此“加不确定性掩膜+时序蒸馏”本身已经不是空白。[4DR360，2026-07 预印本](https://arxiv.org/abs/2607.09629) 已提出占据作为持续场景状态以及 Doppler-guided temporal fusion，作者仍称代码/标签接受后发布。不能以“radar occupancy state+temporal fusion”作为未经比较的新颖性主张。它和上述工作仍主要是当前多任务感知；预测未来状态的可辨识性与监督连接才是我们需证明的区别。

## 4. 候选 A：让运动负责历史对应，再联合学习未来状态

**可否定假设 A：** 当前汇聚 BEV 中仍含可利用的时序几何/运动，但监督没有使它成为稳定的对象对应关系；学习并验证这层对应后，未来占据误差应主要在实际运动对象、长时域和历史遮挡区域下降，而不是仅把所有前景分数抬高。

建议沿 M0 初始化，读取每个历史 anchor 的因果相机/雷达特征，在**参考 BEV 构造和时间聚合之前**预测对象占据及位移，结合 ego 位姿把历史支持传入当前状态，再由联合训练的预测模块输出未来占据。真实框速度或 track 对应仅用作训练监督；推理用预测位移。对无对象、无支持和新出现区域保留当前观测路径，不用全图运动 warp 代替静态场景。直接回归/对应学习优先于重新搭建 M3 速度后验支路；不把同源补偿速度当独立测量。

**核心强对照：** 同一 seed11、数据、容量、优化预算与风险目标下，比 M0 原 checkpoint、同预算继续训练的 M0、等容量历史拼接、相同运动辅助监督但不做对齐、完整对齐。先比较“辅助监督”与“对齐”两个效应，再以不改空间/密度的速度扰动或时间对应扰动检验依赖；扰动可能出分布，只能作为机制诊断，不能直接估计因果贡献。少量训练样本可作 GT 对齐阳性上界，但只能用于可行性诊断，绝不能进入正式推理或候选榜单。

**区分与否定：** 若 GT 对齐也不能改善受支持的未来占据，当前表征/标签可能不适合运输，应结束该机制；若 GT 有效、预测对齐无效，问题是可观测对应的估计；若仅辅助监督有效，贡献应写为任务监督而非运动传输。若候选只优于原 M0、不优于同预算继续训练/拼接，不能归功于新机制。所谓“动态”必须用 track 位移/速度定义，不能把 GMO 可移动类别全当实际在运动。

局部代码依据：[参考融合入口](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:363) 当前为 camera BEV 加 radar tokens；[未来预测连接](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:423)。这是工作副本的静态阅读，不替代根 agent 对冻结 M0 实际运行配置的审计。

## 5. 候选 B：蒸馏占据状态及其变化，而不是蒸馏一个分数偏置

**可否定假设 B：** 几何更强的占据教师能提供硬类别以外的空间结构与状态对应目标；把这些目标接到可训练的参考/未来表征，可改善占据边界和长时域状态，而其作用不能被相同前景重权或普通输出 KL 完全解释。

第一步应固定已有 LiDAR-camera **当前占据教师**，先核查其在本任务标签映射、共同区域和可见性条件下确实强于 M0，尤其是本项目的可移动对象；外部 overall mIoU 高不能替代此检查。教师在训练样本的未来时刻可产生未来表征监督，学生只读截至 t0 的因果输入。未来教师是 privileged target，不是可部署预测器，也不能作为公平 forecasting baseline。教师输入未来观测本身不构成预测新信息；它必须只出现在训练目标和明确标识的上界诊断中。

建议学生联合学习：参考场景几何、未来表征，以及相对于参考状态的变化；教师状态在统一坐标和 mask 下比较。保持原任务占据监督，使用固定容量投影对齐特征，先从 BEV MSE+voxel cosine 或局部关系损失开始，不直接堆叠多个注意力与不确定性模块。对于只在未来可见、学生历史没有证据的区域，表征蒸馏可能平均化或复制教师偏差，不能宣称还原了不可观测信息。

**核心强对照：** 同预算 M0 继续训练；相同区域/权重但只有 GT 监督；普通输出 KL；普通 BEV MSE+voxel cosine；所提“状态变化/对应关系”蒸馏。教师/学生来自同一场景、增强一致、坐标一致；教师固定且明确训练集/预训练暴露。训练期教师-only 与 radar-camera 学生推理成本分开记录。若无教师路径与蒸馏路径不能容量匹配，用相同投影参数的辅助监督对照，不把蒸馏额外参数当机制收益。

**区分与否定：** 先确认教师在目标子域有优势；没有则停止该教师。若输出 KL 或相同区域监督解释全部收益，不主张结构蒸馏创新；若特征距离下降但未来占据/边界不改善，目标只是模仿教师而未连接任务；若只改善当前时刻，不写 forecasting 贡献；若雷达置换/无 Doppler 对照不影响收益，结果可以是预测训练方法，但不能写成雷达信息利用突破。

## 6. 两候选共享的决策纪律

建议一次只让一个机制进入实质训练，全部使用单训练 seed11。训练预算依据数据覆盖与收敛诊断设定，不再机械固定为 1200 步；先把普通继续训练 M0 作为强基座，而不是以“冻结特征+点头”限定问题。缓存、教师计算和正式训练都计入总成本。

主判据保持相同完整样本、标签/可见性、预测时域和 ego/action 契约：同时报原始 GMO、分时域与实际运动子域指标、AP/PR 和校准集固定操作点下的实际 FPR；评价不选阈值。建议预登记一个有实际意义的提升目标（例如 GMO +1.0 pp，数值须在看候选结果前统一），并报告配对场景区间。单 seed 可以节约时间，但区间不覆盖训练随机性，不能写多次训练稳定。

若两个机制都不能胜过预算匹配的 M0，正确结论是当前结构/数据下未形成可用方法；保留 M0 和阳性/阴性证据，重新选择表征或任务，不能再用校准收益、模块数量或外部检测成绩代替成功。真正的 ICLR 故事还需要：可被干预否定的机制、可推广的预测学习原则，以及跨主干/数据条件验证；本分支只提供推进假设，不宣告已经具备发表贡献。
