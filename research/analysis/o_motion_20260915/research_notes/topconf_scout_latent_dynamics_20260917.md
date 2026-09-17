# 顶会世界模型：latent dynamics、对象状态与占据解码连接的低成本调研

日期：2026-09-17。范围是 2024–2026 已正式录用的 ICLR/ICML/NeurIPS/CVPR/ECCV/ICCV/AAAI 工作，优先核对官方 proceedings/OpenReview；没有下载权重、没有运行模型、没有启动训练。用户给出的 Drive-HWM、低秩反事实 carrier、DriveCache 已单列在 `three_world_model_papers_20260917.md`，本报告不重复当作新发现。

当前项目的证据锚点保持不变：保留 O/M0 seed11；O 没有认证 physical-flow readout；固定 D motion head + CRN-CV 对移动目标有互补但静止漂移；额外 loss、feature warp、object-velocity 模块尚未产生联合改进；16-anchor GT-motion oracle 只证明存在有限的 occupancy 互补空间，不能作为模型成绩。新的核心问题是：运动从 feature 读出后，是否以正确对象、正确未来地址、正确可见性进入 future occupancy decoder。

## 1. 录用论文筛选表

|论文与身份（primary URL）|表征/动力学机制|监督与推理输入边界|代码/权重现实与可迁移性|
|---|---|---|---|
|[PLSM, NeurIPS 2024 Main](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html)；[正文](https://proceedings.neurips.cc/paper_files/paper/2024/file/43ba0466af2b1ac76aa85d8fbec714e3-Paper-Conference.pdf)|Parsimonious Latent Space Model 通过最小化 latent state 与 action-induced change 的互信息，使同一动作在不同状态下更可预测；用于 future latent prediction、planning、model-free RL。|DMC/Atari 等控制数据与动作；不是驾驶 occupancy，也没有对象绑定或遮挡建模。|官方 proceedings 有论文，未核到与论文绑定的正式代码/weights。可迁移的只是“让运动变化在 latent 中成为可预测、可干预的低复杂度因素”，不能直接声称改善占据。|
|[UnO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.html)；[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf)|连续 4D occupancy field 以 `(x,y,z,t)` 查询，先预测世界场，再用轻量 ray-depth renderer 转成点云；BEV semantic occupancy 作为迁移任务。|输入是历史 LiDAR sweep；训练目标使用未来 LiDAR sweep 的自监督信息。未来 LiDAR 不是推理输入，但方法依赖传感器时序和 LiDAR 几何，不是 camera+radar 直接方案。|Waabi 项目页存在，但未核到公开可复用官方仓库/weights。关键启示是“未来任务查询直接读同一个时空状态场”，以及把 renderer/下游头与 world model 解耦，适合做当前 decoder-consumption 对照。|
|[DriveWorld, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_CVPR_2024_paper.html)；[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_CVPR_2024_paper.pdf)|Memory State-Space Model 分出 Dynamic Memory Bank（预测未来变化）与 Static Scene Propagation（空间静态上下文），再用 Task Prompt 适配检测、跟踪、地图、motion、occupancy、planning。|多相机驾驶视频与 OpenScene 等预训练数据；下游用任务标注/相应传感器监督。动态/静态是 latent 分支，不等于对象级物理流。|官方 CVPR 页面/论文可核验；未核到作者公开的正式代码和完整 weights。与当前最直接的可迁移点是“静态背景保持通路 + 动态残差通路 + 下游任务显式读取”，但仍没有可见性与对象地址合同。|
|[OccWorld, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02024.pdf)；[项目页](https://wzzheng.net/OccWorld/)|用 VQVAE scene tokenizer 把 3D occupancy 压成离散 scene tokens，再以 spatial-temporal GPT 预测未来 scene/ego tokens，直接由 tokenizer decoder 还原 future occupancy。|论文明确区分 OccWorld-O（GT occupancy）、D/T/S（相机+稠密/稀疏/自监督 occupancy）；训练时用 GT scene tokens 做 masked temporal attention，推理时逐步使用预测 tokens。GT occupancy 只能是训练/特权变体，不是相机推理输入。|作者代码入口存在；weights/数据协议需按 O/D/T/S 分别核查，不能混作单一 camera baseline。它证明“future latent 被 occupancy decoder 真实消费”是可操作的，但 token 级表征缺少逐对象绑定。|
|[GaussianWorld, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html)；[官方仓库](https://github.com/zuosc19/GaussianWorld)|显式分解 scene evolution：ego-motion 对齐静态场、dynamic object local movement、newly-observed scene completion；在 3D Gaussian space 中以当前 RGB 观测修正 streaming occupancy。|当前多视角 RGB +历史状态；训练/验证依赖 SurroundOcc occupancy。论文把静态对齐、动态局部运动、新观测补全分开，未要求 GT motion 作为推理输入。|仓库真实存在，HEAD `b43629eaecffd5a7cbaac1a55517766e6263e4fc`，README 给出 GaussianFormer/GaussianWorld 权重入口和 8-GPU、batch 1、20–30 epoch 训练说明；未下载。适合迁移“静态/动态/新可见区域三路分解”，但 25,600 Gaussians 与当前稠密 occupancy 资源合同差异大。|
|[DriveX, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Shi_DriveX_Omni_Scene_Modeling_for_Learning_Generalizable_World_Knowledge_in_ICCV_2025_paper.html)；[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_DriveX_Omni_Scene_Modeling_for_Learning_Generalizable_World_Knowledge_in_ICCV_2025_paper.pdf)|Omni Scene Modeling 联合几何 point-cloud forecasting、2D semantic representation、image generation；解耦 world representation learning 与 latent future decoding；dynamic-aware ray sampling；FSA 用 task queries 聚合未来 latent BEV。|历史多相机视频；点云/语义/图像目标用于预训练或 foundation-model 生成的标签需逐项披露。future latent 由 encoder/decoder 预测，FSA 在下游读取未来序列；没有把 GT future occupancy 当运行输入。|官方 ICCV 论文存在；未找到 DriveX（Chen Shi 等）对应的作者代码/weights，搜索到的 `fudan-zvg/DriveX` 是同名的 free-form trajectory synthesis 项目，不能混用。最接近当前缺口的是 FSA：任务 query 直接读 future latent，而不是只训练 motion probe。|
|[Dyn-O, NeurIPS 2025 Main](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b03b9ec80fc599bb5161746edff0f322-Abstract-Conference.html)；[论文](https://papers.nips.cc/paper_files/paper/2025/file/b03b9ec80fc599bb5161746edff0f322-Paper-Conference.pdf)；[官方仓库](https://github.com/wangzizhao/dyn-O)|对象中心 slot encoder + Mamba/SSM dynamics；可把 slot 分成 static/dynamic 子表示，并用 VQ、重建、对抗 discriminator 与 slot visibility/existence 预测约束 dynamics。未来 slot 由 world model 直接预测并作为下一步输入。|Procgen pixel observations；训练初期用 SAM video masks 引导 slot extraction，按 schedule dropout；动作、reward、termination 来自 Procgen 环境。不是物理驾驶数据，SAM mask 是外部预处理事实。|仓库真实存在，HEAD `69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077`，4-GPU torchrun 的 encoder 训练；README 要 Cosmos tokenizer weights，未见 Dyn-O model zoo/released world-model weights。代码级拆解见第 2 节。|
|[LPWM, ICLR 2026（录用；官方 papers 页已核到，oral 标签本轮不单独断言）](https://iclr.cc/virtual/2026/papers.html)；[OpenReview 正文](https://openreview.net/pdf?id=lTaPtGiUUc)；[arXiv 正文](https://arxiv.org/abs/2603.04553)；[官方仓库](https://github.com/taldatech/lpwm)|Latent Particle World Model：每个 particle 有位置、scale、depth、transparency、appearance；Context 模块从相邻粒子状态推断逐粒子 latent action，Dynamics 模块按 particle 预测下一状态，KL 约束 posterior/prior；支持随机 rollout、动作/语言/图像 goal。|主训练仅用视频重建与 temporal ELBO；动作、语言、goal 是可选条件。paper 明确不需要显式 tracking，particle filtering 延后到 decoder；因此 object permanence 不是外部 GT track 直接提供。|仓库真实存在，HEAD `4cf53c403433e64c01652ac2adbec66231a46dea`；README 提供 Sketchy/BAIR/LangTable/Bridge 等 MEGA checkpoints，但 GitHub `checkpoints/` 只有占位文件，下载链接未在本轮验证。训练支持 single-GPU，也提供 accelerate 多 GPU。代码级拆解见第 2 节。|

### 单列：已检索但不计入上述 8 篇的待核预印本/项目

- [DOME, arXiv 2410.10429](https://arxiv.org/abs/2410.10429) 与 [官方仓库](https://github.com/gusongen/DOME)：occupancy VAE + spatio-temporal diffusion + trajectory resampling，仓库和 checkpoints 入口真实；本轮未核到其正式顶会 proceedings 收录，因此不把旧的 under-review 稿件状态解释为接收或拒稿。它是生成质量/长时域基线，不提供对象绑定机制。
- [DFIT-OccWorld, arXiv 2412.13772](https://arxiv.org/abs/2412.13772)：decoupled dynamic flow + image-assisted photometric consistency；与项目已有 feature warp/flow 负证据高度相邻，暂不推荐作为新方向。
- [COME, arXiv 2506.13260](https://arxiv.org/abs/2506.13260)：scene-centric forecasting control，报告 3s/8s occupancy 改善；本轮未核到顶会正式 proceedings，不作为已发表证据。OccSora 同样只在本轮作为待核预印本/项目身份处理，未据旧稿件状态断言接收或拒稿。

## 2. 两个官方仓库的代码深读

### 2.1 Dyn-O：slot 状态确实进入递推，但绑定仍需另证

核验仓库：`wangzizhao/dyn-O`，固定 HEAD `69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077`。仓库未落地当前工作区；可用固定 commit 的 [world-model 源码（raw）](https://raw.githubusercontent.com/wangzizhao/dyn-O/69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077/dynamics/mamba/mamba_world_model.py)、[配置（raw）](https://raw.githubusercontent.com/wangzizhao/dyn-O/69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077/configs/dynamics/mamba.py) 和 [训练脚本（raw）](https://raw.githubusercontent.com/wangzizhao/dyn-O/69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077/train_dynamics.sh) 抽查。

1. `dynamics/mamba/mamba_world_model.py::MambaWorldModel.compute_prediction_logits`：把每个时刻的 slot（可选 `slots_visible`/`slots_exist`）投影后，与 action token 拼接，经 `pre_ssm_mixer` 做 slot 间混合；随后 reshape 成 `(batch * slot, time, dim)` 进入 Mamba/MHA backbone。`slot_predictor` 输出下一时刻 slot delta/mode，以及 visible/exist、reward、termination。训练与 rollout 的未来输入都会替换成 `next_slots_pred`，所以这是“动态 latent 被递推模型消费”的真实代码路径。
2. `compute_dynamic_loss`：若打开 disentanglement，`proj_static(slots)` 与 `proj_dynamic(slots)` 分支做 slot reconstruction；可选 `dynamic_vq`，并以 `dynamic_to_static_discriminator` 和 temporal contrastive 约束降低静态/动态互信息。默认 `train_dynamics.sh` 将 `DISENTANGLE_STATIC_DYNAMIC=False`、`DYNAMIC_USE_VQ=False`，所以论文扩展并非默认训练路径。
3. `configs/dynamics/mamba.py::Pred`：有 `pred_delta`、`num_slot_modes`、`slot_pred.use_assignment`、`pred_appearing_slots`、`use_slot_visible_and_exists` 等开关。`slot_pred.use_assignment` 默认 sinkhorn，但“slot identity = 真实对象 identity”没有由环境标签保证；它是预测误差下的软/硬 assignment。
4. `train_encoder.sh` 明确使用 `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4`，并先用 SAM mask 训练 encoder；`train_dynamics.sh` 使用 4 GPU、31 slots、Mamba2、200k timesteps、`PRED_ENC_FEAT=True` 和 `PRED_OBS=True`，输入是 Procgen replay/encoder cache。README 还要求下载 Cosmos tokenizer JIT 权重。故训练资源与 supervision 事实都显著不同于 camera+radar occupancy。
5. README 只有 Cosmos tokenizer 的外部下载说明，没有 Dyn-O world-model checkpoint/model zoo；GitHub 页面 Releases 为空。代码真实、released world-model weights 未核验到。

对当前的可迁移启示：应把 visibility/existence 作为状态合同的一部分，并用 `next_slots_pred` 直接驱动未来 decoder；但不能把 Dyn-O 的 slot reconstruction 或 static/dynamic adversarial loss 直接包装为对象绑定已解决。当前项目尤其要验证 slot/source 的 identity、遮挡和新出现目标，而不是只比较 motion EPE。

### 2.2 LPWM：逐粒子 latent action 解决全局动作混叠，但没有驾驶对象监督

核验仓库：`taldatech/lpwm`，固定 HEAD `4cf53c403433e64c01652ac2adbec66231a46dea`。仓库未落地当前工作区；可用固定 commit 的 [`models.py`（raw）](https://raw.githubusercontent.com/taldatech/lpwm/4cf53c403433e64c01652ac2adbec66231a46dea/models.py)、[`modules/modules.py`（raw）](https://raw.githubusercontent.com/taldatech/lpwm/4cf53c403433e64c01652ac2adbec66231a46dea/modules/modules.py)、[`train_lpwm.py`（raw）](https://raw.githubusercontent.com/taldatech/lpwm/4cf53c403433e64c01652ac2adbec66231a46dea/train_lpwm.py) 和 [`configs/traffic.json`（raw）](https://raw.githubusercontent.com/taldatech/lpwm/4cf53c403433e64c01652ac2adbec66231a46dea/configs/traffic.json) 抽查。

1. `models.py::DLP.sample_from_x`：历史图像先编码为 particle attributes；context 从相邻 frame 产生每步 latent action，decoder 先重建观察帧，`dyn_module.sample(...)` 再生成未来 particle sequence，最后以预测 particles 解码未来图像。`use_all_ctx` 可把整段轨迹的 context 作为条件，但普通推理只给前缀帧和可选 action/language/goal。
2. `modules/modules.py::DLPContext`（约 L5406 起）：共享 causal particle transformer，同时设置 posterior/inverse 与 prior/policy context heads；context 可以是 per-particle（`ctx_pool_mode='none'`），也可池化成全局。动作条件若存在可通过 AdaLN 或 action-as-particle 注入；无外部动作时 prior 采样 latent context。
3. `modules/modules.py::DLPDynamics.sample`（约 L6364 起）：先为缺失 context 调 `context_decoder(... encode_prior=True)`，再用 `ParticleFeatureProjection` 编码位置、scale、visibility、depth、features、background，经过 `ParticleSpatioTemporalTransformer`；预测的位置、外观、透明度等属性再写回序列。`ParticleFeatureProjection` 的 input mask 在 `z_obj_on <= 0.2` 时把属性替换为 learned masks，这使 visibility 进入动力学输入，但不是深度/遮挡真值。
4. `train_lpwm.py`：单 GPU DataLoader 默认 `num_workers=4`；配置示例 `traffic.json` 是 128x128、30 particles、20-step horizon、context_dim=7、150 epochs。README 同时给 accelerate 多 GPU 入口。数据集是 Sketchy/BAIR/Bridge/LangTable/OGBench 等，paper 的主要 temporal ELBO 仅用视频；后续 imitation learning 才需要 paired video-action trajectories。
5. README 的 Model Zoo 列出 5 个 MEGA checkpoint 下载链接，说明 released weights 的发布意图真实；但仓库 `checkpoints/PUT_CKPTS_HERE.txt` 只占位，链接内容本轮没有下载或验证。故只能记录“作者提供外部下载入口”，不能报告本地可复现权重已认证。

对当前的可迁移启示：将一个全局 motion latent 改为“逐对象/逐 source 的 latent action”，并让未来 occupancy query 显式读取它，可能比继续增大 motion head 更接近问题本质。限制是 LPWM 的 particle 不依赖 GT tracking，且图像 particle 与 3D occupancy voxel 的地址合同仍需另建；其结果不能直接迁移为 camera+radar 结论。

## 3. 与当前失败证据的对应关系

|当前缺口|文献能支持的机制|不能越过的证据边界|
|---|---|---|
|motion head 能读出运动，但 occupancy decoder 是否使用未知|DriveX 的 FSA、UnO 的时空 field query、OccWorld 的 future-token decoder 都把预测状态作为下游查询/解码输入；这是“状态被使用”的结构性先例。|不能由论文中的 downstream gain 推导当前 O 的 decoder 已消费 D；必须在当前冻结模型上做输出响应/绑定审计。|
|D 对 moving 有互补而 stationary 漂移|GaussianWorld 将静态 ego alignment、动态 local movement、newly-observed completion 拆开；DriveWorld 也拆 dynamic memory 与 static propagation；Dyn-O/LPWM 将 visibility/existence 放入 particle state。|这些分解使用不同表征、数据和传感器；不能把 3D Gaussian 或 SAM mask 当作当前 camera+radar 的可免费输入。|
|GT-motion oracle 有限 occupancy 收益，但 learned transport 没有联合改进|UnO 的“同一 4D field + downstream renderer”、DriveX 的“future latent + FSA query”提示应改变 source→target readout 合同，而不是继续只加 warp/loss；LPWM 提示按对象产生独立 latent action。|已有 16-anchor oracle、partial-oracle、wrong-time、matched controls 已覆盖“正确位移是否有空间”的部分问题，不能把文献启发重复包装成新发现。|
|遮挡/新出现目标可能是瓶颈|GaussianWorld 明确把 newly-observed completion 单列；LPWM 的 transparency/existence mask 和 Dyn-O 的 visible/exist prediction 可作为观测支持变量。|当前 O 的概率不是 dynamic probability，GT 未来框/未来 visibility 不能进入推理；必须保留缺失与未检出惩罚。|

## 4. 两项低成本、可证伪实验

### 实验 A：先核验实际接口，再做冻结参考的局部响应谱

当前 O/M0 没有已认证的 object motion latent 或 visibility head，因此不能直接在 O 上假设存在该干预接口。第一步只读代码和已完成缓存，确认 Cpl/Fix 的实际状态边，或确认固定 D→transport 的真实可调用边；O 只作为 frozen reference，不能被描述成已有 motion-state decoder。只有在接口确实存在时，才对一个 source/object state 做等范数小扰动，并保持原 source support、mask 和稠密输出合同不变，比较：(i) 正常 object/address，(ii) wrong-object 交换，(iii) wrong-time 交换。若接口只在固定 D→transport 存在，则只在该边上做响应谱，不能外推为 O 的 latent readout 证据；若没有可调用边，实验 A 到此停止，不新造“visibility 置零”控制。已有 oracle、temporal、wrong-time 或 matched 控制只作为已完成证据引用，不重新包装。判据是正常地址的目标区响应是否高于等幅错误绑定、非目标泄漏是否受控；它检验 decoder 是否实际读到并按对象使用状态，不是重新测 GT oracle。

### 实验 B：新模型的 decoder-consumed object-state adapter（预算待 profile 锁定）

这不是现成冻结诊断，而是新模型分支：保持 O/M0 主干和 D motion head 冻结，新增一个 residual adapter，使 future occupancy decoder 的每个目标 query 读取 `[source feature, predicted displacement, camera/radar support]`，用 soft assignment 做 source→target cross-attention；静态残差走原 O，动态残差只从该 attention 输出注入。训练预算不能预先指定为 512 updates，须先做 profile 并锁定显存、吞吐、收敛与过拟合边界；若进入训练，train512 只用于拟合/模型选择，dev200 只作预注册后的开发评估；该集合已用于多轮研究决策，不能称独立最终测试集。比较 `adapter + predicted state`、同参数量 `adapter + wrong-object/wrong-time state` 和保持 support 的零动态残差条件；另以等范数随机扰动控制变化幅度；不使用 visibility 置零作为负控制。GT motion 只作训练损失/独立审计，推理只用当前观测和预测状态。除完整 GMO/FP/FN 外，报告目标/非目标响应比、object-level coverage、未检出惩罚，并与已经完成的 8 阈值 speedgate 完整收益—代价曲线比较，不能只与 simple CV 对照。若正确绑定不优于错误绑定，停止该方向；若 adapter 只提升 EPE 而不提升完整 occupancy，说明 task connection 仍未成立。它与已有 featurewarp/objectvelocity 试验的区别是“未来查询地址和对象状态共同参与 decoder readout”，而不是对稠密场做统一搬运。

## 5. 最多两项推荐方向与新颖性风险

### 推荐 1：Object-addressed future latent state → occupancy decoder

用 LPWM 的 per-particle latent action、Dyn-O 的 visibility/existence state、DriveX 的 future-latent FSA 合成一个最小结构：状态包含 source identity/soft assignment、位移、可见性和 modality support；future occupancy query 通过轻量 cross-attention 读取该状态，静态/背景保留 O 的补全通路。科学问题应写成“何种观测支持与对象地址使 motion state 能选择性改善未来 occupancy”，而不是“加入更强 motion module”。

风险：DriveX FSA、OccWorld token decoder、LPWM particle dynamics 和已有 UniAD/OccFlowNet/CRT-Fusion 已覆盖许多组成件；若没有正确绑定优于错误绑定、目标响应具有局部性且完整 GMO 与共同运动指标同时改善，创新性只能停留在工程组合。

### 推荐 2：Support-aware static/dynamic/newly-observed factorization

借鉴 GaussianWorld 的三路分解和 DriveWorld 的 static/dynamic memory，把当前观测支持（camera/radar、可见性、历史可用性）作为 gate 的输入：静态 ego alignment 保持 O，当前证据只更新它实际支持的 source，历史已建立但当前遮挡的动态状态仍递推；新出现区域可进入独立 completion residual。PLSM 的 action-effect regularization 可作为低权重 latent simplicity 约束，但不应成为主贡献。

风险：GaussianWorld 已公开同样的静态/动态/新场景概念；DriveWorld 已有 dynamic memory/static propagation；若 gate 只按速度或 modality confidence 做路由，容易被审稿人判为常规模块。必须展示支持条件、遮挡分层、wrong-binding 和 full occupancy 的因果链，不能只报 EPE 或单个移动类别。

## 6. 结论边界

已发表文献最实质的共同启发是：让动态表示成为未来任务 decoder 的显式输入，并把对象地址、可见性/存在性、静态背景和新观测补全一起写进状态合同。它们没有证明当前 O/D 可直接迁移，也没有消除 sensor-to-voxel support、occlusion 和 identity 的风险。当前最便宜的下一步是先做实验 A，若局部响应和绑定审计通过，再做实验 B；任何 GT-motion oracle、GT occupancy、GT future visibility 仍只能放在特权诊断或训练目标边界内。


根 agent 复核补充：LPWM 的 ICLR 2026 Oral 身份已在 [ICLR 官方 oral 日程](https://iclr.cc/virtual/2026/events/oral) 核到。上述两仓库的核心源码现已按固定 commit 保存到 `topconf_repo_audit_20260917/Dyn-O/` 和 `LPWM/`，摘要在 `root_crosscheck_agent_sources.json`；子 agent 最初阅读时未落地。
