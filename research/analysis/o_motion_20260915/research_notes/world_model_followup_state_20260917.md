# 世界模型后续调研：部分观测、持久状态与观测更新解耦

日期：2026-09-17。范围限定为 ICLR/NeurIPS/ICML/ICCV/ECCV 2024–2026，外加必要的作者代码核验；本报告不重复已有 17 篇 synthesis，而补充六篇能改变当前路线判断的工作。没有启动训练，也没有修改模型。

## 先锁定当前问题

已有 O 的 future GMO IoU 为 14.7478%，没有认证的物理 flow readout。固定 D 运动 head 对 CRN 未覆盖点提高 moving、却伤害 stationary；V 的速度置换大幅改变速度输入而 future GMO 几乎不变（P−N 约 `2.1e-5` pp，物理 EPE 差约 `3e-9 m`），Cpl/Fix 已把学习位姿修正接入 occupancy 采样地址，但仍没有可靠 joint-motion 收益。因此本次新增文献只在能回答“观测怎样写入状态、状态怎样被动力学使用、终端占据是否依赖该状态”时保留。GT 框虚拟材料点不能当作相机可见表面；光流 probe 也不能当作 occupancy 已使用的证据。

## 新增文献总览

|工作|正式状态与主问题|对本项目的增量|代码核验|
|---|---|---|---|
|COMBO|ICLR 2025；部分 egocentric 观察先重建整体状态，再做可组合动力学|观测更新与动力学 rollout 可以分开；但其状态是 inpainted top-down 图，不是可验证 3D 对象状态|官方仓库 commit `17a07ce5122b54737d8c0e3694a13d7140c93967`，深读|
|PreWorld|ICLR 2025；多视角 4D occupancy 预测|最接近 occupancy：递归 volume feature + 未来 occupancy/rendering 监督；没有 age、visibility、实体关联变量|官方仓库 commit `0b0e0215263c63192fa928587ac9a79258b3fb36`，读核心文件|
|World4Drive|ICCV 2025；意图条件的未来 latent 用于规划|反例：没有显式对象槽也能让未来 latent 影响任务读出；但没有证明 latent 改变几何 occupancy|官方仓库 commit `cffb51adeb1f7d02b49c4b74d7262ded62a33ac8`，读核心文件|
|PIGDreamer|ICML 2025；部分可观测安全 RL 的 privileged training|提供部署合法信息与训练 privileged 信息的明确分工；提示 belief latent 可以代替显式对象槽|官方仓库 commit `7c1d4fe21205667170ad44564b8c12ee85a14e52`，读核心文件|
|PERSIST|ICML 2026（arXiv 页面标注 accepted，ICML 2026 下载页收录）；持久 3D latent state|最直接的持久 3D 状态反例/正例：环境、相机、renderer 解耦；但推理没有新外部观测更新|官方仓库 commit `b37a211667b9c2fb27174790f517cc7019cf103b`，深读|
|TAWM|ICML 2025；显式 `Δt` 的自适应动力学|把 nuScenes 的未来时域和观测间隔纳入 transition；不解决实体绑定|官方仓库 commit `ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c`，读核心文件|

## 逐篇记录

### 1. COMBO：先修正部分观测，再做 compositional dynamics

**论文事实。** 输入是 2–4 个 agent 的 egocentric RGB-D、相机矩阵和历史图像；训练监督是 TDW-Game 约 107k、TDW-Cook 约 50k 视频及合成世界状态/动作条件。推理时可用当前及历史 ego 视图、相机矩阵和 agent 动作，先形成 top-down inpainted state，再 rollout；不读未来真实图像。报告的是合成视频正确率、合作成功率和步数，不是 nuScenes occupancy。论文明确指出四动作条件下仍有约 25% 的 next-state 预测失败，且环境是小规模合成 embodied benchmark。[ICLR 2025 正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7d03c6bf9f07acb4038eea96c63db52d-Abstract-Conference.html)；[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/7d03c6bf9f07acb4038eea96c63db52d-Paper-Conference.pdf)；[作者项目/仓库](https://github.com/UMass-Foundation-Model/COMBO)。

**代码实读。** 在固定 commit `17a07ce5122b54737d8c0e3694a13d7140c93967`，`tdw_maco/agents/combo_agent.py` 的 `act()`（约 144–169 行）将多个 ego RGB-D 通过 `get_ego_topdown` 和 `get_overlay_ego_topdown` 形成 overlay，再把它送入 `CWM.run()` 的 inpainting 分支；同一 `inpainting_top_down` 随后成为 proposer、ranker、intent history 和 rollout 的当前状态。[act() 源码](https://github.com/UMass-Foundation-Model/COMBO/blob/17a07ce5122b54737d8c0e3694a13d7140c93967/tdw_maco/agents/combo_agent.py)。`tdw_maco/combo/cwm.py` 的 `sample()`（约 159–184 行）在无动作文本时调用 `inpainting_model.sample(x_cond)`，有动作时调用 compositional diffusion `self.model.sample(x_cond, text_embed, mask, comp_mask)`；代码没有对象 ID、观测 age 或传感器支持 mask。[sample() 源码](https://github.com/UMass-Foundation-Model/COMBO/blob/17a07ce5122b54737d8c0e3694a13d7140c93967/tdw_maco/combo/cwm.py)。

**迁移判断（推断）。** 可迁移的机制不是“把 flow 接进 decoder”，而是增加一个显式的 **evidence update**：当前合法相机/雷达只改写实际支持的空间区域，同时输出 support、age、coverage 和 association confidence；未被观测区域沿用预测状态。随后 dynamics 只作用于该状态，occupancy query 从未来状态读出。COMBO 的 inpainting 可作为“观测不足时先形成状态”的概念先例，但它生成的 top-down 先验没有物理可见性证据，不能直接当作本项目的传感器更新或 3D 持久对象。

### 2. PreWorld：occupancy 中已有递归状态，但缺少观测更新语义

**论文事实。** 输入是多视角图像、相机/射线、时序信息和 ego 状态；预训练监督为深度、语义、RGB 的 2D volume rendering，微调加入 3D occupancy focal/Lovász/scene-class-affinity 类损失。推理时用当前多视角图像和 ego/camera 状态初始化 volume，未来帧沿递归 volume feature 预测，不读未来真实图像。论文在 Occ3D-nuScenes 上报告 4D forecast 平均 IoU 9.55（无预训练 9.06）和 3D mIoU 34.69（无预训练 33.95），训练配置为 8 张 A100。它是未来 occupancy 相关工作，但不是 object-centric persistence 工作。[ICLR 2025 正式论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/9c979c7a7bcc6a791c3b492697f97e1e-Paper-Conference.pdf)；[作者仓库](https://github.com/getterupper/PreWorld)。

**代码实读。** 固定 commit `0b0e0215263c63192fa928587ac9a79258b3fb36` 的 `PreWorld4DTraj.forward_train()` 先用 `final_conv` 得到 `voxel_feats`，再把 ego state 经 `plan_head` 广播到 volume，用 `fusion_head` 得到 residual fused feature；`occupancy_head`、`density_mlp`、`semantic_mlp` 在 fused feature 上输出未来占据和渲染量，循环末尾以 `voxel_feats = fused_voxel_feats.clone()` 递归。对应的 3D 与 temporal rendering loss 在同一个函数内计算。[核心文件](https://github.com/getterupper/PreWorld/blob/0b0e0215263c63192fa928587ac9a79258b3fb36/mmdet3d/models/detectors/preworld_temporal_traj.py)。

**迁移判断（推断）。** PreWorld 说明“递归隐藏空间 + 未来 occupancy supervision”是值得比较的任务接口，但它的递归变量没有观测年龄、可见性、实体绑定或不确定性。对 dropple 最有价值的是在已有 Cpl/Fix 的地址连接之上，增加可审计的观测更新、support/age 和绑定干预，而不是假定再加一个 shared rigidstate 就能解决问题。若正确/错对象交换对终端 occupancy 无差别，说明增加递归容量仍未解决因果地址问题。

### 3. World4Drive：不需要显式对象槽的未来 latent 读出反例

**论文事实。** 输入是多视角图像、深度、ego 历史信息和驾驶意图；训练监督包括 ego 未来轨迹、语义标签以及实际未来观察 latent 的 reconstruction/KL/cosine 对齐。推理时只用当前/历史合法视图、深度、ego 信息和候选 waypoint，实际未来图像只在训练时作为 target。论文是 ICCV 2025，报告 nuScenes/NavSim 的规划 L2、碰撞率和收敛速度，任务是 end-to-end planning，不是 3D occupancy。[ICCV 2025 正式页面](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html)；[论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.pdf)；[作者仓库](https://github.com/ucaszyp/World4Drive)。

**代码实读。** 固定 commit `cffb51adeb1f7d02b49c4b74d7262ded62a33ac8` 的 `waypoint_query_decoder_simple.py` 中，`forward()` 用 `prev_view_feat` 经 `_temp_decoder` 形成时序 view feature，并在 368 行 `detach()` 后持久到下一次；`wm_prediction()`（约 543–552 行）把 waypoint 编码成 `wp_token`，与各视角 image tokens 拼接，经 view-specific transformer decoder 得到 `wm_next_latent`。[核心源码](https://github.com/ucaszyp/World4Drive/blob/cffb51adeb1f7d02b49c4b74d7262ded62a33ac8/projects/mmdet3d_plugin/W4D/dense_heads/waypoint_query_decoder_simple.py)。`W4D.forward_pts_train()`（约 203–331 行）对每个 modality 计算未来 latent 的 reconstruction/KL/cosine loss，再用 `select_optimal_modality` 按轨迹匹配和 latent loss 选模态。[训练源码](https://github.com/ucaszyp/World4Drive/blob/cffb51adeb1f7d02b49c4b74d7262ded62a33ac8/projects/mmdet3d_plugin/W4D/W4D.py)。

**迁移判断（推断）。** 这是“显式对象槽不是必要条件”的实证反例：时序 latent 和任务 query 可以工作。但它也直接警告我们，future feature 对齐并不等于 future occupancy 被运动改变；当前 V 置换已经显示“速度输入变了”不足以改变 GMO。因此若借鉴，只应借鉴 **状态→终端 query 的可干预接口**，并加入 zero/swap state 后 occupancy 必须特异响应的检查。

### 4. PIGDreamer：privileged training 与合法部署 belief 的分工

**论文事实。** PIGDreamer 针对部分可观测安全 RL，训练中同时使用 partial observation 和 privileged underlying state。其 ACPOMDP 理论、privileged representation alignment、asymmetric actor-critic 共同约束：训练期可以让 predictor/critic 使用 privileged 信息，部署 actor 只能看 partial observation。总损失包括 dynamics、alignment、decoder 和 reward/cost predictor；论文还报告去掉 representation alignment 后收益只剩边际，说明 privileged 信息不是自动有效。正式状态为 ICML 2025。[PMLR 正式页面](https://proceedings.mlr.press/v267/huang25ai.html)；[作者仓库](https://github.com/hggforget/PIGDreamer)。

**代码实读。** 在官方仓库固定 commit `7c1d4fe21205667170ad44564b8c12ee85a14e52`，`AsymWorldModel.loss()` 编码 observation 和 privileged info，调用 `rssm.observe_combined`，并分别计算 dynamics/representation 与 privileged decoder、reward/cost predictor；`AsymPrivilegedWorldModel.imagine()` 将 combined state 分成 naive/privileged，经过 `enhance_func` 后才送入 policy，再在 imagined step 中重复该过程。[loss 源码](https://raw.githubusercontent.com/hggforget/PIGDreamer/7c1d4fe21205667170ad44564b8c12ee85a14e52/AsymDreamer/models/world_model/asym_world_model.py)；[imagine 源码](https://raw.githubusercontent.com/hggforget/PIGDreamer/7c1d4fe21205667170ad44564b8c12ee85a14e52/AsymDreamer/models/world_model/asym_privileged_world_model.py)；[DualRSSM 源码](https://raw.githubusercontent.com/hggforget/PIGDreamer/7c1d4fe21205667170ad44564b8c12ee85a14e52/AsymDreamer/models/utils/rssms.py)。

**迁移判断（推断）。** 可把 GT box/速度只用于训练期上界或 alignment probe，而不把它们放进部署 query。对本项目，privileged 条件应改为“正确实体/时间的状态绑定”上界，部署端只输入合法 camera+radar 证据，并显式测 partial-to-belief 的退化。PIGDreamer 支持“recurrent belief latent 足够”的路线，但其 RL 任务不能证明 nuScenes occupancy 受益。

### 5. PERSIST：持久 3D state 的强正例，同时暴露观测更新缺口

**论文事实。** PERSIST 的 arXiv 页面标注“Accepted to ICML 2026”，ICML 2026 官方 Downloads 也收录 *Beyond Pixel Histories: World Models with Persistent 3D State*；论文把环境、相机和 renderer 建模为演化的 latent 3D scene，报告空间记忆、3D consistency 与长时稳定性改善。[论文/正式状态](https://arxiv.org/abs/2603.03482)；[ICML 2026 官方下载页](https://icml.cc/Downloads/2026)；[作者仓库](https://github.com/francelico/PERSIST)。

**代码实读。** 固定 commit `b37a211667b9c2fb27174790f517cc7019cf103b` 的 `BasePipeline.from_pretrained()` 要求独立的 voxel encoder/decoder、pixel VAE、camera model、voxel/pixel denoiser 和 voxel class embedder；`VoxelFirstPipeline.__call__()`（约 601–778 行）在每个时刻先 diffuse next voxel frame，再可选 re-encode，随后预测 camera，最后用 voxel/camera/action 条件生成 pixel frame；`warm_start()` 以初始 pixel 生成 t0 voxel，之后只沿 rollout 递归。[pipeline 源码](https://github.com/francelico/PERSIST/blob/b37a211667b9c2fb27174790f517cc7019cf103b/pipelines/pipeline.py)。`scripts/run_inference.py` 明确初始 context 是 pixel、camera、可选 voxel 和 action sequence，并提供 `use_camera_gt`、`include_initial_voxel_frame`；没有当前外部传感器逐时刻写回 state 的接口。[推理脚本](https://github.com/francelico/PERSIST/blob/b37a211667b9c2fb27174790f517cc7019cf103b/scripts/run_inference.py)。

**迁移判断（推断）。** PERSIST 给出应拆开的变量：scene state、camera/ego state、observation renderer。它支持在 dropple 中把 occupancy state 与 ego/camera 以及观测证据分开设计，但不能照搬其 voxel diffusion：其 Minecraft/Craftium 类场景、单初始 context 和高成本生成与 camera+radar nuScenes 不同。更关键的是，PERSIST 的“持久”是内部 rollout memory，不等于真实观测更新；若本项目加入合法 observation update 后，正确绑定仍不能选择性改变目标 occupancy，就应否定“显式持久状态本身足够”的说法。

### 6. TAWM：把时间间隔从隐含常数变成 transition 输入

**论文事实。** TAWM 在 transition、policy 和 value 计算中显式条件化 `Δt`，训练监督来自不同 observation rate 下的状态转移、奖励和控制任务回报，并在多种时间步采样训练。推理时输入当前 observation、action 和实际 `Δt`，由 `model.next/reward/pi` 使用同一时间步；不提供对象关联或持久场景观测。PMLR 页面称在相同样本数和迭代数下，对不同 observation rate 的控制任务优于固定步长模型。正式状态为 ICML 2025。[PMLR 正式页面](https://proceedings.mlr.press/v267/nhu25a.html)；[作者仓库](https://github.com/anh-nn01/Time-Aware-World-Model)。

**代码实读。** 固定 commit `ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c` 的 `tawm/tdmpc2.py` 中 `act()` 将 `timestep` 转成 `dt`，同时传给 `encode/plan/pi`；`_estimate_value()` 在 imagined rollout 中把同一个 `timestep` 传给 `model.reward` 与 `model.next`。`online_trainer.py` 的 adaptive time-stepping 分支随机采样环境 `dt`、调整 episode steps 并 reset 环境。[TDMPC 代码](https://github.com/anh-nn01/Time-Aware-World-Model/blob/ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c/tawm/tdmpc2.py)；[训练代码](https://github.com/anh-nn01/Time-Aware-World-Model/blob/ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c/tawm/trainer/online_trainer.py)。

**迁移判断（推断）。** nuScenes 的 0.5/1/1.5/2 s 未来时域和相机/雷达间隔都使 `Δt` 成为低成本可测变量；但 TAWM 只校准时间尺度，不解决源状态是谁、是否可见和 occupancy 地址是否被改写。可作为 state transition 的一个条件量，不能单独替代观测更新实验。

## 本次实读源码快照与行号

以下路径是本次实际读取的快照；仓库短哈希均先用本地 `git rev-parse HEAD` 核对，PERSIST 另用 `git ls-remote` 核对 `main`。PERSIST 的固定 commit 是完整 40 字符 `b37a211667b9c2fb27174790f517cc7019cf103b`；源码链接也使用这一哈希。

|论文|本地快照|可复查主张与行号|
|---|---|---|
|COMBO|`/tmp/combo_wm_scout.O3ES65/tdw_maco/agents/combo_agent.py`；`/tmp/combo_wm_scout.O3ES65/tdw_maco/combo/cwm.py`|`act()` 144–169；`CWM.sample()` 159–184|
|PreWorld|`/tmp/preworld_wm_scout.hE0MSI/mmdet3d/models/detectors/preworld_temporal_traj.py`|`PreWorld4DTraj.forward_train()` 372–530；未来递归 451–528|
|World4Drive|`/tmp/world4drive_wm_scout.banYGE/projects/mmdet3d_plugin/W4D/W4D.py`；`.../dense_heads/waypoint_query_decoder_simple.py`|`forward_pts_train()` 203–331；`prev_view_feat` 361–369；`wm_prediction()` 543–552|
|PIGDreamer|`/tmp/pigdreamer_wm_scout/asym_world_model.py`；`/tmp/pigdreamer_wm_scout/asym_privileged_world_model.py`；`/tmp/pigdreamer_wm_scout/rssms.py`|`loss()` 86–136；`imagine()` 16–42；`DualRSSM.obs_combined_step()` 273–300|
|PERSIST|`/tmp/persist_wm_scout.hSjlFZ/pipelines/pipeline.py`；`/tmp/persist_wm_scout.hSjlFZ/scripts/run_inference.py`|`from_pretrained()` 92–180；`VoxelFirstPipeline.__call__()` 601–778；`warm_start()` 781–848；初始 context 参数 130–143|
|TAWM|`/tmp/tawm_wm_scout.zENNZa/tawm/tdmpc2.py`；`/tmp/tawm_wm_scout.zENNZa/tawm/trainer/online_trainer.py`|`act()` 90–112；`_estimate_value()` 126–153；adaptive `dt` 254–290|

PIGDreamer 的三份文件是从其官方固定 commit 的 raw 源码保存，其余快照来自对应仓库 shallow clone；这些文件只用于代码阅读，没有运行训练或修改模型。

## 统一判断：如何超越失败的 sharedstate

论文证据共同允许两种路线：PIGDreamer 和 World4Drive 反驳“必须持久显式 object slots”；PERSIST 和 COMBO 说明持久/重建状态可能有用，但其任务并没有证明真实传感器证据写回后能改变 3D occupancy。PreWorld 说明递归 volume state 很接近 occupancy 接口，TAWM 说明 transition 应知道时间间隔。我的统一推断是一个**候选设计与待检验假设**：让 belief state 把场景内容、ego/camera、观测 support/age、实体关联置信度和 `Δt` 分开，通过 observation update 后进入 dynamics，再由 occupancy decoder 读出。以上变量分解不是这些论文已经证明的必要条件；它只是能直接检验当前地址连接为何没有产生 GMO 收益的最小结构。

这与已经失败的 shared rigidstate 有三个可验证差别。Cpl/Fix 已经把学习位姿修正接入 occupancy 采样地址，所以差别不在于“有没有结构连接”，而在于连接是否由可审计的观测更新和实体/时间对应支配：

1. 新状态在已有地址连接之外，必须有当前合法观测的写入门控，并保留未观测区域的 age/support；
2. motion 不能只作为并行辅助 head，必须进入产生未来 occupancy 地址的 transition；
3. 终端 decoder 必须做 zero/no-op、正确绑定、错对象、错时间四种干预，显示目标区域响应而非只显示速度或 latent loss 改变。

因此“把 flow 接 decoder”不构成新机制，除非干预证明 decoder 依赖正确关联的 observation-updated state；反过来，若无对象槽的 belief/grid latent 已通过这些干预，则只能说明显式对象持久性在该实现和数据条件下不是必要条件，不能推广成所有场景的定理。

## 一个低成本决定性实验

在现有 O 的 train512 上只训练一个小型 state-update adapter，保持 O、数据、单 seed11 和训练预算固定；不使用 GT 虚拟点、GT 未来框或 GT 速度作为部署输入。当前 GMO 概率可以作为合法的软支持/输入特征，但不能被解释为 opacity、真实可见表面或正在运动的概率。adapter 输入合法的 camera+radar 历史证据及允许的 GMO 软支持，输出空间 state residual 以及 `support/age/association-confidence` 三个附加量；transition 额外输入真实时域 `Δt`。预注册四个终端干预：

|组|state update|终端干预|
|---|---|---|
|A|无持久 update，当前 O state|原始输出|
|B|持久 carry，但 update gate 置零|原始输出|
|C|合法观测 update，正确时间/实体绑定|原始输出|
|D|同 C 的容量和输入，但历史证据错对象或错时间|原始输出|

在同一固定权重上，再将 C 的 state residual 置零、替换为 D residual，并交换 occupancy query 前的 state；记录四个 horizon 的 future GMO、moving occupancy recall、虚假动态增量、目标/非目标响应和 support coverage。物理 moving/static EPE 继续只按已认证坐标绑定的固定 D/CV 读出报告；若 adapter 自身要报告 EPE，必须先实际实现坐标状态读出、固定坐标变换和独立绑定核验，不能把 adapter latent 位移直接称为物理 EPE。保留全部原始材料点分母；未支持点计入代价，GT 只作事后分层。与 O、M0 和全部固定速度策略比较，不从 dev200 后验挑阈值。

**成功标准：** C 相对 A/B 有可解释的目标区域 occupancy 改变和联合收益，且 D 显著削弱或错位该改变；这支持“小型、slot-free、观测更新的持久 belief state”作为下一步候选，再扩大到未参与选择的 scene。**失败标准：** C 与 B/A 无差别，或只改善 latent/速度而 GMO 无响应，或 D 与 C 一样；这只能否定当前 adapter 实现、训练预算和绑定方案下的效果，不能否定所有持久性设计。此时优先保留 O 并继续做观测可靠性/传感器退化与坐标绑定诊断，再决定是否追加另一种持久状态；不因一次短训自动宣告整条路线失败。

## 最值得采纳或否定的三点

1. **采纳：** 用“合法观测 update → `Δt`-conditioned dynamics → occupancy readout”作为最小可证伪接口；重点是 support/age/association，而不是另一个 motion loss。
2. **采纳但降级：** 先做 slot-free belief/grid state；PIGDreamer 和 World4Drive 说明显式对象槽不是先验必需，PERSIST 只提供状态解耦结构参考。
3. **否定：** 仅凭速度分支、共享 rigidstate、future latent 对齐或 flow probe 的数值变化宣称 motion 已进入 GMO；当前 V 置换和 Cpl/Fix 结果已直接反驳这种推断。
