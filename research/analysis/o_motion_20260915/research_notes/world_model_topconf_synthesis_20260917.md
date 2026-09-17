# 世界模型文献与官方仓库调研：当前工作的推进判断

日期：2026-09-17。完成范围：两位 **GPT-5.6-Luna** 子 agent 分别调研占据/运动与 latent dynamics，根 agent 补充几何/融合并复核主要论断。合并去重后为 **17 篇正式会议论文**，覆盖 CVPR/ECCV/ICCV/NeurIPS/ICLR 2024–2026；其中包含邻近的感知和融合工作，并非每篇都是未来世界模型。另单列用户提供的三篇 2026 预印本。没有把 17 篇都当作全文/代码深读：重点检查了 **6 个官方仓库的核心实现**，其余以论文机制与公开材料筛选为主。

结论：最值得发展的方向是 **部分观测下，持续维护有空间/实体归属的场景状态，并让同一运动状态决定未来占据的空间变化**。这是一项待证伪的研究假设，不是已经优于 O 的方法。一般的 Gaussian、slot、flow loss、cross-attention 或静动态分支组合，已有充分先行工作，不能单独作为创新点。

## 1. 覆盖范围和证据深度

|类别|去重后的论文|关键阅读深度|
|---|---|---|
|占据、运动与时空场|UnO（CVPR24）、Cam4DOcc（CVPR24）、DriveWorld（CVPR24）、OccWorld（ECCV24）、OccProphet（ICLR25）、DynamicCity（ICLR25）、DIO（CVPR25）|OccProphet / DynamicCity 实际代码；DIO 正文机制，无已核实官方实现；其余见逐篇表|
|latent state、对象与任务读出|PLSM / *Simplifying Latent Dynamics with Softly State-Invariant World Models*（NeurIPS24）、DriveX（ICCV25）、Dyn-O（NeurIPS25）、LPWM（ICLR26）|Dyn-O / LPWM 实际代码；DriveX 正文 future-state readout；PLSM 机制筛选|
|持久几何、遮挡与观测融合|GaussianWorld（CVPR25）、GaussianFlowOcc（ICCV25）、RaCFormer（CVPR25）、EmbodiedOcc（ICCV25）、GaussRender（ICCV25）、SOAP（CVPR25）|GaussianWorld / RaCFormer 实际代码；其余论文/README 筛选|

逐篇 primary URL、输入/监督/推理边界、官方仓库和权重状态分别在：

- [占据与运动报告](topconf_scout_occ_motion_20260917.md)
- [latent dynamics 报告](topconf_scout_latent_dynamics_20260917.md)
- [几何与证据报告](topconf_scout_geometry_evidence_20260917.md)
- [用户给出的三篇文章](three_world_model_papers_20260917.md)

六个深查代码的固定 commit：OccProphet `d41f8ae9b0b5cb47e0152dfec50115ab9cc0006c`；DynamicCity `eaf0c6e23ed56c8fb8cdaffe15f22e7d341c529a`；Dyn-O `69e0f97d2cecbf35ebb2e3fdf1ecedf5dc0a4077`；LPWM `4cf53c403433e64c01652ac2adbec66231a46dea`；GaussianWorld `b43629eaecffd5a7cbaac1a55517766e6263e4fc`；RaCFormer `4d4c534f84acef9ec390523134367aad40619057`。

“作者给下载链接”“权重页面可访问”“权重文件验证”“本环境推理成功”是不同证据等级。本次 GaussianWorld/RaCFormer 仅额外核到权重落地页面 HTTP 200 和文件名；没有下载这些候选模型权重或运行它们。DIO/DriveX 的对应官方实现没有核实，不能承诺直接运行。DriveX 存在同名项目，不能误用另一篇的仓库。

## 2. 文献改变了什么判断

**运动可读出、运动被使用、运动有用，是三个不同命题。** OccProphet 可配置共享或分开的 predictor，但核到的训练路径分别计算 occupancy/flow 输出，预测 flow 并不进入 occupancy transport，说明同时预测不等于用 flow 搬运占据。OccWorld 的未来 token 进入 decoder，DriveX 的任务 query 从预测未来 BEV 取信息，则展示了明确的未来状态读出路径。但有连接仍不保证模型依赖它，更不保证物理运动准确。[OccProphet 代码](https://github.com/JLChen-C/OccProphet)、[DriveX 正文](https://openaccess.thecvf.com/content/ICCV2025/papers/Shi_DriveX_Omni_Scene_Modeling_for_Learning_Generalizable_World_Knowledge_in_ICCV_2025_paper.pdf)。

**DIO 最有价值的是 source 与未来 query 的联合表示，而非再加一项 loss。** 它用同一隐式函数输出 occupancy/flow，并在训练中以 flow 移动 query 后约束 occupancy 一致性。这提供实例归属与未来位置联合建模的先例；但其未来 occupancy 可直接按时空 query 查询，不能说模型推理必然通过 flow rollout，更不能从 loss 推导因果依赖。我们的研究若强调“运动被消费”，需要额外固定权重干预证明。[DIO 正文](https://research-assets.waabi.ai/DIO/paper.pdf)。

**状态应随观测修正，而不是随观测缺失消失。** GaussianWorld 的状态持续保留并用当前 RGB 更新；Dyn-O/LPWM 在对象/粒子状态中显式处理动态变化、可见性或存在性。这启发将“场景如何演化”和“传感器这次看到了什么”拆开。不过 GaussianWorld 是每步有新图像的 streaming 感知，LPWM 粒子的 depth 是合成顺序属性，均不等于我们的无未来图像 3D 预测。[GaussianWorld](https://arxiv.org/html/2412.10373)、[Dyn-O](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b03b9ec80fc599bb5161746edff0f322-Abstract-Conference.html)、[LPWM](https://arxiv.org/html/2603.04553v1)。

**融合首先是空间与时间对应问题。** RaCFormer 用 query 的预测速度回查历史位置，并读取原生视图特征；SOAP 和 GaussRender 则提醒我们，投影落在图像中不等于该 3D 点具有有效表面观测。当前虚拟材料点在框内，不能全部作为图像表面对应点。[RaCFormer](https://arxiv.org/html/2412.12725)、[SOAP](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_SOAP_Vision-Centric_3D_Semantic_Scene_Completion_with_Scene-Adaptive_Decoder_and_CVPR_2025_paper.html)。

这些先行工作也提高了新颖性门槛：提出“预测运动后 warp 特征”“让任务读未来 latent”“加入静动态分支”都不足够。潜在贡献应是解释并解决**部分观测、源状态不完整、实体绑定不准时，运动为何不能转化为未来占据收益**，并证明解决机制跨越某个具体 head 或数据子集。

## 3. 对照我们的真实证据

目前 O 的完整原始协议 future GMO IoU 为 14.7478%，M0 为 13.9449%；这是 GMO IoU，不是与其他论文任意 semantic mIoU 可直接比较的数字。O 没有认证的物理 flow 读出，材料点运动对照是 CRN-CV。

最新 dev200（100 scenes）完整支持范围、object-equal 的 2 s XY EPE：

|固定策略|moving EPE，m ↓|stationary EPE，m ↓|
|---|---:|---:|
|CRN-CV|3.386739|0.093406|
|覆盖处 CV、未覆盖处全用 D|2.824611|0.148319|
|同上但 D 自身预测速度 ≥0.5 m/s 才使用 D|2.836491|0.113956|

第三行保留约 97.89% 的全 D 补充 moving 收益，同时减少约 62.58% 的额外 stationary 代价；仍有 +0.020550 m 静止误差。八个阈值全部报告，**没有据 dev 选出最佳部署阈值**。该结果来自本轮实际完成的独立 dev GPU 诊断，元数据/计数与内部参照检查通过；本地尚未对全部对象原始输出独立重算。它没有产生 O occupancy 改进结果。[结果与审计](../fixed_speed_routing_dev_shared_20260917_independent_results_v1.md)。

因此新机制不能只证明“优于零/CV”或“AUC 提高”，必须在完整支持范围的 moving 收益—stationary 代价上超过简单速度策略，并进一步提升未来占据。不能靠删掉困难点获得改进。

已有 16-anchor GT-motion、partial oracle、wrong-time 实验提示有限的任务互补空间，但不构成新方法结果，也不是严格上界。O 当前 hard-positive 源仅覆盖约 43–47% 的诊断 moving 虚拟点；这提示源表示可能有瓶颈，却不能直接解释为真实表面/物体只有这么多可见。此前 Cpl/Fix 的小幅占据收益没有对应可靠运动收益，亦不能支持“只要连上两个 loss 就会联合变好”。

## 4. 推荐的整体研究方向

可检验的问题是：**在相机/雷达只能提供部分运动证据时，怎样维护可查询、持续存在的场景状态，使正确的运动更新选择性地改变正确实体的未来占据？**

候选结构分为三个过程：

1. 历史状态先按 ego 与动态转移预测当前状态；几何位置、内容特征、运动、观测时间和数据关联分开记录。
2. 当前相机/雷达只更新其实际支持的部分；未观测不等于静止或不存在，观测置信度也不直接当成物理速度。
3. 未来阶段不读未来图像，由同一状态转移产生未来场，再由空间 query 解码占据。当前未建模区域的补全保持独立来源标记，避免把补全先验当成传感器证据。

源可以是预测对象、空间 primitive 或稀疏 query，不能预先断言 slot 与真实对象一一对应。也不必先整体改成 Gaussian 或 DiT。方法变化的重点是**状态、关联、转移和读出的关系**；实现尺寸由验证和资源测量决定。若保留 O 补全通路，须防止 decoder 绕过运动分支，仅靠 O 获得分数。

与旧 M3 的区别：此处先验证可观测证据、对象归属和终端作用，不将未认证 covariance/NLL 作为主贡献，也不恢复 S3 四槽短训路线。

## 5. 下一阶段两项决定性实验（尚未启动）

### A. 固定模型上的证据归属与终端作用诊断

先完成正在运行的冻结 RAFT train512 诊断。比较真实历史对应、错对象、错时间和相同尺度的替代证据，匹配预测速度/距离/可用支持范围；所有原始评分点保留，缺证据点仍计代价。几何检验只使用当前/历史合法输入，GT 表面/身份若用于分层或 privileged 上界必须单列，不能用于推理 query 选择。

终端干预仅能落在代码中实际存在的 D→transport 或已实现的 Cpl 分支，O 是冻结参考。先记录哪条边存在、输入维度/坐标/时刻、哪些权重冻结，再按固定规则交换实体/时间。**不能凭空假设 O/M0 已有 object motion latent、visibility 或持久状态接口。** 若必须先加 adapter，明确属于下一项新模型实验，不能称作现成模型诊断。

新增证据应是：正确实体的占据变化有特异性、非目标区域泄漏较少，而且变化改善终端误差；不是重复已有 wrong-horizon 总分。已有 oracle/错时结果直接复用作背景。

停止条件：真实对应不优于匹配的错误对应，则暂不训练依赖该证据的新融合网络；若效用只限于 GT 定位/GT 可见性，判为部署几何未解决；若只有响应幅度无任务改善，不宣称有用。

### B. 持久源状态与任务读出的受控验证

当 A 支持有效证据后，在当前 train512 上训练一个足以检验假设的状态更新/读出原型，保留 O/M0 与单 seed11。先 profile 并检查拟合/收敛，再冻结训练预算，不把任意 512 updates 当作足够训练的保证。dev200 只用于预注册后的比较；它已经多轮用于研究选择，不能再称完全独立最终测试集。

主要对照分解两项：瞬时源 vs 持久源；独立 motion head vs 同状态控制 occupancy 地址。参数、数据和训练量匹配；在最终固定权重上再做正确绑定/错绑定干预，以区分增加容量、一般正则化与运动实际效用。独立的物理评分仍遵循原始所有材料点分母，GT 类别只分组评分。

报告四个 horizon 的 future GMO、moving/static 物理 EPE、移动占据召回、虚假动态增量、源覆盖与未覆盖代价；加入目标与非目标区域响应。一个方法若占据涨、运动不涨，只能称占据方法；若运动涨但占据不涨，任务连接未成立。与全部固定速度策略比较收益—代价曲线，不能只挑有利的一项。

晋级条件：出现可解释且非重复的联合收益，并在另一个未参与方案选择的 scene 集上验证；先核查已有所有 train/dev/test 场景使用历史，再定义验证集，不能事后把已经看过的集合称 sealed。场景 bootstrap 可评估样本不确定性，但不能代替多训练 seed；遵从用户要求本阶段不做多 seed。

## 6. ICLR 贡献需要什么；什么暂不做

要达到面向 ICLR 的论证强度，最终需形成“可推广的问题—机制证据—联合效用”的链条：证明困难来自部分观测下的归属/持续性/读出，而非仅某一 loss 权重；证明纠正机制有对象和时间特异性；证明收益在干净与受控传感器缺失条件、不同覆盖/运动分层上成立。若资源允许，再用第二个 backbone 或数据条件检验可迁移性。当前资料不足以宣称已达到该标准。

暂不优先：完整训练视频 diffusion/DiT；换大 backbone 追榜；仅为接近别人的 mIoU 改评估定义；把 DynamicCity 的压缩表征作为未经诊断的主路线；直接增加更多 flow/photometric loss。时空低秩因子可作为后续效率备选，但不能解决错误观测地址本身。

若干净条件下持续无法获得联合收益，而观测更新机制能稳定抑制传感器缺失造成的损害，可把主问题转为**部分观测与传感器退化下的状态可靠性**。届时必须检验人工退化是否代表所声称的场景，并证明优于冻结 O 与简单回退策略；不应预先写成全天候泛化结论。

## 7. 用户提供的三篇文章怎样放入这条路线

- [Drive-HWM](https://arxiv.org/abs/2609.03572)：启发运动相关 latent 与任务接口；规划收益和 optical-flow probe 不证明对象材料运动或 occupancy 因果效用。
- [Low-Rank Dynamics-Effective Latent Carriers](https://arxiv.org/abs/2608.15156)：最直接启发固定权重、错对象/错时间/no-op 的特异性检验。其受控二维环境有精确反事实对，本项目没有；不能照搬为“真实驾驶低秩因果载体”。
- [DriveCache](https://arxiv.org/abs/2608.16354)：启发评估局部变化对最终输出的后果。其对象是 diffusion 缓存、ego action 和终端 latent 保真，不是当前模型的四个未来物理时域，不能直接移植缓存策略当运动方法。

三篇结合顶会代码共同支持的研究准则是：**先弄清楚状态里的运动属于谁、由什么观测支持、怎样影响最终任务，再判断需要怎样的模型和监督。** 以上是文献与当前证据支持的推进建议；本轮没有启动新的候选模型训练，已有服务器诊断由现有监控继续管理。
