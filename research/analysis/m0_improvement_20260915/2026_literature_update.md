**2026 文献更新：观测来源、未来表征监督与滚动漂移**

核查截至 2026-09-15。本次新增核读 5 篇全文：4 篇首次公开于 2026 年；SparseWorld 是 2025 预印本、2026 AAAI 正式发表，单独注明。核对 4 个作者仓库，其中 3 个含实现，OccSim 仍只有项目材料。未使用 skills、安装软件、连接实验服务器、训练或修改冻结协议。[来源与代码版本收据](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/2026_literature_sources.json) 包含正文哈希、仓库 commit、文件哈希及核读行号。

目前最有用的新先例是 **ForecastOcc：以训练期未来图像特征监督预测状态，并让各预测时域持续读取观测上下文**。这说明“增加未来表征损失”“从当前特征初始化未来查询”已经是先例，不能作为我们的新贡献。另一个重要结论是：这些论文使用的“漂移”涵盖不同问题，不能把时域越远成绩越低直接归因为 exposure bias。

| 工作 | 推理输入与任务 | GT 占据是否为输入 | 代码核查 |
|---|---|---|---|
| ForecastOcc，ICRA 2026 | 历史/当前相机；预测未来语义占据，1/2/3 s | 否；训练有未来图像、占据与深度监督 | 作者实现、GPL-3.0，commit `79a4194c3a06` |
| HERMES++，2026-04 预印本 | 相机、文本/指令及未来 ego 条件；未来点云与场景理解 | 否；训练有点云几何监督 | 作者实现、Apache-2.0，`038fa639bf6e` |
| SparseWorld，AAAI 2026 | 历史/当前多视角相机；语义占据与规划 | 否；训练合并多时域占据 GT | 作者实现，`14f5fe3bdd34`；未见仓库整体许可证 |
| OccSim，2026-03 预印本 | 初始静态占据及未来 ego 轨迹；长程地图生成、另行模拟车辆 | 是，占据种子/其 VAE latent | `2093fdf33b42` 只有文档；作者称录用后发布实现 |
| GEM，2026-05 预印本 | 历史相机与历史 ego 状态；连续时间语义占据、规划 | 主相机设置否；另有 LiDAR 初始化变体 | 正文已读；未找到可核实作者实现链接 |

上表输入/任务来自各自[ForecastOcc 正文](https://arxiv.org/html/2602.08006v1)、[HERMES++ 正文](https://arxiv.org/html/2604.28196v1)、[SparseWorld 正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/37347/41309)、[OccSim 正文](https://arxiv.org/html/2603.28887v1)、[GEM 正文](https://arxiv.org/html/2605.17682v1)；代码状态来自下述固定版本仓库检查。外部 mIoU、点云 Chamfer、生成 FID 都不是我们的 fine-GMO 主指标。

**1. ForecastOcc：当前最直接的强对照**

论文 §III-C、§IV-C 使用未来状态对齐损失 FSA。任务损失单独训练为 11.43 mIoU，FSA 单独训练为 18.03，结合为 19.68；但 FSA-only 同时冻结 downstream，故不能把前两者之差视为同优化范围下 FSA 的净效应。当前特征初始化查询优于 learned query（19.68 对 18.36），支持使用已有观测表征。语义占据类别、标签和 1/2/3 s 时域与 M0 不同。[原文表 III、V](https://arxiv.org/html/2602.08006v1#S4.SS3)

以下是 **2026-08-24 作者代码版本**的明确契约：

- **教师是什么层。** `image_encoder` 经 EfficientNet 与 SECONDFPN，输出每相机 256 通道、图像分辨率的 1/16；配置的四尺度通道为 16/24/64/152。目标不是 BEV，也不是占据标签。启用 FSA 时冻结 `img_backbone`、`img_neck`；提取时两者设为 eval，未来分支明确 no_grad，test 时返回 None。[detector L65](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L65)、[L86](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L86)、[L273](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L273)、[配置 L47](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/configs/forecastocc/forecastocc.py#L47)
- **对齐发生在哪里。** FSA 对同一相机未来图像栅格的预测/教师特征计算 SmoothL1 与 flatten 后的 cosine，逐中间层监督；默认总权重 30。当前、历史、未来的同一相机共享 resize/crop/flip/rotate。该损失没有显式 SE(3) 或物体对应 warp，“alignment”指表征目标对齐，不能解释成三维运动校正。[损失 L407](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L407)、[逐层监督 L447](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L447)、[共享增强 L863](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/datasets/pipelines/loading.py#L863)
- **是否用前一生成状态。** 每个 horizon 在 L336 重新从 `curr_feats` 初始化 query，context 始终由 past/current 构造；没有把上一个 horizon 的生成特征加入 memory。当前代码还为各 horizon 建立独立 synthesizer 参数，不能宣称跨时域共享参数。每个未来 voxel 特征随后与当前/历史 BEV context 拼接，再过共同的 3D encoder，形成另一层观测—未来联合修正。[forecasting_module L82](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/dense_heads/forecasting_module.py#L82)、[L323](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/dense_heads/forecasting_module.py#L323)、[联合编码 L370](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L370)
- **坐标及额外监督边界。** 发布实现的 FSA 比较未来自身相机平面的特征；lifting 重用当前相机 calibration，未来 GT occupancy 文件直接按未来帧加载，未在该 loader 中 warp 到 t0。训练还使用未来 LiDAR 投影深度监督。测试 pipeline 虽读取 points，`extract_feat` 中 `pts_feats=None`，预测链使用相机特征。迁移到 M0 时须重新声明 teacher BEV 属于哪个 ego frame，依据 M0 原输出坐标作映射及有效区域检查；不能直接相减两个不同坐标的 BEV。[lifting L286](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/models/detectors/forecastocc.py#L286)、[GT loader](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/datasets/pipelines/forecastocc_loading.py#L20)、[depth targets](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/mmdet3d/datasets/pipelines/forecastocc_loading.py#L61)

可执行的下一科学问题：**在同一容量、占据风险和样本预算下，是保留观测入口、直接多时域预测，还是未来状态监督改善了任务？** 后续若启动新轮，可先固定一种预测器比较原损失、正确配对未来特征监督、相同权重与区域的辅助监督对照；再固定监督比较 autoregressive 与从 t0 直接查询各 horizon。未来教师只能出现在 train target 路径；错时配对只能作为会产生分布变化的机制诊断。若特征距离下降但占据和长时域误差不改善，则否定该表征目标的任务价值。

元数据与复现限制：项目页标 ICRA 2026，arXiv 首投 2026-02-08，其 BibTeX 却误写 2025；这里按 2026。当前配置 `num_layers=8`，论文消融最佳为 3；README 权重表也不同于论文主表。因此不能把当前发布配置称为论文消融的精确复现。[项目页](https://forecastocc.cs.uni-freiburg.de/)、[当前配置](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/configs/forecastocc/forecastocc.py)、[README 与 GPLv3 声明](https://github.com/robot-learning-freiburg/forecastocc/blob/79a4194c3a060450da2b779f55c3b4a335d35d38/README.md)

**2. HERMES++：把当前状态连接到未来，以及直接监督几何表征**

Current-to-Future Link 将当前 BEV 传播为未来状态，结合 world query、文本及 ego modulation；Joint Geometric Optimization 同时约束点云输出与几何 latent。表 V 的 3 s Chamfer 从无 Link 的 2.377 降到 Simple Link 的 1.542，再经文本、ego 和加深达到 1.436。但表 VI 的 generation-only 为 1.434，联合理解为 1.436，不能声称语言联合训练必然优于单任务。多数消融用 25% 训练数据。[正文 §IV、表 V–VI](https://arxiv.org/html/2604.28196v1)

作者仓库的 `HERMESFutureRenderHead` 含 `Curr2FutureLink`、LiDAR SECOND alignment、Gram loss 开关；`stage3++.py` 实际开启后两者。仓库有训练配置和 checkpoint 链接，Apache-2.0；本次未下载权重或运行。README 新闻日期与 2026 arXiv 日期不一致，以 arXiv 提交日期为准。[模块](https://github.com/H-EmbodVis/HERMESV2/blob/038fa639bf6eed2fdb30de76e89bad910030584f/projects/mmdet3d_plugin/models/dense_heads/hermes_future_render_head.py)、[配置](https://github.com/H-EmbodVis/HERMESV2/blob/038fa639bf6eed2fdb30de76e89bad910030584f/projects/configs/hermes/stage3%2B%2B.py)、[仓库](https://github.com/H-EmbodVis/HERMESV2)

对我们的启示是检验**占据损失是否足以使中间状态具有可用几何**，不需要先引入 LLM。强对照应为同容量简单 Link、相同输出监督、增加表征几何监督；若没有表征监督也同样改善，贡献属于预测结构。该文是点云预测与理解，不能以它的 Chamfer 或“语义上下文”替代我们的未来 GMO 证据。

**3. SparseWorld：观测生成的未来查询持续补入，而非只有一个滚动状态**

论文 Eq.7 将上一预测查询与由观测编码、按时间分组的查询合并，并递归修正位置。训练先把多帧未来占据 GT 统一到当前坐标，学习扩展范围与查询时间分配；测试不输入这些 GT。短程消融全模型 11.82 mIoU，去 State Condition 为 11.31，去 temporal mask 为 11.52。其 self-scheduling 是**查询 timestamp 分配**，不是 scheduled sampling 或 teacher-forcing 修复。[正式论文 Eq.7、§3.4、表 3](https://ojs.aaai.org/index.php/AAAI/article/view/37347/41309)

代码 `SparseWorld4DTraj.forward_backbone` 确实逐 interval 拼接当前查询与对应观测查询，也有位置 detach；整体同时改变稀疏表示、损失、监督范围和容量，不能把总收益归因于“锚”。仓库含训练/推理代码及权重链接，但 API license=null，未找到仓库整体许可证；依赖目录自己的许可证不覆盖新增代码。[实现](https://github.com/MSunDYY/SparseWorld/blob/14f5fe3bdd3428002d913988fbfd9b653a1a207e/mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py#L240)、[作者仓库](https://github.com/MSunDYY/SparseWorld)

它支持保留观测来源作为合理先验，却不证明直接复制 t0 足够。当前 persistent-vs-rolling 实验正是更窄的内容对照。若 persistent 只减少 rolling 的损失而不优于 M0/continuation，应考虑观测如何被目标时域读取与监督，而不是仅增加槽数。若日后引入动态 query，应保留等容量静态 query、相同未来 GT 监督范围等对照。

**4. OccSim：静态几何约束的长程生成，不是动态占据预测上界**

W-DiT 从一个静态占据种子开始，依据 ego 变换 warp 前一 latent，使用可见性/随机 mask、条件注入和解码后的感知损失；以后各步条件是上一生成状态，非永久原样 t0。车辆由另一个布局模型及 IDM 变体处理。表 2 的 500 帧 FID：全关 1612.40、仅注入 1672.43、感知+注入 901.57、再加 SNR 744.71。表格并非完整析因，不能证明注入单独是主要收益来源。[正文 §3.1、表 2](https://arxiv.org/html/2603.28887v1)

作者仓库仅 docs/ReadMe，明确称代码和模型待录用后发布，不能承诺官方实现可跑。[当前作者仓库](https://github.com/Orbis36/OccSim/tree/2093fdf33b427c776e434c353dda574bfcc48288)

可借鉴的问题是：保持已观测几何与生成新区域是否需要不同处理。对 M0 应分开评价静态几何、实际运动对象、曾观测/新显露区域；ego 对齐只能解释静态部分。3000 帧静态地图的 FID 稳定不等于 2 s 运动占据更准，随机 mask 也不自动证明消除了 exposure bias。

**5. GEM：直接查询未来时间，是滚动结构的必要竞争解释**

GEM 用相机和历史 ego 状态估计可随时间查询的 Gaussian primitives，直接渲染任意目标时间，避开逐步生成链；另有额外 LiDAR 初始化变体。表 4 将分离的 ego/object velocity 改成统一 velocity 后，mIoU 从 13.60 降到 12.88；完整 4D covariance 为 11.42。正文未找到同表示下 AR-vs-direct 的严格消融，因此不能把整体成绩作为“误差累积被消除”的因果证明。[正文 §2、表 4](https://arxiv.org/html/2605.17682v1)

已核全文和 arXiv 页面，未找到可核实作者仓库、许可证或依赖；不使用另一个同名 LiDAR GEM 仓库冒充实现。[arXiv 原始记录](https://arxiv.org/abs/2605.17682)

这首先支持一个便宜的对照：在同一 M0 观测表征上，用目标时间条件直接输出各 horizon，并匹配容量、监督和样本曝光。若直接预测优于 autoregressive，则重新审视是否需要保留滚动机制；若无优势，不能据此否定所有连续表示。没有必要仅因论文总成绩就重写为 Gaussian 架构。

**研究解释与下一轮取舍**

现有 M0/native-cache 训练在四个未来步中使用自己生成的 latent，并通过全部未来步回传；它不是训练一直喂未来 GT latent、推理突然改喂预测 latent 的典型 teacher-forcing 情况。当前远期误差可能来自状态信息丢失、运动对应、几何表征不足或监督不充分。只有明确展示 train/test rollout 状态来源不同，才适合把 exposure bias 作为主诊断；停止梯度与否属于信用分配问题，不能直接等同于状态分布差异。[本轮 replay 代码](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py)

本轮三臂完成前维持冻结方案。后续优先把两个效应拆开：**预测结构**（滚动、持久观测、直接多时域）与**目标表征监督**（原任务损失、训练期正确配对的未来特征）。先用单 seed、相同风险和容量建立强对照，再决定扩训练覆盖。前景加权、额外深度/几何标签、教师容量和训练成本都应明示，不能把额外监督的收益归为记忆策略。若只是加入 FSA 或 simple Link 已足以解释结果，应将其作为已有方法的有效基线；新贡献仍需一个可被干预否定、超出该先例的机制。

未纳入主表的近邻 **OccTrans**：作者项目页可打开，但 Paper 指向 `<ARXIV PAPER ID>`、Code 指向 `YOUR REPO HERE`，均为模板 404；精确标题检索未找到可打开正文。因此只保留待核线索，不将其列为已读论文或可运行方法。[项目页](https://occtrans.github.io/)
