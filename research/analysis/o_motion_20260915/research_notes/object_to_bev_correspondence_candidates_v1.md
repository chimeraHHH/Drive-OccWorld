# 对象速度信号与 BEV 空间对应：候选研究札记

2026-09-16。仅分析已完成的 CRN-CV v2、K/B、三项 transport 结论和固定源码；没有读取本次 V/G 中间成绩，没有修改正式候选、协议或调度。下述一个主候选及查询诊断均待选择，不是已冻结实验或晋级门。源文件 SHA 见同目录 `object_to_bev_correspondence_candidates_v1_sources.json`。

**核心判断：已有证据支持“带预测几何归属的对象速度有用”，尚不支持“将这些状态汇聚成场景 latent 后，原生 BEV 能有效恢复每个对象该影响哪里”。** 这不是两者矛盾，也不能据此断言当前 conditioner 已失败。

**已证实的实验边界。** CRN-CV v2 在原全部 16074 对象-时域支持上评价；2 s moving XY EPE 为 3.386739 m，D 为 4.208240 m，差值 −0.821501 m，场景配对 95% CI [−1.079484, −0.601945]。其计算把当前材料点交给包含该点的预测框，重叠时取原 score 最高框，位移为该框速度乘名义 h；未覆盖点仍预测零并计入评价。因此被验证的是“几何关联＋速度”的整个输入可得构造，非任意无序速度集合。它没有测试完整假阳性空间的占据代价，材料点真值仍是框刚体代理，且含额外检测模型/历史/预训练差异。[评分源码](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/predicted_object_state_cv_diagnostic_v2.py:64)、[终点结果](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/crn_object_state_cv_dev200_v2/summary.json)。

K/B 的物理反馈已工程连通，但 K−B 未来 GMO 为 −0.000140 pp，CI [−0.000499, +0.000217]；2 s moving K−D 为 −0.001373 m，CI [−0.003088, +0.000310]。两臂共同的占据改善与 K 的非零物理 VJP，没有建立物理反馈的额外预测效用。这是既定同索引读出路径的结果，不能外推成当前对象 conditioner、所有 latent 模型或原观测无运动信息。[K/B 终点说明](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/future_state_motion_dev200_结果与决策.md)。

**源码事实与尚未证实的解释。** 当前 v2 有真实的对象特异 h×v：同一 token 保留 p0、完整旋转/尺寸/类别/score/速度，并追加 p0+h·v；不能沿用 v1 仅加共同时间偏置的批评。对象 token 被 32 个可学习场景 latent 汇聚、再 self-attention；这些 latent 没有被指定给某一对象或某个物理位置。BEV 查询由原 `bev_embedding + prev_features[-1]` 投影生成，模块未显式计算 BEV 米制坐标与对象未来中心之差。因此“哪个对象位于哪个查询附近”的关系需要在两次注意力间学出，而不是由结构直接保留。[v2 编码和查询](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/object_state_conditioner_v2.py:141)。

这只是潜在学习负担，尚无证据把它确认为瓶颈：对象中心仍在输入中；集合重排不变性本身不抹除位置；32 latent 少于检测框数不证明任务所需信息已丢失；原 BEV embedding/特征可能隐含位置，而且原生 future-head 另有 `bev_pos` 和参考点。注入进入原生 query 后还要经过未来 transformer，并非直接占据解码器。[原生注入和位置输入](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_base.py:387)。另一个未分离因素是 CRN-CV 的显式框归属/重叠处理与当前全场景聚合不同，不能把差异唯一归因于 latent 数量。已审 train512 score 分布也不证明低分框是噪声或速度错误。

**更根本的是两种查询语义。** 实际 physical readout 将当前 `F0(x)` 与未来 `Fh(x)` 在同一列索引拼接，经两层 3×3 Conv/GroupNorm 输出位移场；`gather_sparse` 始终在 t0 的 `source_flat_indices=p0` 取值。标签则是固定 t0 LiDAR R 中的材料点位移 `inv(G0)·Ah·inv(A0)·G0·p0−p0`。因此空间坐标系一致，监督目标也一致，不能直接判实现错误；但未来占据是“h 时刻空间 x 有什么”，位移是“t0 的材料点 p0 去哪里”，两种索引含义不同。如果对象证据主要出现在 p_h，当前读出没有显式把该证据关联回 p0。它仍可能借当前特征、原生全局信息和 GN 的全空间统计学会对应，不能声称总感受野严格为 5×5 或大位移必然失败。这个风险比“32 是否太少”更值得检验。[特征拼接](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/future_state_motion_v1.py:31)、[实际采样](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:236)、[标签构造](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/build_sparse_motion_supervision_v1.py:163)、[正式训练连接](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/object_state_forecast_train_v1.py:303)。

**旧负证据没有检验上述双查询结构。** train16 oracle 纯传播 19.6463% 低于 O 21.4686%，固定 half-oracle 22.1499% 则更高：反对用这个源/算子直接替换 O，不否定正确运动的互补价值。learned half-blend 20.67095% 仍低于 O：该固定输出传输链未把位移优势转成完整占据收益。supported A/Z 仅训练 97 参数门控、冻结 O/运动头，A−O +0.086514 pp 伴随 FP −248032、FN +20546：小幅原指标增益不证明运动更准。它们约束了特定的概率传播/冻结模型融合方式，未排除让源对象状态与目的空间地址共同参与可训练未来表示；也不能用 oracle 小样本正结果保证新结构有效。[oracle](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/oracle_transport_train16_结果与决策.md)、[learned transport](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/learned_transport_train16_结果与决策.md)、[supported fusion](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/supported_fusion_train_v2_结果与决策.md)。

**MotionPerceiver 的可迁移依据。** 已下载的作者仓库固定在 `cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f`，本次逐个复核 15 文件 SHA。其 decoder 将规则输出网格坐标做 Fourier 编码，以这些位置查询 latent 后输出占据；输入 `fpos_extra` 也编码位置/朝向，再接其他状态特征。它说明显式位置查询与场景 latent 可以共存，不是“latent 必然破坏对应”的证据。[官方 decoder 固定版本](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/perceiver_io.py#L338-L368)、[本地输入编码](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/object_state_sources_v1/motion-perceiver/src/model/_iadapter.py:329)。官方 `no-ctx` 配置使用该查询方式、128×256 latent，且 `only_vehicles/filter_future=true`；不移植其标签过滤、二维任务或数值配方，亦未在本项目运行或验证官方方法性能。[本地配置](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/object_state_sources_v1/motion-perceiver/cfg/no-ctx.yml:34)。

另核原文 **ImplicitO §3.1–3.3**：它以未来位置 `q=(x,y,t)` 同时查询占据和该处的 backward flow，式(1)为 `p_(t−1)−p_t`；§3.2 用学习 offset 寻找当前传感器证据。其双头共享查询的语义因而对齐。本项目标签却是源材料点的 `p_h−p0`，不能原样套“同 query 双头”，不能仅取负号替代地址变换，更不能混比两种 EPE。该文还指出检测阈值/漏检的对象路径局限。[作者原文](https://arxiv.org/html/2308.01471v1#S3)。共享查询与对象—空间连接已有先行工作，本候选只是针对现有证据的结构检验，不宣称这些概念新颖。

**只保留一个主候选：共享对象动态状态，分别用源地址和未来地址查询。** 对每个当前预测框保留同一个内部对象索引 j、当前锚点 c0_j、尺寸/旋转和动态状态 z_j(h)，同时显式保留预测未来锚点 c_h_j；初始运动假设可来自预测速度，后续变化只由当前可得输入学习。内部索引只连接本样本同一预测 token 的两种地址，不借 GT instance 或假定已建立跨帧跟踪。占据在未来空间 x 查询 `(x−c_h_j, z_j(h))`，生成原 future-head 的残差；物理位移在任意源网格点 p0 查询 `(p0−c0_j, z_j(h))`，并保留对象局部材料坐标供旋转/非纯平移读出。两读出共用动态状态及其未来锚点参数，而不是只有一个名义共享损失。

物理前向对完整规则源网格定义，再按原标签采样损失；地址和软归属完全由预测框/固定网格构造，GT 点身份/框归属不能进入前向，所谓对象身份仅为输入检测假设/原 box index。保留原 O 稠密占据与 D 残差基底、初始零输出；全框连续寻址，不加 score 阈值/GT ROI/框内输出硬门，不把 CV 直接当最终 flow，也不硬搬运 O0 的概率或按 O0 前景掩码复制特征。漏检、重复框、背景及未来新生占据不从评价排除，仍保留 O 主干覆盖路径，不能宣称对象分支已经解决它们。因此它检验的是可学习共享表示中的“同一对象，两种地址”，不是对旧输出 splat 再调融合系数。直接访问对象的内存成本需要预检，可精确分块而不可用删框代替。

**一项关键控制：只切换物理查询的地址。** 同容量、同当前状态、同初态/数据/预算，两臂占据均按 c_h 寻址；物理分支一臂以 c0 为键锚点，另一臂以 c_h 为键锚点，query 仍是同一 p0，动态 payload、材料局部坐标通道、监督与全部支持保持相同。从头匹配训练，而非只做终点分布外扰动。若源地址臂未取得可辨别增量，则当前预算下“显式源地址是有效缺失环节”未被支持，停止此机制主张。即使有增量，也仅支持新结构里的地址效用，不能倒推出旧网络必然错位，更不单独证明全部增益来自去掉 32 latent。若要声称预测速度而非对象几何的增量，仍需与既定零速度几何对照分开核验；此处不把地址控制冒充 V/G。

**metric query 降为诊断，不另算主方法。** 可在保留场景 latent 的路径中显式加入物理位置编码，对照预先固定的逐 anchor 错配编码，仅检验新增查询地址是否被有效利用。这不解决源/目的查询语义，也不能作为新的联合动力学方法或与主候选捆绑后作唯一归因。具体编码、地址分支及资源边界须在选中后、看候选结果前固定，本札记不发起训练。

最终仍须评价完整原 GMO 与同支持物理结果，并披露 D 和 CRN-CV 强对照、moving 召回、FP/FN 与 t0 代价；只赢错配控制、只优于 zero、只降低 EPE 或只提高占据召回，都不等于完成联合目标。控制若没有支持其指定机制，就停止该机制主张，不以加种子、延长训练或按本次 score 分位数调阈值补救。当前正式 V/G 是否已保留有效对应，应等既定终点证据，而非由此源码分析预判。

D→候选改善也不能全归功新学习：候选新增了已有正证据的 CRN 信息。若后续改用 CRN-CV 作为直接残差基底，必须另有同基底/同输入的匹配控制并比较原 CRN-CV，不能把更换基底的收益归因于学习。本札记保留原 forward/material 物理评价，不改本轮已冻结 V/G 判据。

本次操作：本地只读源码/已完成汇总及 SHA，另通过网页读 ImplicitO 原文指定章节；只新建本札记及来源清单。未下载新代码/PDF，未运行 Torch/GPU、模型、优化器或评价脚本。
