# 世界模型扩展调研：把历史观测质量与未来动力学分开检验

2026-09-17。使用 GPT-5.6-Luna 做独立替代路线调查，主 agent 核查当前实验、论文版本及三维跟踪/持久状态的官方源码。未使用 skill；本轮没有训练、没有服务器操作，也没有新模型超过 O。

本报告接续 [已有 30 篇正式会议论文的筛选](world_model_research_direction_v2_20260917.md)，新增深入检查 CUT3R、SpatialTrackerV2、DELTA 三个官方仓库的 16 个文件，并补充控制世界模型的反方观点。已有清单包含邻近感知/跟踪工作，不能全部称为开环世界模型；筛选、方法阅读、源码检查和运行复现分别记账。

## 1. 当前判断：值得推进两个竞争假设，不先决定模型形状

**假设 A：主要缺口是历史观测没有形成可靠、具有空间归属的运动状态。** 当前 RAFT 评分取自框内虚拟材料点投影；这些点并不天然对应真实可见表面。换成更强二维 flow 网络不能直接修复这个问题。新的候选应从合法像素/表面 query 出发，用标定和历史对应恢复状态，再研究状态如何影响未来占据。

**假设 B：主要缺口是学习目标和读出没有保留任务需要的未来变化。** 一个隐状态也可能足够，不一定需要显式实体、Gaussian 或完整三维重建。因此几何状态方案必须面对同输入、相近训练量的未来 latent 预测对照。

最新 [future-observation 诊断](../future_observation_readout_train16_结果与研究决策.md) 只支持降低“直接蒸馏当前冻结表示”的优先级：train16/8 scenes 上未来观测经同一 O 读出，future GMO IoU 从 21.468599% 到 21.529183%，但召回增加 4.784575 pp，同时 FP 增加。它没有比较完整 PR 曲线、另一个可学习读出或外部预训练表征。不能把这个单阈值结果写成“未来特征无效”或“几何路线已胜出”。

完整原协议保留 O 的 14.7478% future GMO IoU 与 M0 的 13.9449%；两者不是语义 mIoU。本轮 train16 数值不能与 full5119 横比。O 没有认证的物理运动读出，物理误差仍用已认证 D/CV 等方案单列。

## 2. 新论文、源码给出的具体改变

### CUT3R：读状态与更新状态有明确的代码边界

[CVPR 2025 正式页面](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html)，[方法正文](https://arxiv.org/html/2501.12387v1)。CUT3R 用持久状态结合新图像输出 pointmap，也支持对虚拟视角查询状态。后者是未观察视角的几何补全，不是未来物体动力学。

固定 commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`，[`src/dust3r/model.py:814–899`](https://github.com/CUT3R/CUT3R/blob/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf/src/dust3r/model.py#L814-L899)：每个 view 都可读出结果，但状态更新受 `img_mask & update` 控制，另有 `reset`。这是可独立干预的更新接口；`update` 是输入控制量，不能误称学习得到的可靠性门控。官方提供 checkpoint 入口，本轮未下载和运行。

**迁移价值：** 在新模型中区分历史观测更新、无观测时的状态转移、任务查询。比较正确历史更新、冻结状态、相同长度错关联更新，才有可能说明记忆中保存了什么。已有 O 的递归 BEV 本身不是新颖点。

### SpatialTrackerV2：联合对应可以作为独立深度差分的强对照

[作者项目页，ICCV 2025](https://spatialtracker.github.io/)，[方法正文 v1](https://arxiv.org/html/2507.12462v1)。它联合处理几何、相机运动和点轨迹；本次读到的论文版本与最新仓库需分别识别，不能用早期草稿保证最终代码逐项一致。

固定 commit `7e12274c52077860cebfe007a6290777db43b63c`：

- [`predictor.py:34–87`](https://github.com/henry123-boy/SpaTrackerV2/blob/7e12274c52077860cebfe007a6290777db43b63c/models/SpaTrackV2/models/predictor.py#L34-L87) 接收视频、外部 depth、intrinsics、extrinsics 与 queries。
- [`SpaTrack.py:270–297`](https://github.com/henry123-boy/SpaTrackerV2/blob/7e12274c52077860cebfe007a6290777db43b63c/models/SpaTrackV2/models/SpaTrack.py#L270-L297) 把外部输入写入内部 `*_gt` 字段。字段名不代表输入必须是真值；我们的接入必须记录真实来源，不能把预测深度升级为 GT。
- [`SpaTrack.py:505–546`](https://github.com/henry123-boy/SpaTrackerV2/blob/7e12274c52077860cebfe007a6290777db43b63c/models/SpaTrackV2/models/SpaTrack.py#L505-L546) 会调整内参分辨率，并依赖深度分位数、深度边界构造有效区域。这些筛选会影响远处和困难点覆盖，必须记录。
- [`TrackRefiner.py:188–199`](https://github.com/henry123-boy/SpaTrackerV2/blob/7e12274c52077860cebfe007a6290777db43b63c/models/SpaTrackV2/models/tracker3D/TrackRefiner.py#L188-L199) 有固定相机路径，初始化位姿见 :683–695；但官方 demo 默认 `fixed_cam=False`。不能仅传入外参就声称推理完全保留标定位姿。:954 的 `query_no_BA` 也不等于冻结全部相机估计。

**迁移价值：** 只在截至预测起点的历史窗口运行冻结模型，用已知标定与预测深度检验历史世界坐标轨迹；随后用固定 CV 外推隔离历史估计收益。它尚不是我们的未来占据模型。需先测短历史、远距离和道路场景的适用性；官方 demo 的场景不能提供本项目性能保证。

### DELTA：相机坐标三维轨迹不能直接相减当物体运动

[ICLR 2025 正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6e1e1d2e58afe1007da0bb76fac722a0-Abstract-Conference.html)，[正文](https://arxiv.org/html/2410.24211v2)。它预测已观测视频中像素的 UV、depth 和可见性，适合成为较直接的历史轨迹对照。

固定 commit `3367cda1c74d19e73296165f9826b213211678dd`：[`dense_predictor.py:21–119`](https://github.com/snap-research/DELTA_densetrack3d/blob/3367cda1c74d19e73296165f9826b213211678dd/densetrack3d/models/predictor/dense_predictor.py#L21-L119) 支持 `predefined_intrs`；[`model_utils.py:650–682`](https://github.com/snap-research/DELTA_densetrack3d/blob/3367cda1c74d19e73296165f9826b213211678dd/densetrack3d/models/model_utils.py#L650-L682) 默认焦距取图像宽度，执行 `K^-1 [u,v,1] * depth`，没有在这个函数里去除相机运动。接入 nuScenes 必须用实际 K，并逐时刻变换到共同参考系。深度更新为乘法比例，见 [`densetrack3d.py:395–402`](https://github.com/snap-research/DELTA_densetrack3d/blob/3367cda1c74d19e73296165f9826b213211678dd/densetrack3d/models/densetrack3d/densetrack3d.py#L395-L402)。

**迁移价值：** 比“独立深度＋两帧光流差分”更完整的时间对应基线。比较时两者用同一历史窗口、深度来源和标定，以免把深度模型变化混成跟踪机制收益。官方 depth/visibility confidence 不是经本项目校准的概率；保持原始分母及回退。

上述三者均只完成源码与方法核查，没有加载权重或本环境推理。源码、Git blob 和 SHA256 记录在 [manifest](world_model_extension_repo_audit_20260917/source_manifest.json)。

## 3. 为什么不能立即把 Metric3D＋RAFT 当作主方法

在同一参考系中，位置差分给出历史速度：`v = (p_t - p_(t-dt))/dt`。这是正确公式，但不保证估计正确。深度抖动、像素错配和遮挡换面都会变成速度误差。

**仅用于解释的误差传播例子，非实验数据：** 若两帧同一方向的位置误差各有标准差 1 m、相互独立且时间差 0.5 s，则差分速度误差的标准差为 `sqrt(2) / 0.5 ≈ 2.83 m/s`。误差相关性会改变这个数值；它不是我们的实际误差估计。由此应检验深度的时间一致性，而不能只检查单帧深度图看起来合理。

雷达径向测量、图像对应、预测深度约束的分量不同。将其先转成完整“确定速度”再简单相加会掩盖不确定性。这是方法设计动机，尚不能恢复旧 M3 的 covariance/NLL 主张；旧路线缺少的源归属与终端效用仍需解决。

## 4. 两条路线怎样公平竞争

| 阶段 | 实验与必要对照 | 它真正回答什么 | 不通过后的动作 |
|---|---|---|---|
| 历史信息提取 | 原 D/CV 与八阈值速度策略；独立深度＋RAFT；固定外参的 DELTA 或 SpatialTrackerV2。先统一当前/历史输入、深度、时间和坐标 | 强历史对应能否在原评分人口中增加运动信息，且不靠删去静止/远处点获益 | 若短历史或深度不足，停止扩展该外部模型；不以增加门控层掩盖无信息 |
| 统一状态原型 | A：带空间归属的历史更新与未来状态；B：任务可读的未来 latent。共享 encoder、合法输入、近似参数与训练预算；另有简单增容对照 | 显式几何的额外结构是否必要，或合适目标已足够 | 若 B 同样好，选择更简单路线；若 A 只在受支持子集好，限定主张或补足覆盖 |
| 任务和机制 | 原协议 future GMO、实际 moving recall、静止 FP、各 horizon；有认证 metric head 的候选再测原人口 EPE；固定权重的正确/错关联及状态冻结 | 运动是否提高未来空间预测，而不是只增加响应或改变阈值 | 运动更准但占据不升：任务连接未成立；占据升但运动不升：仅称占据方法 |
| 独立确认 | 先审计 scene 使用历史，再锁定未用于选方案的场景；清洁与有定义的传感器缺失条件；保持 seed11 | 收益是否超出已经反复看的 dev200，以及是否来自历史更新 | 只在人工退化有效：考虑可靠性论文，不能声称自然恶劣天气泛化 |

这不是同时训练一排新模型。先确定一个可工作的冻结历史估计基线，再投入一个完整原型和最有力竞争方案。训练预算由真实 profile 与拟合曲线确定；不把任意短训失败当成研究方向失败。到正式比较时保持充分且匹配的训练量，遵从单 seed 约束。

同表示 teacher 的单阈值校准问题可用已有诊断的补充读出检查解决；不要把它延长成另一串没有模型产出的 AUC 实验。也不再重复已经完成的 GT-motion oracle、wrong-horizon、速度置零作为新贡献。

## 5. 面向 ICLR 的贡献应放在哪里

建议研究的问题是：**部分观测下，怎样学到既保留历史运动证据、又能被未来空间任务实际使用的状态？** 几何状态与任务 latent 是竞争答案。

持久记忆、query、低秩 latent、flow loss、几何 warp 都已有先例。新的贡献若成立，需要同时给出：

1. 一个清楚的失败规律：哪些观测/运动条件下，运动读出改善却不能改善未来场景；
2. 一个解决该规律的状态或训练机制，且优于足够强的简单对照；
3. 机制干预和终端效用的一致证据，并在额外场景或第二种模型/数据条件下成立。

只有一个新融合模块和小幅平均 IoU 提升，还不足以支撑这条论证。当前证据最支持先检验历史三维对应；未来表征路线保留为强竞争者，不能先宣布被排除。

反方报告见 [Luna 独立调查](world_model_luna_challenger_20260917.md)。主审已修正 V-JEPA 投稿状态与 V-JEPA 2.1 代码版本的混用，并把 TD-MPC2 的“无观测重建 decoder”与占据任务解码区分；雷达测量不能被称为控制 action。三个控制/表征仓库的核心代码与 README 也已保存并校验，连同几何部分共 6 个仓库、22 个文件；未运行外部模型。
