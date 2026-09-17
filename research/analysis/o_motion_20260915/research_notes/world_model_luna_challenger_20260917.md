# 世界模型方向独立反方调研：冻结表征、任务充分 latent 与长期 rollout

2026-09-17。本文是对“持续几何/实体场景状态”主张的独立反方检查，保留三篇正式主会论文，并将 V-JEPA 作为状态明确的补充材料；另核查三个作者仓库。它不重复已有 30 篇综述，也不把以下路线声称为本项目收益。未训练、未 SSH、未改变 O/M0、seed11、S3 或外部 baseline 状态。

## 先修正一个容易过度的结论

`future_observation_readout_train16_结果与研究决策.md` 的 16 anchors/8 scenes 诊断中，真实未来观测接入同一个冻结 O 读出后，整体 Future macro GMO IoU 只增加 **0.060585 pp**，recall 增加 **4.784575 pp**，同时 FP 增加 127,199、FN 减少 27,408；四个时域各有不同方向，且不是移动物体 recall 或 EPE。这个结果足以降低“直接把未来观测蒸馏进当前 O head”的优先级，却不能排除：

1. 未来信息被当前分类阈值、插值、ROI 外回退、可见性和新出现内容损失；
2. 未来特征若改用任务相关的目标，可能比当前 logits 更适合未来占据；
3. 一个任务充分的 latent 不必显式成为几何实体，也不必重建所有不可预测细节。

因此，下面的替代方案针对的是“未来状态的充分性、稳定性和长期可滚动性”，不是把 train16 当作 teacher 上界。任何本项目收益仍须用原协议 Future GMO、moving/static 分组、合法历史前缀和固定 seed 独立验证。

## 证据筛选

| 论文 | 正式状态 | 反方价值 | 不能外推的部分 |
|---|---|---|---|
| [V-JEPA, OpenReview submission](https://openreview.net/forum?id=WFYbBOEOtv) | 该页面显示 Submitted to ICLR 2024；本报告没有找到足以证明录用的 primary proceedings 记录，故不计入正式主会 | 掩码时空区域的 latent 预测可以优先保留可预测、任务有用的信息，不要求显式对象/几何 | 仅作补充机制证据；视频分类/动作理解不是 3D Future GMO，不能当本项目提升 |
| [IRIS, ICLR 2023 OpenReview](https://openreview.net/pdf?id=vhFu1Acb0xb) | OpenReview 标明 Published as conference paper at ICLR 2023 | 离散观测 token + 自回归 Transformer + reward/termination head，直接针对 imagined long rollout | Atari 100k 与驾驶占据的观测、动作和误差结构差异很大 |
| [TD-MPC2, ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf) | ICLR 2024，官方站点标为 Spotlight | decoder-free latent dynamics 可把状态压到控制/任务所需量，以 value/reward 约束而不是几何重建为中心 | 连续控制 reward 不是占据标签；无代码证据表明它能直接表达 3D occupancy |
| [PLSM, NeurIPS 2024 main track](https://papers.nips.cc/paper_files/paper/2024/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html) | NeurIPS 2024 Main Conference Track | 用状态与 action-induced latent change 的互信息正则化，使长期动力学更简单、更状态不变；是“改善动力学结构”而非“增加容量”的反例 | 论文环境和控制任务；需要重新定义 action、状态差和占据读出 |

作者仓库实际核查如下。它们是可读代码与权重/推理接口的证据，不等于在本项目上复现成功。为避免 `main` 漂移，下面代码均来自本轮浅克隆的固定 commit；原始 cache 根目录为 `/tmp/luna-wm.coNkHe/`；主 agent 已独立检查 Git blob 并把三个仓库核心源码与 README 固定保存到 [工作区源码目录](world_model_extension_repo_audit_20260917/)，对应 SHA-256 见 [manifest](world_model_extension_repo_audit_20260917/source_manifest.json)。

- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) @ `204698b45b3712590f06245fbfba32d3be539812`：官方 PyTorch codebase，README 提供 V-JEPA 2/2-AC/2.1 检查点入口；本报告读取的是 **V-JEPA2.1** 的 `app/vjepa_2_1/train.py`，不是 V-JEPA1 代码。文件 SHA-256=`6057c8968c90dc96935680e8f617bbe8db836fc30698a053b0aceaa985f39b93`。不是本项目数据上的推理成功。
- [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) @ `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`：作者标明 official implementation，README 提供 300+ checkpoints；核心 `world_model.py` 可实际读到 encoder、latent dynamics、reward、termination、Q heads。文件 SHA-256=`55b85a9481d131b6b42265e15ece17bcc05f4ddf83dfd12c5be74e3a04f0d737`；权重/环境推理成功仅限其公开控制任务。
- [eloialonso/iris](https://github.com/eloialonso/iris) @ `24326aaaa283c527f42b89b44cfdecf2665a7a16`：作者仓库并提供 pretrained models；核心 world model 代码实际实现 observation-token、reward、termination 的交叉熵。文件 SHA-256=`7aefe36fa85bc5aa25ef922350a3416d036f48bd33955d15cea869f043fa2578`。它的 imagined rollout 接口可读，但不代表对驾驶数据可直接部署。

## 竞争路线 A：任务可读的冻结/慢更新 latent 预测

**forward 路径。** 历史相机窗口 `I[t-k:t]`（必要时以合法的雷达/ego token 作为条件）经冻结或慢更新 encoder 得到多尺度 `z_ctx`；predictor 接收 context mask 和 future mask，只生成未来 latent `ẑ[t+Δ]`；占据 head 读取 `ẑ`，而不是读取未经过 predictor 的 t0 logits。训练目标是同一 encoder 的未来窗口 target `z* = stopgrad(E(I[t+Δ]))`，用归一化 L1/L2 或 cosine loss；可加入只在可见/有标签网格上计算的 occupancy auxiliary CE，但不能把真实未来观测放进部署路径。

官方 V-JEPA2.1 代码的可审计路径是 [`train.py` lines 361–368](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py#L361-L368)（复制 target encoder，建立 predictor）和 [`lines 613–627`](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py#L613-L627)（context encoder/predictor）；[`lines 677–703`](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py#L677-L703) 将 predictor 输出与 target latent 的 masked loss、context loss 相加。这里的“冻结/慢更新”是可迁移接口，V-JEPA2.1 的视频数据和模型规模不是本项目配置；它不能反向证明 V-JEPA1 已录用或本项目收益。

**部署时信息边界。** 推理只使用 `I[t-k:t]`、当前可用雷达和 ego 变换；future image、future radar、future GT 位移均不可见。若使用预训练 V-JEPA 权重，只能称 frozen feature inference；若训练 adapter，必须记录 adapter 训练数据、参数和未来窗口定义。未来目标的 encoder 不能被误称为部署 teacher。

**为何可能优于简单增容。** 未来 latent target 提供时间对齐和表征级监督，要求预测器保留对下游有用的变化；它改变了信息瓶颈与目标，而不是只增加 head 宽度。它也能主动丢掉逐像素不可预测噪声，可能减少当前“recall 上升但 FP 增加”的不稳定校准。这个判断是机制假设，不是本项目结果。

**否定实验。** 在同一 O encoder/readout 容量和完全相同历史前缀下，比较 (i) 当前 latent 直接外推、(ii) masked future-latent predictor、(iii) 仅增容 head；固定阈值并在未参与设计的 scene 上测 Future GMO、moving recall 和 stationary FP；只有配置并认证 metric motion head 后才测该候选的 EPE。若 (ii) 不能超过 (i)/(iii)，或只提升 latent loss 而 GMO/moving recall 不升，则淘汰这个具体实现，不否定外部表征或全部 latent 路线。

## 竞争路线 B：任务充分、decoder-free 的 latent dynamics

**forward 路径。** 当前融合观测编码为 `z_t = E(I_t, radar_t, ego_t)`；动力学模块执行 `z_{t+1}=f(z_t, c_t, Δt)`，其中 `c_t` 可包含 ego motion、时间间隔和经过可靠性门控的传感器变化，但雷达测量不定义为 agent action，也不能把每个雷达速度强行当作实体速度。未来 `z` 直接接 occupancy/value-style heads；训练用 latent consistency / TD-style target 和未来 occupancy CE，禁止依赖像素重建作为唯一约束。部署时递归 rollout `f`，每一步只用起点状态和已知 ego/time/协议允许的条件，不读取未来观测。

TD-MPC2 官方 [`world_model.py` lines 25–30](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py#L25-L30) 同时定义 encoder、dynamics、reward、termination、policy prior 和 Q ensemble；[`lines 103–121`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py#L103-L121) 明确 `encode(obs)` 与 `next(z,a,task)` 的 latent 接口；[`lines 186–207`](https://github.com/nicklashansen/tdmpc2/blob/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py#L186-L207) 由 latent/action 读取 Q。论文的“decoder-free”是其控制世界模型事实，不能直接改写成 occupancy 事实。

**部署时信息边界。** 只允许初始历史观测、ego 运动、时间间隔和明确列入协议的 sensor token。没有未来相机/雷达时，缺失项保持缺失并由 dynamics 处理；不能以未来 teacher 或 GT 位移初始化 rollout。

**为何可能优于简单增容。** 若 GMO 只需一个任务充分状态，显式保存所有几何细节会让 rollout 累积误差和遮挡不确定性变成负担。这里的 decoder-free 只表示 latent dynamics 没有观测重建 decoder；occupancy 任务仍然需要自己的 occupancy decoder/head。它可能在 2 s 长时域更稳，且把“可用于任务的信息”与“看起来像场景的细节”分离。这说明持续实体场景状态的必要性尚未确立；控制任务中的成功不是驾驶占据上的直接反例。

**否定实验。** 在相同参数预算下做三臂对照：O、显式场/实体状态、decoder-free latent dynamics；对实际存在的条件 `c_t` 做 ego-only、速度置零、错时刻和随机置换干预。若 latent dynamics 对速度置零/错时刻不敏感、长期 IoU 下降更快，或 moving recall 的增益伴随更差 stationary FP，则暂不晋级；速度路由的物理收益—代价对照只用于具备认证运动读出的候选，不能与 occupancy 指标跨单位比较。

## 竞争路线 C：离散 token 的序列模型与显式 rollout 校验

**forward 路径。** 观测 encoder/tokenizer 将每个时刻的 BEV/多模态融合结果变成离散 token `o_t`; 将合法的 action/ego token 插入序列；causal Transformer 根据历史生成下一时刻 observation token，并并行预测 occupancy-related target、时间终止/有效性 head。长期 rollout 使用 KV cache 递归生成，直到固定 2 s 或协议终止。训练是 teacher-forced token CE 加未来占据 CE；推理不使用真实未来 token。

IRIS 官方 [`world_model.py` lines 25–74](https://github.com/eloialonso/iris/blob/24326aaaa283c527f42b89b44cfdecf2665a7a16/src/models/world_model.py#L25-L74) 实际定义 observation/reward/end heads 和 token embedding；[`lines 81–95`](https://github.com/eloialonso/iris/blob/24326aaaa283c527f42b89b44cfdecf2665a7a16/src/models/world_model.py#L81-L95) 执行 causal Transformer 与三个输出；[`lines 97–114`](https://github.com/eloialonso/iris/blob/24326aaaa283c527f42b89b44cfdecf2665a7a16/src/models/world_model.py#L97-L114) 用 tokenizer 编码 observation、拼 action token 并计算 observation/reward/end CE。README 还描述 pretrained checkpoints 和 imagined play，但这些 checkpoint 是 Atari 环境证据。

**部署时信息边界。** 初始历史 token、ego/action/time token 可用；未来观测 token 不可用。必须限制 context 长度并报告 KV cache/重置策略，因为仓库 README 的 20-frame memory flush 是工程事实，不能当作无限长期记忆。

**为何可能优于简单增容。** 离散化把预测问题变成可审计的序列概率，能显式测 token NLL、错误累积和 horizon collapse；causal history 也可能比给每个空间格加容量更有效地保留时间依赖。它是“长期建模接口”竞争者，不是自动的 3D 几何建模器。

**否定实验。** 训练同预算 token predictor，并按 0.5/1/1.5/2 s 逐步评估 token accuracy、occupancy IoU、moving/static recall/FP；同时做 scheduled sampling 与 teacher forcing 对照。若 token 指标改善但 occupancy 不改善，或 1 s 后 rollout 崩溃并低于 O/简单持久化，则淘汰。

## 竞争路线 D：让 dynamics 对状态变化具有可迁移的结构约束

**forward 路径。** 在 latent `z_t` 上预测条件变化 `δ̂_t = g(z_t, c_t, Δt)`，得到 `ẑ_{t+1}=z_t+δ̂_t`；`c_t` 中的 ego motion/time 可作为外生条件，雷达测量不能直接冒充 agent action。除未来 occupancy loss 外加入 PLSM 风格的正则，使相近状态下同一**控制条件**的 change 更接近：可实现为 batch 内 `I(z_t; δ̂_t | c_t)` 的可计算上界/对比近似，或分桶后的 change variance penalty。这个正则只作用于 dynamics，不假设 slot 是实体，也不需要未来观测部署。

PLSM 论文明确给出残差形式 `z̃_{t+1}=z_t+Δ̃_t^{a_t}`，并以降低 latent state 与 action effect change 的互信息为目标；正式状态和全文见 [NeurIPS 2024 页面](https://papers.nips.cc/paper_files/paper/2024/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html) 和 [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/43ba0466af2b1ac76aa85d8fbec714e3-Paper-Conference.pdf)。论文结果涉及长 horizon latent prediction 和 downstream control，不能当驾驶 occupancy 结果。

**部署时信息边界。** `z_0` 来自当前历史；后续只用 ego/time 等已定义的外生条件。雷达速度是对交通参与者的测量，不是 agent action；即使作为额外条件输入，也不能假设 PLSM 的“同一 action 效果”对其他交通参与者成立。不得把正则化后的 latent change 宣称为物理速度或 EPE。

**为何可能优于简单增容。** 当前 Cpl/Fix 表明重新接入共享刚体状态会共同增益但高速召回下降；这条路线直接约束“变化如何依赖状态”，在 ego-controlled 或可验证控制条件上可能减少 latent 漂移，而不增加几何模块容量。对交通参与者运动的适用性没有证据，故将其降为低优先级结构诊断，而非主方案。

**否定实验。** 在相同 encoder/dynamics/head 下比较无正则和 PLSM 正则，固定 action/time，按 horizon 测 latent drift、Future GMO、moving recall、stationary FP。若正则只降低 latent loss 却增加静止误运动，则不晋级；与 O 比较 occupancy，与速度路由比较认证的物理运动输出，不构造跨指标单位的 Pareto 结论。

## 综合判断与建议顺序

train16 结果不支持立即蒸馏“真实未来 logits”，但仍留下两条强反方假设：一是预测**任务可读的未来 latent**比预测几何实体更合适；二是任务所需状态可能是低维且 decoder-free，持续几何不是必要条件。IRIS 和 TD-MPC2 说明长期 rollout 的关键可分别落在序列概率或任务 head；V-JEPA 说明冻结 target/predictor 的监督接口是可实现的；PLSM 说明结构化 dynamics 可能比增容更重要。

建议只做一个小型分叉评估，而不是同时重训四套系统：复用已完成的 O teacher/readout 诊断，仅针对未解决的校准或目标问题补充必要检查，然后比较同预算的任务 latent dynamics；只有长期漂移是主要失败模式时，才加 PLSM 风格 change regularizer。离散 token 路线作为第三候选，优先做短序列可行性和 rollout 崩溃审计。

所有“可能优于”都只是文献机制到本项目接口的迁移假设。latent 候选没有经过独立认证的 metric motion readout，就不能声称改善 EPE；同一 O encoder 上的 probe 也不能证明外部 DINO/JEPA 表示无效。正式晋级仍须满足原 Future GMO、moving recall、stationary 代价、源人口 EPE 和未参与设计 scene 的匹配证据；任何论文的控制分数、冻结分类分数、README、checkpoint 或仓库可运行性，都不能替代这些证据。
