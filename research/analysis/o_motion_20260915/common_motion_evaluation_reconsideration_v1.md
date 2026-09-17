# 五帧 binary GMO 的共同运动评价：真实 QA 后的修订建议

2026-09-15。**建议停止把组件质心位移作为 O 的共同 motion-EPE 代理；保留完整 GMO 评价，优先补充 GT-only 运动分层风险与占据新增/腾空评价。** 两者都直接评价双方的最终占据输出，不需要为 O 人造一个弱 flow。它们可以支持“运动对象的未来占据与空间变化更准确”，不能单独证明实例位移或身份运动更准确。本文件是待冻结的评价建议；没有新增训练、GPU 作业或候选成绩，也没有修改原合同、GT 或预测。

## 1. 真实 QA：组件不是实例，不能靠改阈值挽救

同一冻结 train-cache 原序前 16 anchors、8 scenes；80 个 O 原生 fine-GT confusion 与旧 oracle 逐行精确一致，80 个 GT→自身 GOSPA 为零。下表由完整 `records.jsonl` 独立 CPU 汇总，计数单位为 **anchor×instance 出现次数**，不是独立物体数量。输入、分母、互斥失败分类与 SHA 见 [数值记录](common_motion_evaluation_reconsideration_v1_qa.json)，原始结果见 [QA summary](server_results/probes/common_output_readout_train16_v1/summary.json)。

| 时域 s | GT 前景体素 | 唯一框归属比例 | GT/O 组件数 | GT 组件面积中位数（BEV 格） | 多组件实例数 | t0/h 两端完整纯组件可用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 147,406 | 95.66% | 1,029 / 257 | 3 | 153 | — |
| 0.5 | 149,080 | 94.55% | 1,058 / 211 | 3 | 146 | 10/436 = 2.29% |
| 1.0 | 148,033 | 94.16% | 1,068 / 177 | 3 | 150 | 3/427 = 0.70% |
| 1.5 | 143,935 | 93.80% | 1,067 / 151 | 3 | 160 | 8/416 = 1.92% |
| 2.0 | 143,209 | 93.90% | 1,174 / 121 | 2 | 168 | 8/415 = 1.93% |

这里的“完整纯”只指一个纯组件包含该框内全部被标注的有效前景，并排除框内重叠歧义，不指完整物体表面。在 2 s 的 415 个两端框均存在的实例对中，按固定优先顺序分解：8 个通过、66 个至少一端无 GT 前景、168 个有前景但碎成多组件、3 个无碎片但有重叠、170 个仅关联单组件却仍不满足完整纯净。0.5–1.5 s 同样有 175–193 个最后一类。

**因此，证据支持“稀疏表面组件→实例的假设失效”，不能说碎片化是唯一主因，也不能说 GT/框存在整体坐标错位。** 约 94%–96% 的体素能唯一归框，反对把问题简单归为整体错位；但少量框外体素、相邻实例、BEV 投影连接和标注边界差异足以使整个组件失去纯净资格。当前统计不能区分剩余错位、标注误差和表面不完整各占多少。不得据此改变 purity、closing、面积门槛或只保留成功端点；这种低覆盖代理不适合决定是否“运动超过 O”。

## 2. 一手论文与作者实现实际测量了什么

以下均阅读了方法/评价正文及具体实现，非只读摘要。固定仓库版本见 [来源清单](common_motion_evaluation_reconsideration_v1_sources.json)。

**Waymo Occupancy Flow Fields，RA-L 2022。** 其占据是代理对象的 BEV 栅格，区分 observed/occluded；soft-IoU 为 `Σ(py)/(Σp+Σy−Σpy)`，官方 PR-AUC 使用 100 个阈值。EPE 必须输入预测二维 flow；当前实现仅在 GT flow 两坐标至少一个非零的格上平均欧氏误差，不能照搬为包含静止目标的 EPE。Flow-warped 指标用预测 backward flow 回采 GT origin occupancy，再乘预测未来 occupancy，仍需要 flow。空 GT waypoint 会被跳过，评价域和我们完整稀疏 GMO 不同。所核 commit 的 `_flow_warp` 已因依赖废弃而直接抛 `NotImplementedError`；不能把当前 master 当可即插即用实现。[论文](https://arxiv.org/abs/2203.03875)、[作者指标代码，L155–308](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/utils/occupancy_flow_metrics.py#L155)。

**Cam4DOcc，CVPR 2024。** 未来 GMO IoU 是各时域 `TP/(TP+FP+FN)` 的平均；GMO 指可移动类别，并不表示该车或行人实际在运动。其 GT 预处理包含未来新生/可见性条件及缺失时的运动补偿，不能把这些条件追加入我们的评分 mask。其 backward centripetal flow 把体素关联到先前实例中心，并非物理表面点的真实 scene-flow；VPQ 则要求实例分割/关联。作者 `fast_hist` 是 GT 有效域上的几何混淆，没有从 binary GMO 自动获得的 motion-EPE。[论文 §III 与 flow 附录](https://arxiv.org/html/2311.17663v3)、[作者 metric_util.py](https://github.com/haomo-ai/Cam4DOcc/blob/542f14a9d9e142d9faf9044df47c8488a7c3166a/projects/occ_plugin/utils/metric_util.py)。

**OccWorld，ECCV 2024。** 占据评价用语义 mIoU 与几何 occupied-IoU，报告未来时域平均；规划 L2/碰撞针对自车轨迹，不是其他对象的运动误差。作者 `multi_step_MeanIou` 按每时域累积 TP、GT 数和预测数，忽略指定 ignore-label；这仍然只是未来集合重叠，无法判断同一实例去了哪里。其空 GT 类处理也不是我们既定的全域分母规则。[论文 §4.1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02024.pdf)、[作者 multi_step_MeanIou，L56–110](https://github.com/wzzheng/OccWorld/blob/1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e/utils/metric_util.py#L56)。

**UniOcc: A Unified Benchmark…，ICCV 2025（不是 2023 年同名方法）。** 除 IoU 外加入物体形状合理性、时间形状一致性与静态背景一致性。公开参考实现的时间接口显式要求 `flows`，关联后先移除质心并 PCA 对齐再算形状 IoU；所以物体在错误位置保持同形状也可能很好，不是位移正确性。背景一致性只比较变换后的重叠视域，亦不能替代与 GT 的完整 FP/FN。参考代码还在 `.items()` 二元组上检查 `len<3`，该路径会跳过形状；这是所核公开示例的静态缺陷，不能由此推断论文实验失效，更不能直接当成熟 flow-free 指标复用。[论文 §3.5](https://arxiv.org/html/2503.24381v2)、[作者接口与实现，L133–191](https://github.com/tasl-lab/UniOcc/blob/f627c020149e266947cd40cd1b1aac7d95ab841f/uniocc_eval.py#L133)。

**Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting，CVPR 2023。** 对给定未来射线评价深度 `mean|d−d̂|`、`mean(|d−d̂|/d)`；对射线还原点集评价双向最近邻平方距离平均，即 `CD=½(mean_P min_G ||p−g||² + mean_G min_P ||g−p||²)`。最近邻没有身份约束，不能变成 flow-EPE；重复/密度变化也不具有逐目标基数惩罚。作者 evalkit 对近场两集合都裁 ROI，任一为空竟返回 0，不能复制这个漏失漏洞；query 脚本还每 5 根射线抽一根，非全体素评价。我们 binary GMO 的 0 类同时包含非 GMO/空域，不能以 GMO 的首命中去对照原 LiDAR 的首物理回波。[论文 §4](https://openaccess.thecvf.com/content/CVPR2023/papers/Khurana_Point_Cloud_Forecasting_as_a_Proxy_for_4D_Occupancy_Forecasting_CVPR_2023_paper.pdf)、[作者评价代码](https://github.com/tarashakhurana/4d-occ-forecasting/blob/e1ea9434113d303986684395a961aaacac90d470/cvpr23-evalkit/evaluate.py)、[射线采样代码](https://github.com/tarashakhurana/4d-occ-forecasting/blob/e1ea9434113d303986684395a961aaacac90d470/cvpr23-evalkit/generate_query_rays.py#L48)。

## 3. 优先实现的一个共同指标套餐

**A. 保持全部原生成绩及假阳性账目。** 原 `512×512×40`、固定 t0 R、五帧输出、原 logits 插值→argmax、每帧有效 GT 域均不变；主分数仍为既定的逐时域全样本 pooled confusion→四未来 IoU 平均，t0 单列。同报完整 FP/FN、precision/recall。若双方保留最终 logits，可附完整自然 NLL/Brier；若输入只有硬 binary，不能捏造概率算 NLL。新指标不改变原 GMO 超过 O 的门槛。

**B. 对 GT 前景做运动条件风险分解，不用框裁剪预测。** 当前帧每个有效前景体素，只依据该帧原始框做唯一归属；未知/重叠保留。复用已在训练标签分布分析中固定的 global-xy 端点位移/真实 dt 分层：`[0,0.1]、(0.1,0.5]、(0.5,5]、>5 m/s`，全部报告，不据新预测改阈值。低速组称“低端点运动”，不直接称物理静止；端点差不能识别往返、纯转动与小标注抖动。原始属性可另作有来源的描述，不能偷偷覆盖矛盾。未来新生无 t0 对应、缺失框、未归属体素保留 unknown 类，不填零速度。

令 `V` 为完整有效域，GT 正类按速度组/未知划分为 `D₁…D₄,U`，全部 GT 负类为 `N`。逐 voxel 风险满足精确分解：

`R = [Σ_k Σ_(x∈Dk) ℓ(p(x),1) + Σ_(x∈U) ℓ(p(x),1) + Σ_(x∈N) ℓ(p(x),0)] / |V|`。

保存每项总损失和计数，同时报告分组 FN/前景召回；**N 中所有假阳性一次不漏**。不能给每个 FP 强行贴 moving/static 标签，不能把 GT 动态框外 FP 删除后称“动态 IoU”。可附实例等权 GT-support recall：每个有 GT 支持的实例均进分母，即使预测完全为空；没有 GT 前景的实例记不可评并计数，不视为正确。此包衡量运动对象的未来占据风险，仍非位移误差。

**C. 占据新增/腾空评价，检查静止残影。** 在固定 `V₀∩Vh` 上，GT 和预测都从原 binary 输出直接形成四状态 `(b₀,bh)∈{00,01,10,11}`，累计完整 `4×4` confusion，报告新增 `01` 和腾空 `10` 的 IoU/precision/recall，其他状态预测成该状态均算 FP。空预测、静止复制、假新增不能因缺少可匹配组件被略去。该交集仅用于此附加指标，覆盖数量必须报告，A 的全域评价仍不变；若候选应保留 O 的 t0，先核 t0 exact，以免初始误差混入。它是拟议的简单 occupancy-change 诊断，**不是已有官方 flow 指标，也不包装成新方法**；遮挡/稀疏标注变化亦会改变四状态，不能单独推出物理运动。仅有五帧边缘概率也不能假设 `p₀p_h` 就是校准的联合概率。

先冻结上述公式、速度来源/分层、未知与空组规则，再评同一批 O/候选；逐样本/时域保存充分统计，场景配对 bootstrap 沿已有 seed11/10,000 次规则，并披露开发集曝光。GT 自比较、全空预测、复制 t0、添加 FP、正确整数平移是必要解析检查；它们是指标程序验收，不是模型成果。此方案可在每次现有 full-GT 评价后附加计数，不需要 tracker、组件关联或新 GPU 训练。

## 4. 不能识别的目标仍需单独面对

两个同形物体交换身份，与保持原身份的占据集合可以在所有五帧完全相同；对称物体旋转也可能不改变占据。相同五帧输入的确定性评分函数必给相同分数，所以 **无 ID/flow 的占据边缘不能一般性识别真实对应、scene flow 或轨迹身份**。这不是换一个几何距离就能解决的问题。

Chamfer、体素质量上的非平衡运输距离或 GOSPA 可以补充位置与漏失/多余质量成本，但仍只测未来几何；非平衡运输还需冻结质量单位、截断/删除成本与稀疏计算方式。现有 component-GOSPA 保留为 QA，不再推进质心 EPE；不追加一组易被碎片密度左右的主分数。

若后续必须比较 flow，可为**每个冻结模型**追加同结构的输出序列读出器，限定只读相同五帧最终 occupancy/logits，采用相同 train-only 标签、seed11、初始化、样本顺序、步数、损失、容量与预算；评估集不用于读出器选择。所得结论是“给定同等读出训练后的运动可解码性”，不是 O 原生 EPE。还要并报整个占据域 FP/FN、flow 标签有效支持和静止/移动分层，避免只评价成功检测部分。若读出器借用新模型内部 flow 或 O 不可得 latent，比较即不公平。本轮不启动该训练。

目标仍是同时超过固定 O 的占据能力和改善运动：共同套餐负责可观测的未来几何/变化，独立 annotation-derived motion-head 误差负责显式运动监督证据；只有新增公平读出或可比较原生 flow 后，才允许声称在共同 EPE 上超过 O。
