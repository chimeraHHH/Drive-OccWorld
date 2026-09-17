# MoGaF：未来预测与当前 Cpl/Fix 的重合及边界

结论：应作为对象分组、刚体运动约束和未来几何生成的相关先例，不能借其渲染成绩证明本项目的 GMO 或完整材料点 EPE，也不能据此承诺当前候选有效。核查固定 v1 原文与官方项目，未运行模型。

## 原文事实卡

以下英文压缩卡只保留核验所需事实，避免长篇复述单一来源。

MoGaF fits 4DGS per scene using the first 60%/80% of frames. Reconstruction combines rendered RGB/depth/mask losses with rigid-group anchoring or nonrigid smoothness. Forecasting independently trains one forecaster per group on optimized Gaussian trajectories: per-Gaussian translation/quaternion inputs, masked temporal context, trajectory reconstruction plus acceleration regularization, then autoregressive continuation. This description does not establish one strictly shared predicted SE(3) per rigid group.

Future-image metrics and point tracking are separate evaluations. Table3(a) jointly removes group optimization and forecasting; Table6 separates them. For the one-layer forecaster, EPE is .250 with neither, .261 with optimization alone, and .220 with both: the optimization-only result does not support uniformly improved EPE.

Grouping seeds front Gaussians; B.2 reassigns only the nearest half of remaining unassigned Gaussians. The limitation identifies insufficiently observed geometry and missing inter-group interaction. Reported tracking includes occlusion accuracy, but the inspected text does not establish our complete-grid/object-support coverage contract.

对应已读位置：[§4.3，式10–12](https://arxiv.org/html/2602.21668v1#S4.SS3)、[B.4，式31](https://arxiv.org/html/2602.21668v1#S2.SS4)、[B.3，式27–30](https://arxiv.org/html/2602.21668v1#S2.SS3a)、[§5.2](https://arxiv.org/html/2602.21668v1#S5.SS2)、[§5.3](https://arxiv.org/html/2602.21668v1#S5.SS3)、[C.3/Table6](https://arxiv.org/html/2602.21668v1#S3.SS3)、[B.2](https://arxiv.org/html/2602.21668v1#S2.SS2a)、[Limitations](https://arxiv.org/html/2602.21668v1#S6.SS0.SSS0.Px1)。

## 代码核验边界

2026-09-16 实际打开的[作者项目页](https://slime0519.github.io/mogaf/)标注 CVPR 2026；Code (Coming Soon!) 的真实 href 就是项目页自身，没有指向实现仓库。因而本次**无可锁定的官方实现 commit、无已核权重或代码许可证**，也没有拿 SoM/Gaga 的代码替代 MoGaF。预测阶段是否反传重建参数、四元数归一化、残留未分组 Gaussian 的预测路径、tracking visibility/query mask 等仍缺可执行实现核验；不能把论文公式升级为逐代码证明。

## 对本项目的判断（本地设计分析）

- 当前候选跨训练场景共享参数，当前输入是官方 CRN 预测状态与观测 BEV；原始 box-derived displacement 只用于监督。我们用同一对象残差位姿定义完整材料点场与未来空间 Gaussian 地址，并保留 O 稠密分支。这里必须分别回答状态误差与占据误差，不能把一个图像质量指标当双目标证据。
- 本轮 Cpl/Fix 的可识别问题是：**在同 CV 基底、相同 learned values/物理场及容量下，把 learned pose 接入未来空间地址是否有额外收益。** Fix 的值路径仍可向 pose 反传；Cpl/Fix 差异不是“有无运动监督”，也不是“有无对象模块”。这才是当前对照能回答的范围，不扩张为通用对象表示优于非对象表示。
- 我们的完整 GT occupancy 包含未覆盖/未知归属体素；物理分母保留原全部合法源点，包括 CRN 未覆盖点，后者预测为零。不能为接近可视化效果而改成只统计检测成功、可见或成功重建对象。遮挡准确率、渲染质量、源点 EPE 的事件空间不同，须维持原分母逐项报告。
- 若物理 EPE 改善但 GMO/运动组风险未改善，只支持运动状态学习；若 GMO 改善而 EPE 无增益，只支持任务利用或补偿。需要同时看相对 CRN-CV/D、相对 O 以及 Cpl−Fix，且披露 t0、全部 FP/FN。文献重合不替代本轮实际证据，也不新增本轮晋级门。

不应主张“首次用对象 SE(3)/Gaussian 预测未来”“刚体约束保证物理正确”或仅靠组合模块宣布新颖性。尚未公开实现的未知点属于核验缺口，不能充当本项目创新证据。

来源与网页响应 SHA、精确章节 URL、未核项记录于同目录 `mogaf_forecasting_boundary_v1_sources.json`。当前冻结实验与协议均未修改。
