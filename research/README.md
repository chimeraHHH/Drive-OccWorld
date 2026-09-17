# RadarFlowOcc research checkpoint — 2026-09-17

本分支记录代码、实验协议、已核验结果和失败路线。研究目标仍是超过保留模型 O，并提高真实运动预测准确性；**目前尚未达到该目标**。单 seed 11；外部模型只使用官方冻结权重。历史 M3/S3 文件仅为研究记录，不表示恢复训练或重新支持其机制主张。

## 当前结论

| 证据 | 结果与边界 |
|---|---|
| 原协议完整 5,119 样本 | O future GMO IoU 14.7478%，M0 13.9449%；这是二分类占据指标，不是语义 mIoU。O 的改善尚不能解释为物理运动更准。 |
| 独立 Metric3D + RAFT，train16 / 8 scenes | 原评分人口上静止误运动明显，未胜过已有 CV+D；保留失败结果。 |
| 官方两帧 DELTA，train16 / 8 scenes | 匹配网格、当前深度和评分人口后，优于匹配 RAFT，但仍未胜过 CV+D。96 对推理完成、零优化更新，三组评分独立复算。 |
| 自有未来状态，train4 工程 fit | 96 次更新全部完成，确认任务连接；该短训中的接近零运动已由下面更长学习曲线进一步诊断。 |
| 固定 codec 的 train4 学习曲线 | P/J/D 各 512 次更新完成。moving 2 s EPE 0.676/2.121/1.774 m；future IoU 2.863/22.093/20.233%。J 占据优于 D，但运动更差；没有 held-out 或超过 O 的证据。 |
| 密集历史 DELTA，train16 / 8 scenes | 同一起止时刻增加为 5–7 帧，96 条序列均完成推理与独立几何/评分核验。优于两帧 DELTA，仍在四个 horizon 的 moving/stationary 上落后于 CV+D；停止继续扩展外部三维速度路线。 |

最新物理对照，2 s XY EPE，单位 m、对象等权、越低越好：

| 方法 | moving（137 个对象） | stationary（184 个对象） |
|---|---:|---:|
| CRN-CV | 3.022346 | 0.092182 |
| CV+D，固定 0.5 m/s 规则 | **2.737708** | 0.101027 |
| CV + 匹配 RAFT，覆盖外替换、FB 筛选 | 3.294486 | 0.390325 |
| CV + 两帧 DELTA，覆盖外替换、官方 vis | 2.997021 | 0.266921 |
| CV + 密集 DELTA，端点差分、端点 vis | 2.898934 | 0.188692 |
| CV + 密集 DELTA，LSQ、全帧 vis | 2.893744 | 0.188366 |

这些 EPE 使用 GT 定义的虚拟材料点评价固定的输入侧运动场，不能称为 O 的运动读出，也不构成 held-out 泛化或新占据收益。两种质量筛选并不等价；完整报告同时保留不筛选比较。当前 train16/dev200 已反复用于开发，必须另行锁定独立确认场景。

## 从这里阅读

- [用户新增方向：用雷达动态先验减少视觉历史与计算](analysis/o_motion_20260915/雷达动态先验与视觉历史效率_实验计划.md)。已加入后续 roadmap，包含真实图像历史审计、Doppler 消融、流式推理成本与 CRISP/TEOcc 近邻；目前是待验证假设。
- [最新学习曲线结果与决策](analysis/o_motion_20260915/dense_state_learning_curve_v1_结果与决策.md)；[真实曲线图](analysis/o_motion_20260915/dense_state_learning_curve_evaluation_v1/learning_curves.png)；[648 指标复算](analysis/o_motion_20260915/dense_state_learning_curve_evaluation_v1/aggregate_audit.json)。
- [固定 codec 的物理单任务/联合任务学习曲线设计](analysis/o_motion_20260915/research_notes/dense_state_learning_curve_design_v1.md)；[冻结协议](analysis/o_motion_20260915/dense_state_learning_curve_protocol_v1.json)。本轮仍只用既定 train4 做可学习性诊断。
- [O 验证模型卡](analysis/m0_improvement_20260915/O验证通过模型卡.md)；[完整评价汇总](analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1/metrics.csv)。
- [最新密集历史结果与停止决策](analysis/o_motion_20260915/delta_dense_history_train16_结果与研究决策.md)；[几何独立复算](analysis/o_motion_20260915/delta_dense_history_geometry_audit_v1.json)；[LSQ 聚合评分](analysis/o_motion_20260915/delta_dense_lsq_train16_evaluation_v1/summary.csv)。
- [两帧 DELTA 结果与决策](analysis/o_motion_20260915/delta_history_train16_结果与研究决策.md)；[聚合评分](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/summary.csv)；[独立复算](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/independent_score_audit.json)。
- [Metric3D + RAFT 失败结果](analysis/o_motion_20260915/metric_surface_history_train16_结果与研究决策.md)。
- [世界模型文献与官方仓库综合判断](analysis/o_motion_20260915/research_notes/world_model_research_direction_v3_20260917.md)。这是阶段性调研；其中“尚未运行”反映该报告写作时点，后续 DELTA 运行状态以上述结果报告为准。
- [任务直接使用的未来状态：设计、文献依据与局限](analysis/o_motion_20260915/research_notes/task_native_state_design_20260917.md)；[新状态实现](analysis/o_motion_20260915/dense_task_state_v2.py)；[训练集小样本 fit 协议](analysis/o_motion_20260915/dense_task_state_fit_protocol_v2.json)。v2 已完成真实 train4 工程 fit、位移干预与全网格评分；[结果与下一步](analysis/o_motion_20260915/dense_task_state_fit_v2_结果与决策.md)。任务连接通过；该 32 更新阶段不能替代后续 512 更新的可学习性诊断，没有超过 O 的证据。
- [密集输入认证](analysis/o_motion_20260915/server_results/diagnostics/dense_history_inputs_train16_v1/complete.json)；[已运行的密集 DELTA 实现](analysis/o_motion_20260915/extract_delta_dense_history_v1.py)。

密集历史比较已完成：增加输入观测有收益，但不足以胜过现有强对照。停止继续扩展外部三维速度路线，保留 O/M0 与冻结诊断输出，已完成自有未来状态与 identity readout 的小样本工程比较。512 更新学习曲线已确认小样本运动可学习，但联合占据收益与物理准确性并不一致。固定最终 checkpoint 的 16 组内容/位移交叉读出也已完成：P 位移传播当前内容有收益，P 训练后内容却使结果退化；J 依赖内容—位移配合，D 内容直接读出更好。下一步先明确持久内容与运动迁移的分工，再冻结较大数据协议；后续同时验证 Doppler 是否能减少真实视觉历史和系统成本。

## 归档与复现边界

`analysis/` 保留原相对目录和字节内容；[snapshot_manifest.json](snapshot_manifest.json) 记录每个源文件和 SHA256。历史版本、草稿、已失败实现同时保留，不能从文件存在推断它已运行成功。根项目补入 P2 的模型接口及其模块/测试，是已有实验代码的归档，不表示本次重新训练验证。

数据集、权重、逐点预测、原始目标、外部仓库镜像、SSH 配置和服务器环境不在此次提交中。报告中的部分原始证据链接因此只在完整本地工作区存在。代码保留原实验路径与资产哈希；在新环境运行需取得许可的数据、官方权重并配置实际路径。此分支不是开箱即用的模型发布。

后续每个有意义的实验完成、失败或方向决策节点，更新结果及此索引，运行 `tools/sync_research_snapshot.py --workspace /path/to/dropple`，审查差异后 commit/push。该工具只复制白名单文件，不启动任务、不自动提交。
