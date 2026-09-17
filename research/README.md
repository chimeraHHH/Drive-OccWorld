# RadarFlowOcc research checkpoint — 2026-09-17

本分支记录代码、实验协议、已核验结果和失败路线。研究目标仍是超过保留模型 O，并提高真实运动预测准确性；**目前尚未达到该目标**。单 seed11；外部模型只使用官方冻结权重。旧 M3/S3 不恢复。

## 最新工作报告

[中文 PDF（9页）](reports/work_report_20260917/main.pdf) · [可编辑 LaTeX](reports/work_report_20260917/main.tex) · [报告数据与编译说明](reports/work_report_20260917/README.md)

报告覆盖9月14日至17日：O 强对照、运动连接、冻结视觉运动证据、完整 T/J/D、T 训练集拟合、文献与少视觉历史计划。所有图表为真实数据，未完成实验以红字标记；报告包有独立的来源清单。

## 当前结论

**train512 的 T/J/D 已完成且均未达标。** 2026-09-17 12:42:52 UTC 服务器只读检查显示序列 EXITED_ZERO；D 的训练/评价文本随后取回并通过独立复算。[完整结果与阶段决策](analysis/o_motion_20260915/dense_material512_完整结果与阶段决策.md)。本实验族与历史 connected-motion J/D 不同。

| 原 dev200 / 100 scenes | O | T | J | D |
|---|---:|---:|---:|---:|
| 未来 GMO / % | 14.6640 | 9.2876 | 9.0291 | 8.2462 |
| moving recall / % | 43.4521 | 30.2357 | 26.7237 | 21.2271 |
| 2s moving XYZ proxy EPE / m | 无认证原生运动头 | 4.9610 | 5.2058 | 5.2207 |

同源 CRN-CV moving XYZ EPE 为3.4093m；三臂当前保持均为9.5999%。每臂2,048更新，独立复算796项聚合标量。O 与三臂的输入条件、容量和结构不同，不能把分差解释为单一模块的因果效果；不扩大这些失败候选到 full5119 或六条件少帧训练。

固定最终 T 在 train512 的零更新评价也已完成：未来 GMO10.8284%，低于自身保持11.3461%；120,042个分类标量独立核验。失败不只是开发集泛化问题，但尚未证明收敛或当前表示不可逆丢失运动信息。

完整原协议5,119锚点/150场景上，O 为14.7478%，匹配精度 M0 为13.9449%；这是二分类 GMO IoU，不是语义 mIoU，且未证明 O 运动更准确。保留 O/M0。

冻结 Metric3D/RAFT/DELTA 和密集历史诊断已完成；更多历史能改善 DELTA，却仍未胜过既有 CV+D 的运动收益—静止代价取舍。停止继续扩展当前外部三维速度路线，保留所有失败结果。

**Doppler 替代视觉历史是高优先级待验证假设。** [真实输入审计](analysis/o_motion_20260915/视觉历史效率_输入审计与实施边界.md)确认 O 为三个相机时刻、每时刻六视图，雷达每路至多五 sweeps；单槽 BEV 不等于单帧。旧缓存不绑定 sweep 配置，修改 nsweeps 不能认证输入改变。H1 窗口模块和 probe 已编写，**尚无真实 H1 前向通过回执**；H=1/3 × 纯视觉/雷达几何/Doppler 六条件尚未训练，也无端到端效率结论。

## 从这里阅读

- [J 的真实 dev200 结果与决策](analysis/o_motion_20260915/dense_material512_J_结果与研究决策.md)；[独立计数审计](analysis/o_motion_20260915/dense_material512_J_evidence_v1.json)；[T/J 输入及当前输出精确一致](analysis/o_motion_20260915/dense_material512_T_J_boundary_evidence_v1.json)。

- [T 的真实 dev200 结果与决策](analysis/o_motion_20260915/dense_material512_T_结果与研究决策.md)；[独立计数/persistence 审计](analysis/o_motion_20260915/dense_material512_T_evidence_v1.json)；[同源 XYZ CRN-CV 对照](analysis/o_motion_20260915/dense_material512_T_physical_reference_v1.json)。
- [JEPA-WMs 预测状态与读出审查](analysis/o_motion_20260915/research_notes/jepa_wms_predictive_state_audit_20260917.md)；[四文件固定源码身份](analysis/o_motion_20260915/jepa_wms_source_identity_v1.json)。只读论文/源码，没有训练外部模型；机器人 planning 结论不能直接迁移为本项目收益。

- [train512 未来状态训练协议](analysis/o_motion_20260915/dense_material512_training_protocol_v1.json)；[原 O 共同评价协议](analysis/o_motion_20260915/dense_material512_evaluation_protocol_v1.json)；[共同 codec 证据](analysis/o_motion_20260915/shared_current_codec512_evidence_v1.json)。
- [用户新增方向：用雷达动态先验减少视觉历史与计算](analysis/o_motion_20260915/雷达动态先验与视觉历史效率_实验计划.md)。已加入后续 roadmap，包含真实图像历史审计、Doppler 消融、流式推理成本与 CRISP/TEOcc 近邻；目前是待验证假设。
- [少帧输入实施边界](analysis/o_motion_20260915/视觉历史效率_输入审计与实施边界.md)；[712 anchors 历史链](analysis/o_motion_20260915/visual_history_population_evidence_v2.json)；[真实雷达缓存反例](analysis/o_motion_20260915/radar_cache_sweep_identity_evidence_v1.json)；[CRISP history 与效率专项审计](analysis/o_motion_20260915/research_notes/crisp_history_substitution_audit_20260917.md)。
- [最新学习曲线结果与决策](analysis/o_motion_20260915/dense_state_learning_curve_v1_结果与决策.md)；[真实曲线图](analysis/o_motion_20260915/dense_state_learning_curve_evaluation_v1/learning_curves.png)；[648 指标复算](analysis/o_motion_20260915/dense_state_learning_curve_evaluation_v1/aggregate_audit.json)。
- [固定 codec 的物理单任务/联合任务学习曲线设计](analysis/o_motion_20260915/research_notes/dense_state_learning_curve_design_v1.md)；[冻结协议](analysis/o_motion_20260915/dense_state_learning_curve_protocol_v1.json)。本轮仍只用既定 train4 做可学习性诊断。
- [O 验证模型卡](analysis/m0_improvement_20260915/O验证通过模型卡.md)；[完整评价汇总](analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1/metrics.csv)。
- [最新密集历史结果与停止决策](analysis/o_motion_20260915/delta_dense_history_train16_结果与研究决策.md)；[几何独立复算](analysis/o_motion_20260915/delta_dense_history_geometry_audit_v1.json)；[LSQ 聚合评分](analysis/o_motion_20260915/delta_dense_lsq_train16_evaluation_v1/summary.csv)。
- [两帧 DELTA 结果与决策](analysis/o_motion_20260915/delta_history_train16_结果与研究决策.md)；[聚合评分](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/summary.csv)；[独立复算](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/independent_score_audit.json)。
- [Metric3D + RAFT 失败结果](analysis/o_motion_20260915/metric_surface_history_train16_结果与研究决策.md)。
- [世界模型文献与官方仓库综合判断](analysis/o_motion_20260915/research_notes/world_model_research_direction_v3_20260917.md)。这是阶段性调研；其中“尚未运行”反映该报告写作时点，后续 DELTA 运行状态以上述结果报告为准。
- [任务直接使用的未来状态：设计、文献依据与局限](analysis/o_motion_20260915/research_notes/task_native_state_design_20260917.md)；[新状态实现](analysis/o_motion_20260915/dense_task_state_v2.py)；[训练集小样本 fit 协议](analysis/o_motion_20260915/dense_task_state_fit_protocol_v2.json)。v2 已完成真实 train4 工程 fit、位移干预与全网格评分；[结果与下一步](analysis/o_motion_20260915/dense_task_state_fit_v2_结果与决策.md)。任务连接通过；该 32 更新阶段不能替代后续 512 更新的可学习性诊断，没有超过 O 的证据。
- [密集输入认证](analysis/o_motion_20260915/server_results/diagnostics/dense_history_inputs_train16_v1/complete.json)；[已运行的密集 DELTA 实现](analysis/o_motion_20260915/extract_delta_dense_history_v1.py)。

本阶段小样本工程拟合、学习曲线、内容/位移交叉读出及扩大到 train512 的比较均已完成。完整阶段决策以上方最新报告为准；历史报告保留其原写作时点，不从旧的“运行中”文字推断当前状态。

## 归档与复现边界

`analysis/` 保留原相对目录和字节内容；[snapshot_manifest.json](snapshot_manifest.json) 记录每个源文件和 SHA256。历史版本、草稿、已失败实现同时保留，不能从文件存在推断它已运行成功。根项目补入 P2 的模型接口及其模块/测试，是已有实验代码的归档，不表示本次重新训练验证。

数据集、权重、逐点预测、原始目标、外部仓库镜像、SSH 配置和服务器环境不在此次提交中。报告中的部分原始证据链接因此只在完整本地工作区存在。代码保留原实验路径与资产哈希；在新环境运行需取得许可的数据、官方权重并配置实际路径。此分支不是开箱即用的模型发布。

后续每个有意义的实验完成、失败或方向决策节点，更新结果及此索引，运行 `tools/sync_research_snapshot.py --workspace /path/to/dropple`，审查差异后 commit/push。该工具只复制白名单文件，不启动任务、不自动提交。
