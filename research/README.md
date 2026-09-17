# RadarFlowOcc research checkpoint — 2026-09-17

本分支记录代码、实验协议、已核验结果和失败路线。研究目标仍是超过保留模型 O，并提高真实运动预测准确性；**目前尚未达到该目标**。单 seed 11；外部模型只使用官方冻结权重。历史 M3/S3 文件仅为研究记录，不表示恢复训练或重新支持其机制主张。

## 当前结论

| 证据 | 结果与边界 |
|---|---|
| 原协议完整 5,119 样本 | O future GMO IoU 14.7478%，M0 13.9449%；这是二分类占据指标，不是语义 mIoU。O 的改善尚不能解释为物理运动更准。 |
| 独立 Metric3D + RAFT，train16 / 8 scenes | 原评分人口上静止误运动明显，未胜过已有 CV+D；保留失败结果。 |
| 官方两帧 DELTA，train16 / 8 scenes | 匹配网格、当前深度和评分人口后，优于匹配 RAFT，但仍未胜过 CV+D。96 对推理完成、零优化更新，三组评分独立复算。 |
| 相同起止时刻的密集历史输入 | 已认证 16 个锚点、96 条相机序列、654 张不同图像，每条 5–7 帧；读取不到未来目标。这里只完成输入认证，密集 DELTA 推理和评分尚未完成。 |

最新物理对照，2 s XY EPE，单位 m、对象等权、越低越好：

| 方法 | moving（137 个对象） | stationary（184 个对象） |
|---|---:|---:|
| CRN-CV | 3.022346 | 0.092182 |
| CV+D，固定 0.5 m/s 规则 | **2.737708** | 0.101027 |
| CV + 匹配 RAFT，覆盖外替换、FB 筛选 | 3.294486 | 0.390325 |
| CV + 两帧 DELTA，覆盖外替换、官方 vis | 2.997021 | 0.266921 |

这些 EPE 使用 GT 定义的虚拟材料点评价固定的输入侧运动场，不能称为 O 的运动读出，也不构成 held-out 泛化或新占据收益。两种质量筛选并不等价；完整报告同时保留不筛选比较。当前 train16/dev200 已反复用于开发，必须另行锁定独立确认场景。

## 从这里阅读

- [O 验证模型卡](analysis/m0_improvement_20260915/O验证通过模型卡.md)；[完整评价汇总](analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1/metrics.csv)。
- [两帧 DELTA 结果与决策](analysis/o_motion_20260915/delta_history_train16_结果与研究决策.md)；[聚合评分](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/summary.csv)；[独立复算](analysis/o_motion_20260915/delta_history_train16_evaluation_v1/independent_score_audit.json)。
- [Metric3D + RAFT 失败结果](analysis/o_motion_20260915/metric_surface_history_train16_结果与研究决策.md)。
- [世界模型文献与官方仓库综合判断](analysis/o_motion_20260915/research_notes/world_model_research_direction_v3_20260917.md)。这是阶段性调研；其中“尚未运行”反映该报告写作时点，后续 DELTA 运行状态以上述结果报告为准。
- [密集输入认证](analysis/o_motion_20260915/server_results/diagnostics/dense_history_inputs_train16_v1/complete.json)；[待运行的密集 DELTA 实现](analysis/o_motion_20260915/extract_delta_dense_history_v1.py)。

下一步固定原历史端点、query、深度模型、标定和完整评分人口，比较两帧与密集历史，分别报告端点差分和真实时间戳 LSQ。该比较增加了输入观测，属于信息量诊断。若仍无法兼顾 moving 收益与 stationary 代价，停止扩展外部三维速度路线，转向任务可用未来状态与直接 future-latent 的竞争实验。

## 归档与复现边界

`analysis/` 保留原相对目录和字节内容；[snapshot_manifest.json](snapshot_manifest.json) 记录每个源文件和 SHA256。历史版本、草稿、已失败实现同时保留，不能从文件存在推断它已运行成功。根项目补入 P2 的模型接口及其模块/测试，是已有实验代码的归档，不表示本次重新训练验证。

数据集、权重、逐点预测、原始目标、外部仓库镜像、SSH 配置和服务器环境不在此次提交中。报告中的部分原始证据链接因此只在完整本地工作区存在。代码保留原实验路径与资产哈希；在新环境运行需取得许可的数据、官方权重并配置实际路径。此分支不是开箱即用的模型发布。

后续每个有意义的实验完成、失败或方向决策节点，更新结果及此索引，运行 `tools/sync_research_snapshot.py --workspace /path/to/dropple`，审查差异后 commit/push。该工具只复制白名单文件，不启动任务、不自动提交。
