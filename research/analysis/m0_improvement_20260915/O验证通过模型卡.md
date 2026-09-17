# O：完整原任务验证通过的 M0 强对照

2026-09-15。用途：后续研究可复用的固定、已训练预测模型。**O 是“原始 M0 完整模型＋O 的完整未来预测头”，不是一份独立完整模型 checkpoint。**完整验证已结束，来源与统计经[独立审查](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/full_objective_independent_audit.md)通过；这不构成 ICLR 新方法声明。

## 资产与加载契约

以下是 H200 上的实际资产路径，不是本机副本。

**O 预测头 checkpoint：**
`/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915/campaign_objective_v1/runs/O/latest.pth`

文件 SHA256：`ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70`。
加载后的 head 状态摘要：`1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb`。后者是规范化 tensor-state 摘要，与文件 SHA 含义不同。

**必须搭配的原始 M0 epoch24：**
`/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/assets/m0_epoch24.pth`

文件 SHA256：`0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc`。

**冻结配置：**
`/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/runtime/configs/S0.py`

SHA256：`c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303`；同 SHA 本机源码见 [S0.py](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py)。原生仓库为 `/home/wangning/Workspace/RadarFlowOcc-sota-p2`。

已执行的保存路径复用冻结训练块：[memory_experiment.py:249](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:249)。`latest.pth` 的顶层字段为 `arm`、`metadata`、`pass_index`、`update`、`future_pred_head`、`optimizer`、`sample_orders`；其中 `future_pred_head` 是整个未来预测头的 state_dict，包括递推网络和完整三层占据读出，共13,274,016个参数，并非仅最后分类层。它没有保存原 M0 图像/雷达编码器等完整模型权重；附带 optimizer/顺序记录也不等于已经提供可无缝续训的通用接口。

实际评测先由 [build_native_model](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:102) 严格载入原 M0 的完整 `state_dict`，再按原生单槽契约构建副本，以 `model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True)` 覆盖 O 预测头；检查实际 tensor 摘要后全模型 eval、冻结参数，不恢复 optimizer，也不安装 O 训练损失适配器。加载过程见 [原生共同评测:303](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/joint_native_evaluation.py:303)，实际 v2 入口将各副本固定为 native1 并增加[来源检查](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_joint_evaluation_v2.py:511)。因此已执行命令中的 `--checkpoint` 指原 M0；O 文件由评价协议的 `final_sources.O` 指定，不能混用两者。

**已执行入口与命令出处：**`jobs/objective_joint_full_v2/package/objective_joint_evaluation_v2.py`，Python 环境为 `/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/envs/hym_driveocc_m3/bin/python`，实际版本 PyTorch2.1.2+cu121。完整 argv 保存在[完成作业 state.json 的 shards.0/1.command](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/state.json)，加载证明见[loaded_models.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/evaluation/shard_00_of_02/loaded_models.json)。本卡未发明或执行新的推理命令；旧 argv 含已完成输出目录和已到期截止时间，是可审计执行记录，不应原样重跑。

## 训练、任务与固定版本

O 从原 M0 初始化，只训练完整 future_pred_head；原图像/雷达观测主干冻结，使用原生 t0 状态缓存。原 history=2、memory=1，原雷达五 sweep 及 ego/action 元数据契约保留；未加入未来图像教师或未来占据输入。原粗 GT `256×256×20` 上优化 CE 权重[1,5]与 Lovász，三个 decoder 层共六项，每层池化 t0＋四未来时域；sem/geo 仍前向计算以检查有限值，但不贡献优化目标。

训练只有 seed11：256个训练场景、512个不同 anchors，四轮共2048次样本使用，累积4，固定最后512次 AdamW更新；lr1e−5、weight decay0.01、clip35，warmup50＋cosine，与 C 匹配。没有挑最佳中间 checkpoint。预测为二分类 fine-grained GMO：class1是可移动语义集合，不等于当前确实运动；class0包含未占据及静态背景，255忽略。评价保留原 `512×512×40` GT、原生上采样/argmax，输出时域0/0.5/1/1.5/2秒；主指标只平均四个未来时域分别池化的 GMO IoU，不是 binary mIoU。

冻结训练协议：[objective_supervision_protocol_v1.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_protocol_v1.json)，SHA `275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d`；完整评价协议：[objective_joint_evaluation_protocol_v2.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_joint_evaluation_protocol_v2.json)，SHA `b1b1b046b5acf9dc1207a19e0b7b740d6db38a0982a3794f670fa06862189f28`。两者记录完整源 SHA 链；训练器 SHA `3fce7837cf9e58e7c72ca747b6e609f60c4f78b57e3dba23826dc3862a49e4cc`，实际评价入口 SHA `f8e3da3de084f5cde3d35b9574124d9684b7ecf77387bd03cf5828c02f7d88ca`。观测计算保留 TF32，O 与 M0_fp32 的 future matmul 关闭 TF32、cuDNN TF32开启，全部 FP32。

## 已验证效果与使用边界

四模型共同完整 nuScenes 原生 usable validation：5,119个 anchors、150个场景，各场景自然33–35条；无训练、阈值拟合或子集筛选。10,000次 seed11 场景配对 bootstrap，保留每场景全部样本。

| 模型/比较 | 未来平均 GMO |
|---|---:|
| 原生 M0 / M0_fp32 / C / O | 13.94438% / 13.94485% / 13.72765% / **14.74779%** |
| O−M0_fp32 | **+0.80294 pp**，95% CI [+0.68178,+0.94061] |
| O−C | **+1.02015 pp**，95% CI [+0.92960,+1.11151] |

O 对 M0 的四未来时域均提升，但 t0−M0_fp32 为 **−0.13958 pp**，CI [−0.16824,−0.10944]；t0−C 为+0.08814 pp。相对 C，未来 pooled FP减少96,713,587而 FN增加8,281,704；相对 M0_fp32，未来 pooled FP/FN分别减少26,205,756/321,957，但后两时域 FN增加。它不是所有错误类型、时域或下游安全性的统一改善。

保留原 M0 和匹配控制 C。C完整路径为 `/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915/campaign_memory_v1/runs/native1/latest.pth`；SHA `a12426c51b785fb34c3f6c09ec7fdf3c62b9da89e82f5a10d539a851a5a2d72b`。F、坐标 A、双槽等既有结果继续披露，不因本次成功删除；未给未晋级分支虚构 full 分数。

validation 有历史曝光且包含开发场景；不是新盲测。区间未校正多重比较、不覆盖训练种子或模型选择不确定性。只验证了这个单 seed、固定训练预算的 O；没有证明通用泛化、因果机制或 ICLR 新颖性。完整结果 summary SHA `199347570b9d34130c6366ef3fb138bbc08c1310a50010a1f200061fe428f463`、complete SHA `cdf54175b84613b1e175fe783f490ab62681c18060c5289981f2be426d2a53e5`。本卡依据真实 CPU checkpoint 审计、GPU 加载回执和本地源码复核；本次未再次搬运或读取大权重。
