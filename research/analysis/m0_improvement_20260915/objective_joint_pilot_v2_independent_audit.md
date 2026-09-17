# Objective joint pilot v2：独立核验

结论：**PASS_FROZEN_COLLECT_ON_ACTUAL_MIRRORED_PILOT**。本记录整理已完成的独立只读核验，不增加复验、训练、评价或源码改动。Pilot 仅证明这两条固定样本的工程一致性及资源可行性；**未计算或评估统计效果**。

## 已复走的证据链

使用已冻结 `objective_joint_merge_v2.py` 的 `collect`，对真实回传的共同 contract、两分片 manifest/index/loaded_models/pilot_reference_sources、两条 sample records/complete 重新核对哈希、身份、模型权重收据、原始 GT 计数、四模型混淆矩阵和资源统计。重新计算的 contract SHA、分片收据和资源汇总与真实合并 summary 完全一致。未创建新的实验结果文件。

两条样本对应 official index 2036、2998，sample token 分别为 `297c52902a384b84ba179f3160ac54e9`、`01076466d815421c857d2c8cf8034488`，来自两个不同场景。模型集合严格为 M0、M0_fp32、native1/C、O；没有更换候选或 checkpoint。

两条样本均满足有效输入投影哈希相等、原始 GT 精确一致、M0 全部 5 时域 × 3 层 logits 原精确门通过、四模型各自五时域完整 GT 混淆矩阵与历史参考精确一致。四模型 `turn_on_plan=False` 且全部模块处于 eval 状态。

两条样本的**完整输入哈希均不相等**，记录没有将其误写为完整边界一致。唯一允许排除的比较字段是 `inputs.plan_dict.sample_traj`；双方均为 `torch.Tensor`、float64、形状 `[1,1800,5,3]`，分别有 23,570 和 23,490 个数值元素不同。双方完整树、投影树和字段哈希、类型、形状及差异统计均保留。排除仅发生于 pilot 对历史缓存的比较副本，真实模型输入没有改动。正式完整评价不应用该历史缓存投影。

## 固定哈希

| 对象 | SHA256 |
|---|---|
| v2 evaluation protocol | `b1b1b046b5acf9dc1207a19e0b7b740d6db38a0982a3794f670fa06862189f28` |
| v2 producer | `f8e3da3de084f5cde3d35b9574124d9684b7ecf77387bd03cf5828c02f7d88ca` |
| v2 merger | `0394ff12140891f70f38527f290b7f71e3c6b52bfe76bde80002620c20ac003b` |
| v2 campaign wrapper | `837bafed3419c0c187ea4827a368ce7567c4e02cd7143e8a52550a84b7dd90be` |
| Pilot summary | `423da1537731aed3deaed72303faa1532c20b0fb7169a4b157978c0033839db5` |
| Pilot merge complete | `e4b2690d7045a8bf9b0cc49a3ac22bf87d0a282537e191ea17167fbeb2adb3fb` |
| Common contract | `58a20b5c4457cb26a95ecd7b46eaf210b117a1ab5b97eb023ac8709262aa97b7` |
| Sample hash ledger | `49e63ad4b1c408508312335ae9aa6fc6252bd1fde197e0f4a0df9859bdca6cc2` |
| Shard 0 complete | `112066006774ec70a3bf42eab5f3e8236d501ab503945bd354917ddc08466154` |
| Shard 1 complete | `affefacd9218dc0670a8e9cba2ee8c1c0afb58b07e69364b9fea37c8c19e8740` |
| 原始失败 difference | `aecf29305e3d4b086e834df4b391580e0964a330a22f4d91c09566b4036b95a5` |
| 原始失败 diagnostic complete | `9577fa4cadd0c1be8c8c189dba6f85e8ed330ea912f7f7047967ced717f635fb` |

原/新包内源码字典均已逐文件核对；重新派生的九项受控 AST 变换及逆变换恢复门与真实 receipt 一致。C/O 的 checkpoint SHA、实际加载 head SHA 与协议及完整开发集审计相同；M0 和 M0_fp32 的 head SHA 均为冻结初始化。

## 同 SHA 本地镜像映射与边界

服务器工作根为 `/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915`；本地工作根为 `/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915`。

- 服务器 `campaign_objective_joint_pilot_v2/evaluation` 对应本地 `server_results/campaign_objective_joint_pilot_v2/evaluation`；summary 对应本地同 campaign 的 `summary_v1`。
- 开发结果原协议绑定服务器 `campaign_objective_v1/summary_v1`。只为定位同 SHA 文件，在内存读取指针中将其映射至本地 `server_results/campaign_objective_v1/summary_v1`：summary SHA 为 `46c37b5b074e73896e357e0bc9744a538abb01b288521769a577f459be54ad6b`，complete SHA 为 `29a79f3225505cf8784d1a70e8207f36ff2f8b13792bd51929b86a896d3ba3ca`。原评价协议及输出文件未编辑，实际传入 `collect` 的评价合同仍为原冻结内容。

本地核验**没有重读大型 checkpoint、原始图像/雷达/GT 数组，也没有重新运行神经网络**。权重和逐位预测一致性的证据来自已锁定 producer 实际执行的门及其完整哈希绑定回执；本次独立复核覆盖回传文件和冻结门的链条，不冒称本地重新做过 GPU 数值验证。新增三项运行来源的核验没有被追溯描述成旧缓存创建时已锁定的来源。

## 资源解释

两条样本耗时 2.880953、3.273153 秒，均值 3.077053 秒；两分片驻留 46.920968、46.578991 秒，包括加载与检查，合计 93.499959 秒。这不是外层 job runner 的总驻留或 GPU kernel 时间。峰值 CUDA allocated 为 5,340,401,152 bytes，即约 4.974 GiB。

按较长分片 2,560 anchors 简单外推，均值/较慢样本约对应 2.19/2.33 小时，另有初始化与汇总成本。两条样本不足以估计全数据的时延尾部或内存极值。正式 21,600 秒/分片、32 GiB/卡、两分片的授权，是合理的硬上限：约为较慢样本外推的 2.58 倍，内存上限约为已观测 allocated 的 6.43 倍；不是保证耗时，也不是要求运行满六小时。完成即退出，外层 22,800 秒另涵盖检查、CPU 汇总与清理空间。

未查看部分完整评价性能，也未据本 pilot 宣称 O 优于 M0。候选的完整评价结论仍须等待冻结的全部 5,119 anchors / 150 scenes 共同评价和统计汇总。
