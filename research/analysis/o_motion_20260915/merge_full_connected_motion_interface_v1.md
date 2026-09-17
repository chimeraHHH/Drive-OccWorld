# Full O/J/D 合并入口

仅接受已经完整完成的一个全量分片，或两个连续互斥分片；固定 5119 anchors / 150 场景。pilot、缺片、半成品、失败标记、重复/乱序身份都会拒绝，且不生成部分统计结果目录。

```sh
python merge_full_connected_motion_v1.py \
  --shards COMPLETE_EVALUATION/shard_00 COMPLETE_EVALUATION/shard_01 \
  --evaluator-sha256 6f8656a249973bd2a45cb781086170eb8bbc72821ecff0556c576728bc82c1a2 \
  --protocol connected_motion_protocol_v2.json \
  --stream-engineering server_results/training/native_full_o_stream_preflight_v1 \
  --training-run ACTUAL_COMPLETED_CONNECTED_MOTION_TRAIN_V2 \
  --training-complete-sha256 ACTUAL_TRAINING_COMPLETE_SHA256 \
  --selection ../m0_improvement_20260915/full_validation_selection_v1.json \
  --reference-dir ../m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1 \
  --sparse-labels full_sparse_motion_local_v1 \
  --out NEW_SUMMARY_DIRECTORY
```

路径可用本地镜像；上述相对路径以 N 为运行目录。一个分片时只传 shard_00，它必须独立覆盖全部 5119。两个分片固定 `[0,2559)`、`[2559,5119)`，不支持混用别的分片规则。

输出：summary.json（原生占据/物理 EPE 与共同输出统计）、input_bindings.json、physical_coverage.json（逐锚 NPZ 覆盖审计）、report.md、complete.json（四件 SHA）。不会选模型、改阈值、改损失或重新训练。

原统计函数保持源字节与调用：`summarize_connected_motion_v1.summary(..., repetitions=10000, seed=11)` 和 `summarize_common_change_v2.summarize(..., bootstrap_repetitions=10000, seed=11)`。两者原占据点估计及配对区间必须完全一致。物理标签按真实 valid/object_index 独立重建应有的 `(instance,horizon)` 集合，逐个核 P/J/D 的组别、有效 source_points 和实际 dt，包括无有效对象的 anchor；不只检查三臂彼此相同。zero EPE 保留原统计函数的三臂一致性检查，不另算新基线。

本地依赖：producer 实际20份源及其依赖、两个冻结 summarizer、原 metric；实际 `imports()` 会逐项核 SHA。科学函数 SHA 分别为 `3b0822c7d4c9218211dfec7c7d6854fd67b28f9b43640134dfe2228690a2c40e`、`318dee1a19fbd5b0863f50d2e0346723d7050329b55170cb8efac60929b962ae`。

数据资产：每个 evaluator shard 的 complete + manifest/loaded_models/records/physical_objects/summary 五件；完整 full_sparse 的 manifest/complete/三审计文件和全部 5119 NPZ；原 full O complete/summary/raw_confusions/sample_hash_ledger 四件；原生两锚 PASS 的 complete + 四件；J/D 训练根 complete + 五件 top 产物，以及各臂 complete/manifest/training/development_records 四件。实际 checkpoint tensor 及 optimizer 身份由每个 producer 的 strict loader 已核，合并器严格复核其 receipt 对训练最终指针；不重新 Torch 加载或恢复优化器。

约 114.8 万物理行只解析/保留一次，逐锚 NPZ 用后释放，不额外构造百万条 fixture；原科学统计函数可能需要数 GB 内存，建议在可用内存充足的本机或服务器 CPU 上运行。本次未修改统计实现来追求速度。

本机验证：5 项微型边界测试通过（共同漏行/多行、空 anchor、错误 group/points/dt一ULP、重复/缺片）；Python 3.10 AST、CLI、20 份实际源码 SHA/导入、真实两锚 PASS 文件链、原 full5119/150 参考、真实 full_sparse manifest 及四份实际 NPZ 支持解析通过。未分析未完成的候选 full 数据，未生成 mock 结果，未使用 GPU。
