# 监督粒度诊断脚本独立审查

2026-09-15，只读源码审查。对象为 `supervision_granularity_probe.py`，最终 SHA256：`ee936ab26571a95eec0cf06fe7a858ff7641292d895d4ed14282b77da47069bd`。未连接服务器、运行模型、反向传播或修改冻结源。

**结论：未发现阻断当前只读诊断的代码问题；这不是运行成功或科学假设成立的结论。** 完整 GT 混淆矩阵与原损失的真实复现仍须由脚本运行时逐条通过，失败不得输出 COMPLETE。

审查依据为实际 `drive_occworld.compute_occ_loss/evaluate_occ_records`、`world_head_v1._downsample_occ_target/loss_occ/loss_voxel`、`semkitti_loss`，以及已完成的 `objective_metric_audit.md` 和 `posthoc_train_diagnostic.py`。

1. **轴与原生调用路径。** 观察器装在模型实例上，原 `compute_occ_loss` 负责预测/GT 的 XY 轴和五时域展开。原 `loss_occ` 下采样一次，再将 `target_voxels_prepared=True` 传给观察器。脚本从冻结 `loss_voxel` 抽取原始前缀获得真实 `256×256×20` logits/GT，未另行猜测轴序或二次下采样；观察结束后继续调用原 `loss_voxel`，finally 恢复两个方法。最后层 coarse confusion 的 class 轴为 1，矩阵行是真 GT、列是预测。

2. **背景与 ignore。** 粗标签直接取原 mode 函数结果，不用 max pooling 或普通多数投票替代。全 GT 的 2×2×2 block 分组与原 reshape/permute 相同，原 0/1/255 到粗 0/1/255 的 footprint 同时核对行和、列和与八体素计数。`nGMO` 与 ignore 数量分别记录；其他模型和中间层核对同一 coarse GT 哈希。粗 ignore 覆盖的有效原 GMO 保留在原始 full-GT 评价中。

3. **概率细节。** semantic 两类分别用 `softmax0/softmax1`；geometric 前景严格用 `1-softmax0`，geometric specificity 明确使用原 `softmax0`，没有错误替换为二次取补或 softmax1。两者 FP32 相消的差异及其在 GMO 上的数量单独记录。统计归约在 float32 中完成，转为 Python float 后计算展示比值；因此这些比值不是原 FP32 除法/BCE 的逐位重建，报告已明示。

4. **聚合层级。** 三层分别记录每时域与“当前＋四未来”合并统计。原 12 项 loss 仍由原函数计算；逐时域软 P/R 不相加还原非线性 pooled loss。粗/完整 GT 指标分别命名，future macro GMO 按原聚合器定义计算，不把当前帧混入未来主均值，也不以 coarse IoU 替换 full GT 指标。

5. **固定样本、模型与复现。** 固定旧 train16 的 token/scene 顺序及原 summary/records 哈希；完整模型参数按旧最终 checkpoint SHA、头张量 SHA、样本训练顺序和 manifest 严格加载。原 full confusion 在每模型、每样本、五时域上要求与旧 posthoc 矩阵完全相同；原 12 项 loss 要求相同键与 `rtol=1e-5, atol=1e-6`。原/粗 GT 类计数还各自与混淆矩阵行和核对。没有评价集推理或据分数重新选择 train16。

6. **执行与解释边界。** eval/no_grad、所有参数冻结、逐模型前后头状态 SHA 相同，无 optimizer、阈值选择或大张量输出；已有输出目录拒绝复用，600 秒单次上限，无自动重试。代码成功后只能描述这 16 个已训练样本的监督粒度与概率行为。没有分项梯度或目标干预，不能据标量 loss/软交集集中声称某损失控制了训练更新，更不能据此宣布替换损失会提高总体性能。

本次无待修阻断项。实际依赖导入、CUDA 数值复现和完整 64 次固定模型记录，交由服务器运行收据判定；本审查不替代这些证据。
