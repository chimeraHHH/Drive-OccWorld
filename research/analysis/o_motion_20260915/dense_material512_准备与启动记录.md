# 从小样本机制检查到 train512 未来预测对照

日期：2026-09-17。目标仍是超过保留模型 O，同时提高运动预测准确性。本文件中的 current-codec 结果和数值检查都不是达成目标的证据。

## 已完成的共同表示

`shared_current_codec512_v1` 已完成，实际 2,048 更新、512 个原始训练样本、256 个场景，耗时 996.64 秒。仅使用当前占据标签训练 encoder/decoder，未使用未来标签值训练该 codec；递推模块、内容更新和速度头保持新初始化。服务端 runner 以 EXITED_ZERO 结束，runner/child 均已核验不存在。

从 512 个逐样本混淆矩阵独立求和，当前帧 IoU 为 **16.505674%**。这是训练集当前帧二分类 GMO IoU，不能与 O 的验证集未来 IoU 混比。全部 2,048 个样本顺序与预先固定的四轮 seed11 permutation 一致；原始 checkpoint 字节与 encoder/decoder 参数哈希已经额外加载验证。

证据：[共同表示复算](shared_current_codec512_evidence_v1.json)、[checkpoint 参数核验](shared_current_codec512_parameter_evidence_v1.json)。完整文本回执位于本地 `server_results/training/shared_current_codec512_v1/`，权重保留在服务器。

## 本轮要区分的解释

前一轮 train4 中 T 的物理误差较低，但小幅 IoU 差小于同种子 J 重跑的数值差异。因此停止围绕 train4 继续调参，扩大为完整 train512，并固定可重复的数值路径。

| 组 | 未来渲染内容 | 读出方式 | 共同条件 |
|---|---|---|---|
| T | 当前内容保持固定 | 按递推位移传播 | 共同 codec、原始输入、初始化、样本顺序、损失与预算 |
| J | 自由更新的未来内容 | 按递推位移传播 | 同左 |
| D | 自由更新的未来内容 | 直接读出 | 同左 |

T 保留全部递推状态容量，后续速度仍可依赖先前隐状态。其约束是最终渲染内容固定；不能将密集源体素称为已恢复的对象身份。D 是同容量的读出对照，不代表最强通用世界模型，O 仍是必须超过的参考。

每组仅一个 seed11，固定 2,048 次更新，encoder/decoder 冻结；原 CE+Lovasz 加 0.1 倍对象/速度组平衡的物理损失。无中途最佳模型选择。完整训练协议和代码哈希在启动前固定。

数值策略为严格 deterministic FP32，关闭两种 TF32。原占据损失与有梯度的 trilinear resize 在 CPU，原稀疏物理损失在 CUDA。完整原尺度、非零位移的 T/J/D 合成前反向检查已通过，各组重复损失与梯度哈希完全一致；未做任何优化更新。它证明该计算路径可执行和本进程重复一致，不证明完整训练跨进程复现。首次 probe 因工作目录造成原库导入失败，第二版显式固定仓库工作目录后通过，失败记录保留。

证据：[完整计算图检查](material512_determinism_evidence_v2.json)、[训练协议](dense_material512_training_protocol_v1.json)、[训练设计](research_notes/dense_material512_training_design_v1.md)。

## 服务器顺序与评价边界

固定顺序为 **T 训练→T 评估→J 训练→J 评估→D 训练→D 评估**。由服务器独立 controller 和已有所有权约束 runner 执行；不依赖本地终端常开，不自动重试。每组训练上限 10,800 秒 / 24 GiB，评估上限 3,600 秒 / 16 GiB。失败时停止后续阶段，保留实际完成的工件；不重启已经完成的训练，也不操作其他用户任务。

评估固定在原 dev200 / 100 场景：复用经过认证的 O 原始逐样本统计，在同一 fine XYZ 网格和完整 GT 分母上比较未来 GMO、动态区域召回、到达/腾空区域 IoU、全域 FP/FN。每个候选先完成预测，再读取标签；原始 GT 计数、速度分组分母、时间间隔及 transition domain 都需与 O 回执一致。

候选有自己的当前占据输出，不能假定与 O 当前输出完全相同；共同输出指标可以比较最终行为，但单凭占据变化不能唯一归因于运动。物理 EPE 只比较 T/J/D 的同一原始刚体代理标签；O 没有认证的匹配速度头，不能把这些 EPE 称为 O 的物理误差。

评估读取训练完成时由 controller 固定的 complete SHA，并检查 actual final checkpoint、共享初始参数以及 encoder/decoder 确实未变。三个 arm 的训练计划不受中间评估结果影响。

证据：[评估协议](dense_material512_evaluation_protocol_v1.json)、[服务器顺序](dense_material512_server_plan_v1.json)、[启动前输入认证](dense_material512_preflight_v1.json)。实际状态以服务器 `dense_material512_v1/state.json`、对应 job 的 `state.json` 和活动 PID 为准。

## 结果出来后的判断

若只有占据提升、运动相关指标退化，不算达到目标。若只有物理误差改善、未来占据仍落后 O，也不算达到目标。若在 dev200 同时改善，进入已记录的场景不重叠确认与原 5,119 样本完整协议；dev200 已反复用于开发，不能作为最终泛化证据。若三组均受当前表示限制，首先分辨表示/解码瓶颈与状态机制问题，不再回到 train4 搜索有利数值。

用户提出的“Doppler 减少视觉历史和真实计算”仍是高优先级独立假设。本轮使用已缓存 BEV，因此不能用本轮训练速度或单个 BEV slot 支持少相机帧和端到端效率结论。

## 已核验的启动快照

2026-09-17 08:40:46 UTC：H200 wrapper、controller 和 T_train 三个进程均通过 PID/UID/startticks 核验存活。T 已完成 26/2048 更新，最近 25 条记录的平均间隔为 2.465 秒/更新，三个可训练模块均有非零参数更新。这是运行证据，不是性能结果；后续仍以实时回执为准。
