# 自有未来状态：训练接口通过，但运动尚未学好

2026-09-17，真实 H200 实测；没有 mock 数据。当前结论：**实现和工程拟合已完成，尚未超过 O，也没有建立新的运动准确性证据。**

## 已完成的工作

冻结合法观测 BEV，用自有三维 latent、source-indexed 位移及 decoder 预测未来占据。两个臂从相同 current-codec 初值开始，输入、参数量、样本顺序、训练更新数和物理监督相同；transport 搬运内容，direct 用 identity readout。未来输出没有 O logits/future-feature 残差路径。详见[设计及文献依据](research_notes/task_native_state_design_20260917.md)。

独立审查发现 v1 decoder 的 coverage 附加输入会混淆搬运解释，故主动停止 v1，保留源码、协议、Git 提交和服务器终止回执。已记录 53 条更新日志，停止检查在一次 optimizer.step 之后，可能另有一次未记录更新；没有使用这个未完成版本的模型结果选路线。v2 的 decoder 只读内容，coverage 仅用于几何归一化。

v2 固定 seed11，样本序号 `[396,445,60,115]` 来自预定 shuffle 前四项；只访问这四个训练样本。current-codec 32 次更新、transport/direct 各 32 次，共 96 次更新。两臂均 **107,458 个参数**。H200 训练用时 **51.06 s**、峰值 **3.559 GiB**；额外只读评分用时 **19.03 s**。两个 runner 均 EXITED_ZERO，评分 runner/child 已核对退出。

通过了 XYZ 位移、边界权重流失、fractional splat 梯度、y-major token 到 XYZ 布局、非对称 impulse 上采样峰值和初始两臂预测一致性检查。v2 decoder 改变 coverage 张量时输出完全相同。真实 occupancy loss 对 transport 独立 velocity 头的梯度范数为 **0.16949**，direct 为 None。这证明任务连接存在，不证明它已学到正确运动。

## 同一训练人口的结果

下表仅是 4 个训练样本的固定 final32 诊断；**不能与 O 的 5,119 样本 14.7478% 直接比较，更不能称为泛化性能。** 使用完整 `512×512×40` 标签、ignore=255、原 trilinear logits 插值；先累加每个 horizon 的混淆矩阵，再对四个 future IoU 取平均。

| 读出方式 | future GMO IoU |
|---|---:|
| transport，正常位移 | 12.30345% |
| 同一 transport 模型，位移置零 | 11.51078% |
| 同一 transport 模型，位移反向 | 10.74089% |
| direct | 12.23518% |

transport − direct 仅 **+0.06827 pp**。正常/置零/反向干预表明这个固定模型确实使用了位移；不能据此把两个独立训练模型的差异全部归因于正确的物理运动。direct 的三种渲染干预输出一致，符合其定义。

2 s 的原始 voxel-center 刚体材料点标签，三维 EPE、对象等权：

| 模型 | moving，40 个对象 | stationary，49 个对象 |
|---|---:|---:|
| 零运动参考 | 11.61073 m | 0.01907 m |
| transport | 11.45472 m | 0.19828 m |
| direct | 11.40479 m | 0.26893 m |

moving 相对零运动的改善很小，静止误运动增加，transport 在 moving 上还略差于 direct。形状拟合 loss 从 1.28554 降到 0.46350；未来 occupancy loss 从约 0.49921 降到 0.41506/0.41650，不能把这些 loss 下降替换成物理运动学好了的结论。

样本实际未来时间与名义 0.5/1/1.5/2 s 最大偏差 **2.203 ms**。模型仍输入名义 horizon，物理标签按数据实际时间生成；这是本轮有限样本的偏差，不代表整个数据集时间完全一致。

## 研究判断与下一步

这次把“运动是否进入任务输出”从代码猜测变成了可干预的事实，但仍没有解决“怎样学到足够准确的运动”。32 次 rollout 更新只是工程拟合，不能据此否定该结构，也不能据此扩大成论文主结果。

下一阶段应先检查这类状态的运动可学习性和占据梯度是否抑制大位移：固定这批样本和统一 codec，对比充分的物理单任务拟合与联合拟合的学习曲线，随后才锁定较大数据的正式预算与更强直接 future-latent 对照。一次成功的条件至少包括物理误差实质下降、静止误运动受控、原占据指标保持，以及位移干预方向合理。保留 O/M0，不复活 S3，不继续扩展外部 depth/tracker 速度替换，不新增多 seed。

证据：[工程完成及训练摘要](dense_task_state_fit_evidence_v2.json)、[完整读出审计](dense_task_state_readout_audit_v2.json)、[冻结协议](dense_task_state_fit_protocol_v2.json)。本地独立复算了 6 个聚合混淆矩阵、30 个 horizon IoU 与 6 个 future 均值，容差 1e-15；没有把这称为逐点 EPE 的第二份独立实现。权重与原始标签未上传 GitHub。
