# 固定 transition/readout 的 2×2 诊断：设计审查

结论：建议作为**只读、事后模块交换诊断**，先做原有两条固定 anchors 的工程预检；全部门通过且另有冻结预算后才评价原 dev200。不是新的训练臂，也不是本次 full 的新增候选；任何 hybrid 分数均不能用于宣布研究目标成功。

## 估计对象与两条分解路径

记 (Y_{FR}) 为 features 来源 (F\in\{C,O\})、readout 来源 (R\in\{C,O\}) 的同一指标，顺序始终是 **features/readout**。C、O 均使用同预算固定最终 checkpoint。总差 (\Delta=Y_{OO}-Y_{CC}) 有两条严格恒等的分解：

\[
\Delta=(Y_{OC}-Y_{CC})+(Y_{OO}-Y_{OC})
       =(Y_{CO}-Y_{CC})+(Y_{OO}-Y_{CO}).
\]

必须同时报告四个条件、两条路径和

\[
I=Y_{OO}-Y_{OC}-Y_{CO}+Y_{CC}.
\]

第一条在 C readout 下先替换 features，第二条在 C features 下先替换 readout。交互项表示交换效果随另一个模块而变；它不是自动成立的“协同机制”证明。IoU 本身也是非线性比率，即使 logits 层结构简单，指标层仍可出现交互。不能挑一条路径作唯一贡献分配，更不宜将相互抵消的差值写成百分比归因。

主统计保持原定义：每个未来时域先汇总 confusion，再求 GMO IoU，最后四时域平均；逐时域、t0、完整 confusion 同时保留。若正式 dev200 做区间，应按同一 100 场景成对重采样，并直接计算各差值与交互项的重采样分布；不能拼接端点得到交互项区间。全部是历史暴露开发集上的事后诊断，不能充当新盲测。

## 科学识别限制

这里能够定义的是 **固定已训练模块替换的效果**，不是损失改变通过某模块产生的唯一训练因果贡献。两个模块在训练中共同适应；未观察到独立随机干预下重新训练的反事实。

潜空间还存在表示坐标的歧义。一般例子：令 (z'=Az)、(g'=g\circ A^{-1})，联合预测 (g'(z')=g(z)) 不变，但交换 (g(z')) 或 (g'(z)) 的结果可以变化。因此 OC/CO 下降，可能表示特征分布或潜空间坐标与 readout 不兼容，**不能单独证明 features 所含信息变差**。这也不等于此前已否定的空间坐标修正路线重新成立。

同初始化、相同架构及固定观察状态 (z_0) 会约束这种自由度，但不能消除未来特征的共同适应风险。上述线性例子说明一般识别问题，不声称它是当前网络在全部约束下已验证的精确对称性。本轮不拟合 stitching 映射、不调阈值、不补训练；那些操作会引入新的估计对象和对照。

## 实际源码边界与必须通过的门

已只读核对当前源：[future_pred](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:465) 在 L477–480 只用 `len(bev_pred_head)` 复制 t0 观察特征；L536/545 生成未来特征，L576–582 回灌 `pred_feat[-1]`，L591 才统一调用 `forward_head`。L563 的额外读出受规划分支控制。当前 [特征生成路径](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_base.py:280) 及 transformer 不消费 `bev_pred_head` 参数；[分层读出](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:162) 单独处理各层特征。

建议捕获 C/O 各一次**末端** `forward_head` 输入，再用两套原冻结 readout 分别重放，形成 CC/CO/OC/OO。这样避免整模型临时换参影响 rollout 或留下状态污染。预检须确认：

- 来源是协议绑定的 C/O 固定最终 head；读出包括全部 `bev_pred_head.*` 参数和相关 buffers，不只最后 Linear。其余 transition 参数归属明确，前后 state digest 不变。
- 同一完整 cache 输入、GT、sample/scene identity、action/ego 条件与原 evaluator；每次重放使用独立输入副本。全部模块 eval、无梯度/optimizer，数值策略与原 C/O FP32 评价一致。
- `soft_weight=False`、原单槽及相同 readout 分支数/结构，规划、flow、M3/motion 等反馈分支关闭；末端 `forward_head` 恰调用一次。若这些条件变化，本审查的解耦依据不再适用。
- 捕获特征轴为 `[5,3,B,HW,C]`，预测轴及每项长度按原生 receipt 核对；不得交换时域、decoder layer 或类别。同一 features 来源在两次 readout 重放之间保持逐位不变。
- CC/OO 全五时域、全部三层 logits 与各自未拆分原前向精确一致；最终层五时域 full-GT confusion 与对应旧 dev 记录精确一致。仅 IoU 相等不足以通过工程门。
- **t0 强门**：C/O 的 t0 特征逐位相同，因此各层所有体素必须满足 `OC(t0)=CC(t0)`、`CO(t0)=OO(t0)`，同时 logits、预测和 confusion 一致。若不成立，先定位模块边界、输入或数值策略问题，不进行科学解读。未来特征可以相同或不同，不应把“必须不同”设为门。

两条预检样本只用于工程与资源核验，不作效果结论。通过后也不得把事后选出的 hybrid 加入已冻结 full；完整评价继续保持既定候选集合。本文件是设计意见，尚未代表实现审核或真实预检通过。
