# 下一轮结构实验：共享预测刚体状态

2026-09-16。实现草案；N/P/Z 完整终点及独立审查均已完成通过。本文不是已运行的协议，不恢复 S3、不延长 V/G，不新增训练种子。先完成必要 CPU/native 四步预检，再冻结正式源与资源。

## 选择依据和研究问题

同一 V 终点在原速度、帧内速度置换和速度清零下，原完整占据及物理指标几乎不变；与 V/G 训练对照一起，支持停止当前 scene-latent 速度条件化路线。它不定位唯一原因，不能证明旧索引错误。新问题是：让同一可检查的运动变换既决定材料点位移、又决定未来占据特征出现的位置，是否能产生超出强 CRN-CV 与原 O 的学习收益？共享对象状态/flow 引导占据本身已有 UniAD、ImplicitO、Occupancy Flow Fields 等先行工作，此轮先检验效用，不宣布新颖性。

## 两臂共同模型

1. 当前原预测框全部保留既定八类，不新增 score threshold/top-k/GT 匹配。只使用当前预测状态与原当前观测 BEV；GT 仅进入原 loss/evaluator。原官方检测器冻结，不进行训练。
2. 物理几何接口使用原 float64 centered boxes、G0、全规则源格点；完整 3D box membership、最高原 score、strict > 保留原序同分、无覆盖为零，与已验证 CRN-CV 完全一致。owner 固定于当前输入，不随未来改变。学习特征可使用原 float32 packed27，但不用于声称原 CV 几何逐位相同。
3. 对象上下文取 `prev_bev_input[:,-1]` 的当前 BEV：在预测几何中心及局部 z=0 的四个 footprint 角点的 XY 位置作 bilinear sampling，五点平均；grid_sample 使用 align_corners=False 和 zeros padding。完整旋转先把点变换至 R。超界框保留，禁止根据采样是否越界删框。
4. 每个 h 单独以 [packed27, current_context256, h/2] 为输入，284→128→128 的 SiLU MLP。共享网络覆盖四个固定名义时域。零初始化的 128→6 头输出中心速度修正和 R 中的角速度修正；乘 h 得 δc、δω，构造 `c_h=c0+h*v+δc` 和 `R_h=Exp(δω)R0`。不额外学习尺寸，不用真实未来时间/位姿。
5. 材料点场：所有规则源点按固定 owner 查询，`d_h=h*v_owner + δc_owner +(Exp(δω_owner)-I)(p-c0_owner)`；无覆盖点恒为0且继续计分。保留 float64 基底与输出；零残差分支直接保留 CV 字节并维持正常加法 Jacobian。旋转指数零点必须通过有限梯度测试。全场定义之后才由外部原 source_flat_indices 取监督。
6. 未来对象特征：从 [hidden128, δc3, δω3, h/2] 用 135→32→32 SiLU MLP 产生值向量。每个原 BEV 柱的 XY 中心以预测 `(c_h,R_h)` 计算连续空间权重。权重为原 score 乘二维高斯核；协方差为 `R_h diag((l/2)^2,(w/2)^2,(h_box/2)^2) R_h^T` 的 XY 块，加固定格子积分方差 `cell_xy^2/12 * I`。尺寸/原 score 不学习或裁剪；该权重是空间特征先验，不声称 score 已校准为概率。
7. 以 `sum(w_j*value_j)/(1+sum(w_j))` 汇聚；分母的1对应零值背景项，空对象返回零。固定分块仅改变计算组织，不删框。32→256、bias=False 的零初始化投影提供对象残差，在原三个占据 decoder 分支的未来特征入口分别相加，t0 不注入。原未来预测主干、全部解码层、O 稠密路径与 evaluator 保留。新对象场显式使用输出 BEV 坐标，不把旧 hidden tensor 宣称为物理材料坐标。所有物理/占据前向不接受 GT identity、GT mask 或原稀疏支持。

## 唯一匹配控制

候选 Cpl 的占据空间核使用学习后的 `(c_h,R_h)`；控制 Fix 的占据核使用 `(c0+h*v,R0)`。两臂都拥有完全相同的 CV 物理基底、可学习物理修正、对象值向量、O 初态、容量和 loss；只切换“学习位姿修正→占据空间地址”这一条前向边。控制的物理 δc/δω 仍学习、值分支仍共享同类信息，不故意用错误源地址。配对初始化与 RNG、数据顺序一致，seed11。

原 O 占据损失 CE[1,5]+Lovasz、原分组/对象/点归约的 SmoothL1(beta=.5)物理损失沿用。新参数与 whole future-head 的优化预算在预检后冻结，首轮仍采用与已完成研究可比的 train512、2048例、512更新、dev200固定末点；本轮不做中途挑 checkpoint。资源参数需测实际峰值/每步耗时，不由估计直接启动。

## 预检与判断

必要检验包括 CV/owner 与原几何逐点一致、固定R的三轴旋转/平移、无覆盖保留、SO(3)零点前后向、初态O与CV复现、Cpl/Fix 初态一致、未来位置变化确实改变占据地址、首步之后物理与占据对共享运动参数的真实梯度，以及原native四次AdamW更新/accum4。所有临时权重丢弃，正式从原O+同初始化重建。

正式比较两臂各自相对 O 的原未来 GMO、各自相对 CRN-CV 和 D 的原完整支持物理 EPE，再比较 Cpl−Fix；四时域、moving/stationary/ambiguous、原 FP/FN、moving recall 与 t0 全部报告，原场景配对bootstrap不改。只赢 Fix 或只优于 D 不算成功；不能把替换 CV 基底本身的增益当学习收益。缺预测框的材料点零输出会限制物理上限，必须保留覆盖统计而非排除样本。共同模型即使优于 O+CV、但 Cpl 不优于 Fix，也只能支持对象表示/修正效用，不能主张所选连接机制成立。

未经候选+强控制的完整开发结果，不延长训练、不做全5119验证。若出现同时改善且通过原晋级证据要求的候选，再单独冻结完整验证和更广训练覆盖；不把暴露过的单种子开发结果写成盲测或总体泛化结论。

实现分工：`crn_full_grid_geometry_v1.py` 负责无GT精确基底，`shared_rigid_prediction_geometry_v1.py` 将原export owner映射到保留对象顺序并保留原两套float64数值路径；`rigid_object_motion_v1.py` 负责可微刚体场；`joint_rigid_object_features_v1.py` 负责五点当前上下文、共享位姿/值编码和全对象空间汇聚；`shared_rigid_native_bridge_v1.py` 负责三个decoder入口及物理场布局转换。各组件已实现并审查，实际原生模型四步配对预检已通过；正式512更新与dev200尚未运行，本文所有新模型成绩均未知，没有mock结果。

实际基础检查已完成：rigid_object_motion_v1 的6项测试在原H200 Python3.10.21/Torch2.1.2 CPU通过（CUDA未初始化）；crn_full_grid_geometry_v1在固定dev[0]完整640000点、432框上与原CV owner/velocity字节完全相同，macCPU构建3.358秒。此为单样本组件验证，不是原生模型预检或整个数据集吞吐。详细回执见 receipts/rigid_object_motion_v1_actual_cpu_test.json 与 receipts/crn_full_grid_geometry_v1_real_first_dev_cpu.json。

对象特征模块7项与bridge3项测试已在原H200环境CPU通过；当前几何适配器另在本机通过3项测试及固定dev[0]完整640000点复算，原CV、原owner、全部当前状态数组字节一致。回执分别为 receipts/joint_rigid_object_features_v1_actual_cpu_test.json、receipts/shared_rigid_native_bridge_v1_actual_cpu_test.json、receipts/shared_rigid_prediction_geometry_dev0_v1.json。以上均不是训练或模型性能结果。

接入解释边界：Fix的值分支仍接收学习后的δc/δω，因此不能要求其占据损失到位姿参数的全部梯度为零；只移除了位姿到空间核位置的连接。对象残差仅注入decoder入口，没有回注自回归memory；物理损失只训练共享对象模块，不直接训练原native future head。二维BEV柱空间核不直接区分center_z，高度可经值向量、完整旋转和三维尺寸的XY投影影响占据；对称框的个别旋转梯度可以自然为零。
