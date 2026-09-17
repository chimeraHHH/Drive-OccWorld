# O 与候选的共同输出评价合同（方案，未冻结）

2026-09-15。结论：**采用同一占据读出上的 GOSPA 作为附加的“未来空间集合误差”，与原生 fine GMO/FP/FN 同报；不把它直接叫 flow 或轨迹准确率。** O 不输出物体 ID/flow，不能人为给 O 接一个弱代理，再与新模型原生 motion head 比 EPE。下述代码是独立 NumPy/SciPy 数学与读出核心，已通过 CPU 解析测试；尚未在真实预测上测量组件分布、资源或验证评价有效性。

## 1. 文献依据和可识别性边界

GOSPA 的 `p=2, α=2` 形式允许把定位、漏目标、假目标代价相加。轨迹扩展 T-GOSPA 还惩罚跨时间的匹配切换，但需要**预测轨迹集合**，不能用若干逐帧集合分数的均值冒充。原作者提供 MATLAB GOSPA 和 Python T-GOSPA；后者为 LP 实现。本次不复制作者代码，也不引入 tracker。[GOSPA 原论文 Proposition 1](https://arxiv.org/pdf/1601.05585)、[作者 GOSPA 实现](https://github.com/abusajana/GOSPA)、[T-GOSPA 原论文](https://arxiv.org/pdf/1605.01177)、[作者 T-GOSPA 实现](https://github.com/Agarciafernandez/T-GOSPA-metric-python)。

不可识别的反例：两个同形状物体在两个位置交换身份，逐帧占据集合可完全相同；仅输入五帧 binary occupancy 的评分器无法判断谁去了哪里。另一个反例是物体形状缩小但质心不变：centroid-GOSPA 不变，occupancy IoU 会变。因此：

- GOSPA 改善可支持“未来占据区域的位置、遗漏与多余区域综合改善”。
- moving GT 分层改善可支持“真实运动对象对应的未来几何预测改善”，仍不等于速度/身份预测更准。
- 时间关联的位移诊断可以补充几何演化证据，但不得冒称 O 原生 flow，或忽略形状/关联误差。

Waymo 的 occupancy-flow 指标明确需要预测 flow，且 flow-warped 指标用它搬运 GT origin occupancy 再与预测 occupancy 相乘；这很适合新 motion head 的单独验证，却不能成为 O 缺失该输出时的共同公平主指标。[Waymo 官方指标源码](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto)。

## 2. 共同输入和无 GT 定位的读出

**共同输入：** 两种模型使用同一五帧最终 native logits、同一固定 t0 LiDAR R、同一 `512×512×40` 完整 GT。只按原 evaluator 做最后层 logits trilinear resize（`align_corners=False`）再 `argmax`；类别 1 为 GMO，平局仍选类别 0。原 `GT∈{0,1}` 有效评价 mask 同用于双方；它是评分域，不是未来框裁剪。不得读取候选 flow、GT 框中心或 GT moving mask 来决定预测组件的位置、数量、拆分/合并。

**固定、无拟合的候选读出：** 对完整有效前景沿 z 做 `any`，得到 `[X,Y]` BEV mask；用 4 邻接组件，无 closing/erosion、无面积筛选、无 top-K。组件位置是其 BEV 占据格中心的等权均值（米），不按预测置信度或高度加权。**GT fine occupancy 也用完全相同的组件读出**，而不是把预测组件质心直接和 GT 3D 框中心当同一种量。这样复制 GT 必须取得零分；不会因只可见半个车体的 GT 质心偏离框中心而给完美占据扣分。[SciPy label 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)。

这个指标应命名 `BEV-component GOSPA`，不能叫 instance GOSPA：一个实例可能碎成多个组件；两个相邻对象可能合并；高低重叠物体会在 BEV 合并；连接桥和断裂会改变数量；中心相同但轮廓不同不受罚。小碎片全部计入可能放大噪声，筛掉又可能掩盖漏人/假目标。第一版选择**不筛**并披露组件面积/数量、碎片及合并分布；真实组件数若超过算力预算，明确停止或另开发严格等价稀疏算法，不能悄悄丢点。正式采用前须做 GT 自身→读出审计以及固定已有 O 输出的可行性检查；当前未作此检查。

## 3. 核心匹配、单位和汇总

设同一帧的 GT 组件中心为 X，预测中心为 Y，γ 为一对一部分匹配：

`G² = min_γ [ Σ_(i,j)∈γ ||x_i−y_j||² + (c²/2)(|X|+|Y|−2|γ|) ]`。

暂建议 `p=2, α=2, c=4 m`，四个未来时域使用同一 c；c 的含义是距离超过 4 m 的匹配不优于“漏一个＋多一个”。这是**未冻结的度量尺度建议**，不是来自候选结果的最佳阈值。若报 `c=2/8 m` 敏感性，必须预先规定全部报告，不挑有利值替换主 c。代码仅实现固定 `p=2, α=2`，不提供漏检/误报不等价权重开关。

`common_output_gospa.gospa2` 用精确矩形线性分配和截断距离成本；距离 `≥c` 记为未匹配。返回 `G[m]`、`G²[m²]`、三项 squared-cost、全部配对及未匹配索引。**三项 m² 可相加；开方后的 m 分量不可直接相加。** 完全空集合相互得 0；多个 FP 累加，不按模型预测数量归一化。等成本最优解可能不唯一，总 G² 不变但匹配分解/运动分层可能变；须固定输入排序和 SciPy 版本，披露 ties 的解释边界，不能用微小改价把有利分层“固定”下来。[SciPy 精确分配接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。

逐样本、逐时域保留原始结果。建议主汇总按同一 scene 内平均 G²，再跨 scene 平均，最后开方；未来四时域等权，t0 单列。原生 GMO 仍保留其既定 pooled confusion→per-horizon IoU→future mean 口径，不把两者混用。配对区间使用固定场景重采样（seed11、10,000 次），不是训练多种子置信度。相同数据的历史曝光和模型选择仍需披露。新集合统计不能自动继承旧 GMO 的 +0.5 pp 门槛。

## 4. moving 分层：先全体匹配，再分解，不能造静态 FP

每个时域先把 **全部** GT GMO 组件与全部预测组件做一次 GOSPA。随后才用原始实例标注关联 GT 组件，用同一实例在世界坐标的位移/真实 dt 定义 moving；不能用预测 flow、GMO 类别或 ego 位移定义 moving。

关联属于评分阶段：依据该帧 GT 组件的真实前景体素与框的重叠建立 ID；只有归属同一实例且无重叠歧义时，才进入清晰实例分层。一个组件含多实例、框外不确定点或多个组件对应同一实例时，分别标为 mixed/unknown/fragmented，不以 GT 框重切**预测**组件。唯一关联比例和未关联量必须逐类报告；低覆盖则不将该分层作为决策依据。

全匹配中已匹配静止 GT 的预测保留为静止正确项，**绝不**拿它与 moving-only GT 再匹配后算 FP。moving/stationary/mixed/unknown 只划分 GT 对应的定位和漏项；未匹配预测的真实运动类别未知，保留 **global false** 单列，不凭最近 moving GT 或预测速度强分。`truth_group_breakdown` 正是此规则，输出不是各组独立 GOSPA，不能分别开方后声称可相加。

## 5. 可以补充的时间诊断，以及不能跨越的声明

如果前述 GT 组件↔原始实例关联有足够覆盖，可预先定义在 t0/h 都有唯一组件的 GT 实例集合 E；这个集合只由 GT 决定，对 O 与候选完全相同。每帧仍使用第 4 节全体匹配。在 i 的两端都匹配时，可计算

`e_geom = || (ŷ_i,h−ŷ_i,0) − (x_i,h−x_i,0) ||`。

这里 x 是 **GT 占据组件**质心，不是框中心；它测的是对应区域的质心演化误差。若任一端未匹配，不能删除该实例：另报缺失率，并可预先固定一个截断风险 `min(e_geom,c_motion)²`、缺任一端记 `c_motion²`，分母始终为 |E|。这是自定义 **censored geometric-displacement diagnostic**，不是 GOSPA/T-GOSPA 或官方 EPE；FP 仍由全集合 GOSPA 惩罚。不要只汇报两模型共同“幸存”的匹配样本，造成漏检被忽略。

若把式中的 GT 位移替换为真实框中心位移，则预测组件仍不是物体中心：遮挡、可见表面变化、形状变形会产生偏差。必须同时算“完美 GT occupancy 经相同读出”相对框位移的自身误差；不能把这个读出偏差归咎于模型运动。框关联仅在评分阶段使用且不改变 ŷ，但它提供了 GT-assisted correspondence，因此这项诊断**不评价模型自己恢复实例身份的能力**。

更强的轨迹声明需要可审计的、仅使用预测占据的共同关联器，或者两模型原生提供可比的轨迹输出；随后才可用作者 T-GOSPA。现在凭五帧 binary GMO 强行重建 O 的速度/身份并不唯一，因此本合同不制造这个代理。当前可靠的升级表述应限于“真实运动对象的未来几何更准确，并在同一读出下改善区域位移诊断”；若新 motion head 本身在 box-derived 标签下 EPE 更好，应另列为新模型的运动估计证据，不直接说它在 EPE 上击败 O。

## 6. 本次实际交付/验证

- `common_output_gospa.py`：`gospa2`、`truth_group_breakdown`、`bev_component_centers`；只依赖 NumPy/SciPy，不读模型/GT标注文件。
- `test_common_output_gospa.py`：13 项实际 CPU 测试通过，包含位移、删除、新增 FP、距离截断、非贪心最优分配与直接穷举部分匹配一致、空集合、静态匹配不变 moving FP、XYZ/4邻接、同质心形状盲点、身份交换不可识别。
- 尚未完成真实预测组件质量/数量测量，未冻结数据集评测合同，未评任何模型的新成绩，未训练或修改主 probe。
