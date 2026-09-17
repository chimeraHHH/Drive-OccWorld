# 运动连接作用于概率还是特征：五点有界判断

本次只核 FipTR、Occupancy Flow Fields、Softmax Splatting 三项原论文／作者代码；未读取未完成 J/D 的开发结果，未修改训练或冻结源。

1. **当前证据支持检查表示瓶颈，不支持立即判定概率传播路线错误。** 已完成 A−O 的 GMO +0.086514 pp，却有 moving 前景 recall −0.826871 pp、FP −248,032／FN +20,546；这是同一连接的不同取舍。现行 `supported_motion_fusion.py:39–49,77–95` 用当前 argmax 源、未归一化散射质量 S、以及 O 与 logit(S) 的凸组合。严格可推导：两支都低于前景阈值时门无法恢复前景；W=0 只能回到 O；错误运输覆盖区域却能压低 O。正确位移也不保证离散 S 是校准占据概率。简单改成 S/W 不是中性修复：所有源 p0>0.5 时，W>0 处的 S/W 也>0.5，实质接近把运输支撑全部当作前景。上述为本项目代码推导，尚非实测主因。

2. **特征传播的价值是把对应关系交给解码器解释，而不是保证 flow 更真实。** FipTR §3.3／式(3) 将 backward flow 与 query 拼接生成采样 offset；未来特征再解码为实例 mask。作者 head:257–284、attention:239–280 可见该可导链。其 Table 3 的远域 IoU 有不变／略降，不能引用为普遍优势。我们若后续比较特征载体，须保留 O 补全支路，用同初始化、同位移监督、同训练预算和同容量零位移对照；不能将 97 参数门与更大解码器直接比较便归因于“特征更好”。现有 2D BEV latent 与每高度的 3D source displacement 也需先定义真实映射，不能直接混用。特征分支在同物理 frame 的外接解码，不等于重启已否定的内部坐标改写。

3. **碰撞处理和携带什么信息是两个变量。** Softmax Splatting §3.1–3.2／Table 1 使用归一化重要性权重、传播多尺度特征后做图像合成；作者 `softsplat.py:232–270` 明确分子／分母。它的插帧实验并非我们预测未来占据的证据。可证伪预测是：若瓶颈来自把 S 当概率，而已有位移有用，那么同容量、同源支撑、固定相同位移的特征载体应在保留运动 TP 上优于标量载体；若只是更强局部解码，零位移特征控制也会同样改善。故下一比较要固定支撑／位移和补全能力，不能同时换支撑、碰撞归一化和网络后声称识别了载体作用。

4. **概率级联系已有成功先例，也已有“占据更好、flow 更差”的原始反例。** Occupancy Flow Fields §III-F 式(3–5) 从输入当前占据构建 backward-flow trace，再以 W_t O_t 的损失约束独立预测；不是直接用 trace 替换所有未来输出。Table I（PDF第6页）Interaction 1.2s 加 trace loss 后 soft-IoU .551→.557，EPE .591→.736（格单位），说明一致性改善不等于物理运动误差改善。原任务使用轨迹／地图等输入及 BEV box 占据，与当前相机雷达／细粒度 GMO 不同。官方指标代码 `compute_occupancy_flow_metrics`、`_compute_flow_epe`、`_flow_warp` 保留 occupancy／flow 分量；所核 commit 的 `_flow_warp` 已不可运行，不能声称复现了作者模型或该算子。

5. **J/D 只回答当前载体中任务梯度接通的效应，不直接比较两种载体。** 若 J 相对 D 的占据与共同 moving 风险改善，且相同有效对象集合的物理代理 EPE 也改善，支持先保留现有连接并完成原任务验证。若只涨 IoU、EPE 变差或 moving recall 继续下降，应称任务修正倾向，不能宣称运动更准确；若物理 EPE 改善但占据／运动 TP 仍不改善，才更有理由把“位移已可用但表示／支撑限制其利用”作为下一待否定假说。若 J≈D，也可能是梯度弱、预算／优化限制，不能独断为 carrier 无效。各结果均保留静止组、全 FP/FN、四时域和历史开发曝光，不事后改成功标准。flow-guided feature、可微 splatting、占据-flow consistency 和 task-specific flow 都已有先例；模块组合本身不是新颖性，物理位移与任务修正的可识别边界及受控证据才是待研究内容。

**本次实际查看的一手来源（3项工作）：**

- FipTR：[正文 §3.3、§3.6、Table 3](https://arxiv.org/html/2404.12867v2)；[作者 attention:239](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/modules/flow_guided_self_attention.py#L239)；[作者 head:257](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/dense_heads/fistr_lss_head_timespecificmaskquery.py#L257)。正文在线打开，代码与本地固定原文对读。
- Occupancy Flow Fields：[正文 §III-F、Table I](https://arxiv.org/pdf/2203.03875)；[官方 metrics:155](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/utils/occupancy_flow_metrics.py#L155)。均在线打开。
- Softmax Splatting：[正文 §3、Table 1](https://arxiv.org/pdf/2003.05534)；[作者算子:232](https://github.com/sniklaus/softmax-splatting/blob/43df87a387440446388f8aec72976c5e7a4a3996/softsplat.py#L232)。均在线打开，算子与本地固定原文对读。

已有项目证据：[共同开发评测](common_change_dev200_v2_结果与决策.md)、[当前连接决策](connected_motion_next_decision_v1.md)。文献给机制先例与反例，不把其分数、坐标或性能保证转移到本项目。
