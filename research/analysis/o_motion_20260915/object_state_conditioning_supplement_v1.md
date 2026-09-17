# 对象状态到稠密占据：三项补充及本轮设计风险

2026-09-16；已避开此前 MotionPerceiver／FIERY 等卡。仅读原论文与作者仓库，未执行外部模型、训练或改变 V/G。**没有找到足以直接证明“有噪声的检测状态经过 32 个 latent 就能可靠驱动完整 3D 未来占据”的证据。** 以下分别补充直接连接、空间关联和概率语义，不能互相替代成一个已经验证的方法。

## 1. HPP：直接的运动条件化先例，但公开实现有核验边界

[HPP 原文 v1](https://arxiv.org/html/2402.02426v1#S3.SS3)（2024）§III-C、式(2–6)将预测轨迹位置编码、模式特征及概率融合为逐对象／逐时域特征，再由 BEV 查询交互并形成实例占据；§III-F 联合训练轨迹和占据。它已经覆盖“预测运动前向驱动占据”的一般思路；没有给出 CRN 位置／速度噪声的受控鲁棒性保证。

作者仓库固定 `3f99a809fa97817bf19b32167e28c0b1697ea01b`。实际 [occ_head.py L173–291](https://github.com/georgeliu233/HPP/blob/3f99a809fa97817bf19b32167e28c0b1697ea01b/hpp/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L173) 是逐实例 `traj_query + track_query + track_query_pos`，模式经 MLP 后 max 汇合；BEV 通过学习的实例 mask cross-attention 读取它们。该 `merge_queries` **不直接读取数值 `traj_preds/traj_scores`**，故不能把论文式(2)的完整数值轨迹连接或 DOPP 变体宣布为已由该入口复现。训练用实例匹配标签、BCE/Dice；[L335–459](https://github.com/georgeliu233/HPP/blob/3f99a809fa97817bf19b32167e28c0b1697ea01b/hpp/mmdet3d_plugin/uniad/dense_heads/occ_head.py#L335) 显示 unmatched query 的监督忽略、无 query 时输出零。因此不是我们保留稠密 O 和所有 GT 的可直接替换基线。README 有权重链接，本次未下载或验证其可用性。

## 2. StreamOcc：2026 年的显式对象—空间关联，任务是当前占据

[StreamOcc v3](https://arxiv.org/html/2503.22087v3#S3.SS2)（2026-06-21；作者标注 ECCV 2026）§3.2、式(6–10)先由体素修正对象 query，再按预测框邻域把对象信息注入稠密体素残差；§4.4／附录讨论 query 选择。它处理真实图像感知及误检，但评价的是**当前**占据重建，不是多时域 forecasting 或材料点 EPE。

固定 `6258be52d1bec9378c1d3670e2e293e50f6b9969`；[dqa.py L240–340](https://github.com/moonseokha/StreamOcc/blob/6258be52d1bec9378c1d3670e2e293e50f6b9969/projects/mmdet3d_plugin/models/heads/queryagg/dqa.py#L240) 实际以旋转框角点的离散 min/max **轴对齐包络**建立 query→voxel 边，不是精确 OBB 内点判定；[L445–487](https://github.com/moonseokha/StreamOcc/blob/6258be52d1bec9378c1d3670e2e293e50f6b9969/projects/mmdet3d_plugin/models/heads/queryagg/dqa.py#L445) 按目标体素归一化邻接 attention，保留稠密残差及 FFN。不能把“无邻边”概括为整个模块逐字节恒等。[queryagg.py L352–401](https://github.com/moonseokha/StreamOcc/blob/6258be52d1bec9378c1d3670e2e293e50f6b9969/projects/mmdet3d_plugin/models/heads/queryagg/queryagg.py#L352) 训练筛选还读取 GT 框／类别成本，推理只按预测置信度；这条训练选择逻辑不能迁入我们的仅预测状态输入。配置为 900 queries，而不是 32 个无对象身份的 latent。

## 3. TrajFlow：状态不确定性先例，不等于完整占据风险

[TrajFlow v2](https://arxiv.org/html/2501.14266v2#S4)（2025-08-02）§4、式(12)/(28)对单对象历史位置及速度等状态编码，以未来位置 NLL 学习条件密度；§6 的占据图是密度的派生可视化。数据为 ETH/UCY／inD 轨迹，不是对真实检测漏检的完整占据评价。预测中心密度、体素被任何物体占据的概率和刚体点流是不同量。

**v2 链接的官方仓库是 UMN-Choi-Lab/TrajFlow**，固定 `e7fe0339e004e4a2703e482521e96f3feb13da88`；早期个人仓库仅留作检索记录，不混充 v2。已核 [model/TrajFlow.py L103–167](https://github.com/UMN-Choi-Lab/TrajFlow/blob/e7fe0339e004e4a2703e482521e96f3feb13da88/model/TrajFlow.py#L103)、[train.py L14–28](https://github.com/UMN-Choi-Lab/TrajFlow/blob/e7fe0339e004e4a2703e482521e96f3feb13da88/train.py#L14)：条件来自历史，未来位置仅用于密度训练；[visualize.py L23–59](https://github.com/UMN-Choi-Lab/TrajFlow/blob/e7fe0339e004e4a2703e482521e96f3feb13da88/visualize.py#L23) 在空间网格查询密度后按每帧最大值归一化显示。不能把这种图当作我们可比较的校准 occupancy probability，或据此删掉未检测对象。

## 对 32 latents → native query 的具体推论（尚待本项目证据）

- **速度有影响，不等于对象对应正确。** 现有速度置零预检能证通路，但全场变化可能只来自速度总体分布。若正式结果需要进一步定位，可在固定 checkpoint 上仅置换同一输入对象集合的速度分配，保留速度多重集、全部几何、分数及原评价分母，比较对应的逐对象物理误差与完整占据变化。正确分配与置换接近，只说明该干预下缺少对应利用证据；不证明数据无信息，也不是新增晋级门。
- **32 不是“最多能表示 32 个对象”的定理。** 但全部低分／重复框竞争同一组 latent，且 score 只是输入特征，可能产生混合或稀释；HPP 保留实例轴、StreamOcc 保留局部邻接给出了不同归纳偏置。若出现“单对象速度可敏感、拥挤场景不敏感”，才有理由研究压缩／关联瓶颈；不能仅凭 latent 数量直接扩大模型或据 attention 热图断言因果。
- **硬几何连接也会传递位置误差。** StreamOcc 的包络和训练 GT 筛选不是无代价答案。当前保留 O 稠密通路、不按检测框屏蔽输出的选择仍合理；若以后检验软相对位置关联，应固定输入集合、容量和训练预算，并在明确坐标契约后比较，不能把本项目 native hidden index 自动视作物理 R 坐标。
- **p0+h·v 是状态假设，不是校准的未来分布。** 先保留本轮 V/G 与固定 CRN-CV 强参照，按原四时域物理／占据证据决定是否需要不确定性模型；不因 TrajFlow 的存在就增加 flow-matching／采样头或替换损失。

三个仓库的本次树／releases 核查均未识别 LICENSE；StreamOcc 与 TrajFlow 未在 README／树／releases 找到已训练权重入口，不能推断作者其他渠道不存在。StreamOcc 源头还标注 all rights reserved。这里是可阅读的先例，不承诺可直接复用发布。精确 URL、commit、阅读范围、SHA／Git blob 核验及限制见 `object_state_conditioning_supplement_v1.sources.json`，未进行可运行性或论文成绩复现。
