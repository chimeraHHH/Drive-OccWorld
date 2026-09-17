# 对象状态与未来占据：下一条可否定路线

2026-09-16。本次只读现有结果、两篇原论文及作者代码；没有训练、推理或修改既有协议。**本次文献审读提出一个待选择方向：“可训练的未来状态怎样保留对象身份并产生占据”。** 根任务尚在等待 D/CV 共同诊断，再决定是否选择该方向。下述 slot 架构、S/I 两臂及判读条件均为子任务建议，未获根任务冻结，不构成实验启动或晋级授权。

## 现有证据限定了什么

直接读取 `future_feature_residual_dev200_pooled_v1.json`，200 anchors / 100 scenes：

| 模型 | 四未来 GMO IoU (%) | speed>0.5 m/s 正例召回 (%) | 四未来 FP | 四未来 FN |
|---|---:|---:|---:|---:|
| O | 14.66400 | 43.45212 | 13,647,997 | 3,732,899 |
| R，使用 D 位移 | 15.20117 | 40.71243 | 12,168,810 | 3,853,183 |
| R0，同容量零位移 | 15.23665 | 40.75618 | 12,198,534 | 3,842,160 |

R−R0 的 GMO 差为 −0.03548 pp，95% 场景配对区间 [−0.10282,+0.02471]；moving recall 差为 −0.04375 pp，区间 [−0.60567,+0.54086]。这未辨明已学习运动相对零位移的增量；R 相对 O 的 +0.53717 pp 同时伴随 moving recall −2.73969 pp。不能以召回、阈值或新的加权分数替换原 GMO 主量，也不能据此宣布运动改善。

任务上下文提供的 partial oracle 结果只支持局部合法来源点诊断；本次未重新核验其收据，不把它称为全 flow 上界。当前硬前景源支持不足是待查限制，尚不是 R 失败的唯一原因。

## 两篇原文与代码分别能支持什么

**PowerBEV，IJCAI 2023：本次新增源码核对。** 原文 §3.2–3.4 将未来语义占据与 backward centripetal flow 分开预测；后者从未来前景格指向前一帧同一对象的中心，含对象归属信息，不等同材料点物理位移。原文使用两条同结构、不共享权重的预测分支。[原论文](https://www.ijcai.org/proceedings/2023/0120.pdf)

作者 `MultiBranchSTconv` 的两条 STconv 分支分别输出 segmentation / instance_flow；训练的 CE 与 flow 回归汇总到同一模型，但 flow 不进入语义占据前向。实例 ID 的时序传播显式 `clone().detach()`，是后处理。因此，若在冻结 O 特征上照搬这两个独立头，它不能建立我们所需的“未来占据监督改变物理运动预测”的连接。它值得借鉴的是对象对应的表示，**不推荐将其 flow 直接当现有 D 的替代物**。[双分支代码](https://github.com/EdwardLeeLPZ/PowerBEV/blob/7cdce93e76f3a0e91fcefaf0ab7bfe33f7c53321/powerbev/models/stconv.py#L16)、[损失汇总](https://github.com/EdwardLeeLPZ/PowerBEV/blob/7cdce93e76f3a0e91fcefaf0ab7bfe33f7c53321/powerbev/trainer.py#L114)、[detach 后处理](https://github.com/EdwardLeeLPZ/PowerBEV/blob/7cdce93e76f3a0e91fcefaf0ab7bfe33f7c53321/powerbev/utils/instance.py#L137)。

**FipTR，ECCV 2024：深化已有阅读，不计作新发现。** 本地 `motion_connection_representation_review_v1.md` 已核过其 flow-aware feature 链。本次补查的是：同一实例 query 经时域映射参与各帧 mask 生成，训练分配的 Hungarian cost 累加全部时域 mask cost，让同一 GT 对象序列对应同一 query；未来 mask 的误差可作用于未来 BEV 生成器与实例 query。[原论文 §3.3–3.6](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11758.pdf)、[五帧 mask 生成](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/dense_heads/fistr_lss_head_timespecificmaskquery.py#L245)、[跨帧匹配成本](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_mask_strong.py#L109)。

源码还限制了可迁移结论：实际采样偏移是 `Linear[query, predicted_flow]`，并不被硬约束为真实位移；其 BEV flow 标签是框栅格中心的相邻帧差。代码会以前帧掩码填补某些缺失帧，并对标签为 255 的 flow 位置另施加零流惩罚。这些是作者任务配方，不能照搬成我们的 unknown 或真实刚体运动监督。其 2D 车辆框掩码、VPQ/IoU 也不能直接与原 3D GMO 数值比较。[偏移生成](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/modules/flow_guided_self_attention.py#L239)、[缺失处理和中心流](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/pipelines/motion_labels.py#L414)、[255 位置零流损失](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/dense_heads/fistr_lss_head_timespecificmaskquery.py#L720)。

两库均已固定 commit 并读取相关源码，**没有安装、导入、执行作者模型或验证其 checkpoint**。DriveWorld 本次只定位到 CVPR 原文，未验证官方可运行仓库；不列为候选，不用额外搜索凑篇数。

## 本次仅提出一条待选择候选：对象状态进入可训练的未来生成过程

**假设。** 单纯给终端占据读出补充运动特征，不足以迫使未来表示保存“同一个对象将去哪里”。若对象身份、位姿变化及其空间支持参与未来状态生成，并受到同一占据目标训练，则可能同时改进原 GMO 和受支持对象的位移预测。这里的“可能”需要下述单一判别实验；查询表示及可导搬运本身均已有先例。

建议架构是本项目推演，**不是 FipTR 的直接复现**：

1. 原 O 作为不可变参考。新候选复制 O 的 future head 为可训练初值，观察器及现有 z0 缓存仍冻结；不要把终端 O future latent 和 log-odds 当唯一可训练支路之外的定稿预测。保留原原生时序/坐标合同，在每次未来 transition 内加入对象状态的读写。
2. 从 z0 预测一组对象 slots，每个 slot 包含 t0 软空间支持、共享身份特征，以及四未来中心位移和旋转。预测位姿通过可导的刚体空间映射把对象特征写入下一未来状态；该状态再经可训练 future head 产生原 3D 占据。占据梯度必须到达对象状态、软支持和未来生成参数，物理监督到达同一状态；不能再让 motion 完全冻结，或仅把 flow 接在最终 logit 残差上。
3. 软支持由特征学习，不与 O 的硬前景 mask 相乘。这移除了当前实现强加的硬源筛选，但不保证补出了实际未观测对象。保留场景稠密分支，以承载未分配对象、新生、背景及未知区域；完整 GMO 始终评价所有原有效体素。
4. 不照搬 FipTR 的所有任务损失。保留 O 的 CE/Lovász 占据目标，新增的对象身份/位姿监督必须对应下面已有或可合法构造的标签。三层/五时域、评价轴和物理单位需沿原合同；新增损失系数及参数集合在开发结果前固定。

这与 R 的实质差别是**未来生成状态和对象状态共同训练，运动写入发生在未来生成过程中**。单独把现有 R 的 8 通道改宽、增加门控，不能检验该假设。

### 监督可用性：现成、可构造、缺失分开

| 内容 | 当前证据与边界 |
|---|---|
| 跨帧对象身份、3D 中心/旋转/尺寸 | 原 raw tracks 保存七帧 `instance_token`、`valid_mask`、global center、wxyz rotation、wlh size；明确 `missing_fill_applied=False`。可使用已标注帧，缺失不能补前帧或记静止。 |
| 原 t0 来源点到四未来的物理位移代理 | sparse 标签已有 `object_index`、`instance_tokens`、`center_delta_R_m`、`target_displacement_m`、有效掩码；可沿原同一来源点集合评价。它仍是框刚体代理，不是真实非刚体 scene flow。 |
| 精细 3D 实例占据掩码 | **现成材料未证明已有。** 可以拟构造“原完整占据 GT 正例 ∩ 唯一合法框归属”的部分实例监督；重叠框、无框占据和无合法对应处保持 unknown。框内空间不是全部真实 occupied，不能用填满框替换原 occupancy GT。 |
| 未标注对象、新生且 t0 无对应者 | 可保留原占据监督；不能凭缺失轨迹制造物理位移或稳定身份真值。当前 dense 场景支路必须继续覆盖它们。 |

本地依据：`build_motion_targets_v1.py:234–251`、`build_sparse_motion_supervision_v1.py:180–198`、`motion_geometry.py:74–99`。本次核过提取源码，未重新逐条盘点全部 raw tracks；因此新 partial instance target 生成器、覆盖统计、软支持训练与 3D slot 写入均为**尚未实现和验证**。训练前需核实对象数上限、未知比例、跨帧身份/轴/旋转及 source-support 覆盖，不能把本报告当作这些检查已经 PASS。

### 候选判别实验草案：序列身份是否产生双目标增量（未冻结）

若后续选择本路线，建议采用同一新架构、同一 O 初始化、同一 seed 11、train512 / dev200、样本顺序和每臂 512 更新作两臂对照；该预算及比较尚未由根任务冻结：

- **S：序列匹配。** 一个 slot 在所有有效时域对应同一对象；按有效 mask/位姿组成序列匹配成本。
- **I：逐帧匹配。** 相同容量、相同预测状态/掩码、相同有效标签与损失项，各帧独立匹配。它仍得到每帧检测/占据监督，区别是没有固定跨帧身份约束。不要改成随机错误轨迹负对照。

两臂均使对象状态参与可训练未来生成；这样主比较 S−I 检验身份连续性的增量，S−O 检验实际原任务提升。物理评价先在 t0 以同一规则匹配预测 slots 与原合法 GT 对象，然后固定该对应到四未来，禁止逐未来重匹配掩盖身份交换。对全部原有效来源点按预测刚体状态计算原代理 EPE；无法匹配/容量溢出的对象须显式计数并判为未完成可比评价，不能只报告成功匹配子集。继续给出四时域、各速度组、FP/FN 和到达/离开占据变化，避免只显示平均提升。

**建议的双目标判读条件（非已冻结晋级门）：** 原四未来 GMO 超过冻结 O，且序列身份相对 I 有可辨明增量；同一合法支持的物理代理 EPE 超过现有已训练 D，并优于 I。若仅 GMO 提升、仅 moving recall 改变、仅 EPE 胜过 zero、或仅 VPQ 改善，都不视为完成“占据及运动同时改善”。场景配对区间可以描述固定模型差异，仍不覆盖单 seed 和开发曝光的不确定性；这里只是开发筛查，full 原任务验证仍为后续独立步骤。

**建议的停止条件（待根任务确定）：** 标签/容量/坐标合同不成立则不启动；同预算 S≈I 则停止“身份连续性足以带来增量”的当前实现；S 的 GMO 提升但 EPE 不改善，则记录任务修正、停止双目标成功叙事，不再靠追加门控或改开发指标补救。若新状态分支在少量真实 train-mode 预检中没有占据到位姿的有限非零梯度、资源超限或无法学习对象支持，也先停止，不用 dev 选权重。

低预算依靠冻结昂贵观察器、复用 z0、单 seed 两臂和固定 512 更新，不训练外部基线。slot 容量、稀疏/分块实例归约和训练资源必须由 train 数据与真实预检确定；**未测吞吐、峰值或承诺小时数**。本报告不授权扩大现有实验预算，也未触发自动运行。

## 来源回执

[专用来源清单](dynamics_representation_next_routes_v1_sources.json) 记录论文 URL / SHA、官方 repo commit、逐文件 SHA、重读/新增范围及本地证据 SHA。作者源文件与论文快照保存在 `dynamics_representation_source_reads_v1/`；均未执行。本文只提出上述一条路线，没有以未验证的代码、额外数据或外部模型训练作为前提。

本次来源清单最初使用通用名 `sources.json`。写入前的工具检查记录明确显示该文件不存在，未覆盖历史内容；现已更名为上述专用文件名，通用路径不保留本次副本。
