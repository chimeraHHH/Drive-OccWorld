# 历史 GT 姿态地址对照：预定设计审查

2026-09-16。**设计无实质阻断；本页未运行图像、模型或新统计。** 本诊断只问：保持原相机、原材料点与固定 7×7 ZNCC 后，历史框刚体轨迹给出的地址能否改善匹配，且改善是否超过同地址规则下的 broken 对照。它不是新候选、部署特征或本轮晋级门。

## 几何和缺测边界

仅用 raw slots `[−2,−1,0]`，在事先确定的相邻 sample 区间内检查同 instance、双方有效及直接 annotation 链。区间由时间决定，不按标注可用性或分数换区间；中间端点同时属于两闭区间时固定优先较早区间。中心线性插值，单位 quaternion 采用符号等价处理后的 shortest-arc SLERP，近同向用稳定归一化插值；不缩放尺寸，不以 clamp 掩盖越界，不补框、不外推、不访问 future 框。

GT bracket 和最晚可用姿态边界使用 **raw sample timestamp（箱姿态时刻）**；原 D 延拓仍保留 **camera timestamp − LiDAR timestamp**。两者差异应原样记账，不能交换时间域或统一修正“50 ms”。原 current 相机晚于 GT t0 的记录在该 reference 中记 missing；这不意味着原生已用 current sensor 是未来泄漏。

列向量路径为 `p_local=inv(A0) G0 [p0;1]`、`p_global(τ)=A(τ) p_local`、`p_camera=inv(GCτ) p_global(τ)`。同一当前材料坐标用于两幅图，不能再减一次 ego motion。GT current 与 past 地址都可能改变，因此这是**两端地址替换效应**，不是仅改变 past endpoint。原 camera selection、zero-current texture 门、D/zero true/broken 分数全部保留。GT patch 的纹理可描述，但不据此新增筛选；broken 仅沿原规则 roll past 图，GT current patch 不动，两者完全同支持。

## 最少输出与固定统计

令 `U_h` 为原 future-valid、CRN-uncovered 人口；优先展示原 no-radar 分层。`V_h` 为原 photo-valid，`C_h=V_h∩两端GT姿态可插值∩两端GT投影合法`。原无效、current/past 无 bracket 或断链、GT 投影越界须互斥记账；不得以 C 的覆盖替代 U。所有点继续取 `1/n_original_valid_object,h`，完整贡献除以原完整对象数；条件均值则除以该人口权重和，二者分开命名。

1. **支持账本。** 保留全部原身份/原 score、camera 和 mask 一致性检查；报告 U、V、C 的点数、对象数、原权重、完整分母收益/负代价触达量与各缺测项。记实际 bracket、插值比例及 camera/sample 时间差；不把 projected-valid 称为无遮挡。
2. **两个主配对比较。** 在同一个 C 上，分别计算 `ΔD=corrGT−corrD=eD−eGT`、`Δ0=corrGT−corrZero=eZero−eGT`；对 broken 同样计算 `ΔD_b/Δ0_b`，并报告 `ΔD−ΔD_b`、`Δ0−Δ0_b`。各报原对象加权条件均值、正/零/负比例和完整原分母贡献；不要把这种差分称为已识别的因果效应。原 future GT all/stationary/ambiguous/moving 各给总体；固定七 speed bins 只做 no-radar/all，不展开全交叉矩阵。
3. **一个次级连接表。** 用实际 past-camera 时刻的 `H_cam=GTdisp_R,past`，定义 `Bpast_cam=||H_cam,xy||−||H_cam,xy−dt_camera_to_lidar*vD,xy||`。在同一个 C 上报原 true/broken/speed 对 `Bpast_cam>0` 相对 `<0` 的加权 AUC，零收益单列、缺任一类返回 null。这是新 camera-time 刚体插值 proxy，不能覆盖或重命名旧 sample-time Bpast；不对非负 GT 几何误差硬造 benefit AUC。

四个未来 h 复用同一图像对及 GT reference，只因原 future 支持、权重、分组变化而分别展示，不是四份独立历史证据。未来 GT 分组仅供后评分，不能用于 source、view、bracket 或纹理选择。

## 可证伪解释

| 共同人口上的现象 | 支持的解释与下一动作边界 |
|---|---|
| GT 对 D 和 zero 的真实改善均为正，且超过各自 broken 改善，主要出现在历史确有材料点运动的记录 | 支持地址/运动假设对当前 matcher 有影响，值得再检查可观测表面与可部署对应；不证明 D 时间延拓是唯一瓶颈，也不证明能判别未来风险。 |
| 对 D 改善，但对 zero 无改善；或真实与 broken 同幅改善 | 前者可能只是 D 在静止位置制造误差，后者可能是换到更易匹配的纹理/位置。均不足以说明 matcher 已识别真实运动。 |
| 在历史运动及 D 历史误差较大的记录中，GT 地址仍不能改善真实匹配，或不优于 broken | 当前“虚拟体积点＋固定 patch”的观测连接缺乏支持。下一项应转向测得表面锚点与冻结对应来检验，而非继续调当前 patch/阈值；仍不能否定图像包含运动信息。 |
| GT 与 zero 地址相同或接近 | 应呈现地址距离与残差差的连续数值，允许零/近零的预期结果；不能要求静止点 GT 胜过 zero。future stationary 不等价于历史静止，也不据分数另挑“接近”阈值。 |

GT 框刚体、线性/SLERP 插值和虚拟体积点仍不等于 camera-time 实测可见表面。尺度/透视、遮挡、非刚体、重复纹理和曝光变化都可能令正确几何的 ZNCC 更差，所以它**不是数学 upper bound**。Bpast_cam 与未来收益仍共享 D 幅值，其 AUC 也不能单独证明时间因果。全部结论限定为 D 已见过的 train512 离线诊断；不能据 GT score 宣称部署 gate、运动模型或占据预测已改善。

依据：已冻结 [相机 core](../history_camera_evidence_v1.py)、[真实历史连接结果](../history_motion_alignment_train_analysis_v1.json) 与 [研究备忘](history_camera_frozen_flow_feasibility_v1.md)。本页只规定新对照解释，不更改任何既有源码、规则或结果。
