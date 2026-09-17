# 二分类任务中的 scaling 约束

2026-09-15。源码与已完成真实 soft-count 记录的再分析；没有调用模型、计算新梯度、修改训练或读取部分 full 结果。

**在当前 GMO 二分类任务中，两个 scaling 项重用了 precision/recall/specificity 统计。它们的名字不能证明我们有两类不同的几何、语义监督；重用也不能直接证明损失有害。** O 删除这些项后的机制仍须与梯度尺度、实际优化轨迹和完整验证结果区分。

## 先确定“几何”实际约束了什么

冻结 S0 为 `num_classes=2`、`use_separate_classes=False`、`use_background_classes=False`，四个 loss 家族权重均为 1。`LoadOccupancy.get_seq_occ` 的 106–110 行把原标签 `[0,1,8,11,12,13,14,15,16]` 合并为 0，把 `[2,3,4,5,6,7,9,10]` 合并为 GMO=1。这里 0 含静态背景及未占据体素，不等于物理自由空间；255 才按原路径排除。GMO 是可移动语义类，不等于此刻确实运动的体素。

`geo_scal_loss(..., non_empty_idx=0)` 对这个已经合并的 target 计算 `target!=0`；因此它在本任务中同样监督 GMO 与非 GMO 的区分。不能据函数名把它解释成额外的真实几何、距离或自由空间观测。本文没有否定在原始多类场景补全任务里构造占据/非占据统计的作用。

源码 SHA：S0=`c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303`；`semkitti_loss.py`=`e6a4b9accef1f47cf031f84e18fac925d2764b4105b8010486c8229e589abaa4`。标签映射来源是父协议绑定的 `loading_occupancy.py`（SHA `ac9190193630eca9f3118136e2d0b46bf528217f50244e37e6fc1cbff4841d1e`）。

## 有条件的代数展开

考虑单个 decoder layer 在原有效 GT 范围上池化当前及四个未来时域。令正类为 GMO，soft TP、FP、TN、FN 使用精确互补的概率 p 与 1−p，定义

\[
 P=\frac{TP}{TP+FP},\quad N=\frac{TN}{TN+FN},\quad
 R=\frac{TP}{TP+FN},\quad S=\frac{TN}{TN+FP},\qquad \ell(x)=-\log x.
\]

假设两种 GT 类都出现、相关分子分母为正、处于 BCE 的普通可微范围，并先忽略浮点误差。在二分类中，class0 的 recall 等于 S、specificity 等于 R，class1 的 precision 为 P、class0 的 precision 为 N。因此

\[
 L_{sem}=\tfrac12\{\ell(P)+\ell(N)\}+\ell(R)+\ell(S).
\]

设原 geo 的分母常数 ε=10⁻⁵、GT 正负数为 G₁、G₀，则

\[
 L_{geo}=\ell(P)+\ell(R)+\ell(S)+\delta_\epsilon,\quad
 \delta_\epsilon=\log(1+\epsilon/(TP+FP))+\log(1+\epsilon/G_1)+\log(1+\epsilon/G_0).
\]

两个 GT 分母修正对预测参数是常数；预测总质量分母的修正不是常数。权重均为 1 时，合计为

\[
 L_{sem}+L_{geo}=\tfrac32\ell(P)+\tfrac12\ell(N)+2\ell(R)+2\ell(S)+\delta_\epsilon.
\]

这给出了重复统计项的有效权重，而不是“多两个损失就多两种独立信息”。它与全数据每个未来时域分别池化再平均的 GMO IoU 不是同一个函数：原损失还包含当前时域，并按 anchor 和 layer 组织。重用统计可能是有意的风险权衡；上述恒等式本身既不证明应该删除，也不提供 ICLR 新颖性。

## 原实现为什么不能直接当作这个恒等式

`sem_scal_loss` 用 softmax 的两个通道各自计算；`geo_scal_loss` 用 `1−p0`。FP32 中 `p1` 与 `1−p0` 不必逐位相同。sem 按实际存在的 GT 类计数，缺类时还跳过相应项；不能把上面“两类均出现”的展开延伸到所有样本。geo 带 ε 而 sem 没有这一分母常数。

此外，PyTorch 2.1.2 的 CUDA BCE 前向把 log 截在 −100；反向把 `p(1−p)` 分母限制在至少 10⁻¹²。因而普通 −log 的代数或梯度推导不能越过这些数值分支。[实际版本源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/native/cuda/Loss.cu)

## 用已完成的真记录检查展开范围

重读并认证 `supervision_granularity_v1` 的 4 个模型（M0_fp32、C、persistent2、rolling2）×固定 16 个训练 anchors×3 层，共 192 条池化 soft-count 记录。**这些记录没有 F/O**，不能当成 O 的新前向或梯度证据。全部 192 条记录的两类 GT 都存在；按保存的比值，没有进入上述前向 log 截断或反向分母下限范围。这里的判断基于保存统计，不声称逐位重建 CUDA 输入或执行分支。

从各项自身保存的 soft 统计，以 float64 重算对应 scalar，与原 FP32 loss 的最大绝对差：sem 为 **1.069×10⁻⁶**，geo 为 **2.629×10⁻⁶**。这是近似 scalar 复算，不是 CUDA bitwise parity。将各通道已舍入的统计代入上面的理想合并公式及 ε 修正，最大差扩大到 **0.0360722**。这是统计代入的残差，没有重建一组严格互补的概率张量，也不是该理想恒等式的反例。

最大差对应 M0_fp32、先前事后关注的稀疏 anchor `706d7015b80d4329b80d4a8c52139ab4`、layer1。GT GMO 只有 147；语义通道的 soft intersection 为 6.56786×10⁻⁵，而 `1−p0` 路径为 6.47306×10⁻⁵。原 sem+geo 合计 50.12518，理想合并值约 50.08911。原始 192 行全部保留，未删除该 anchor，也没有由此把误差解释成训练退化原因。

记录与脚本：`binary_affinity_reanalysis_v1/{summary.json,numbers.csv,complete.json}`、`binary_affinity_reanalysis.py`。来源 summary SHA `f76c849cfd28fb74ec8c3ad38856d56b8ec47628813268970ae0d61ca6d08dd6`，各模型 JSONL SHA 已逐一核验。没有新增 mock 数据、模型调用、梯度、optimizer update 或性能分数。

因此，后续应研究**任务定义、统计汇总方式与优化目标如何对应**，而不能把本轮理解为几何模块失效，或把“重复了若干项”直接包装成一个新算法。现有真实 VJP 表明 sem/geo 高度同向；它支持分析共享统计和权重尺度，不支持“两个损失相互冲突”的叙事。完整 O 效用、只读特征/读出诊断及尺度控制应分别提供各自范围内的证据。
