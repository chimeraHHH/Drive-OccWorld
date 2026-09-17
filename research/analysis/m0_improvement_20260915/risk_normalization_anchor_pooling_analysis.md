# 风险归一化：逐 anchor 损失、全局 GMO 与实际更新

2026-09-15。读取现有源码、三份审计文档、真实 VJP 收据和已完成 C 的训练日志；只做 CPU 标量重算。未连接服务器、未增加前向/训练、未修改 GT、冻结源或协议。F/O 的本地日志仍是不完整快照，本报告不分析其未完成结果。

**可成立的结论是：当前目标给每个 anchor 的非线性风险相同外层权重，与最终跨 anchor 池化的 GMO 不同；稀疏样本的异常初始梯度也延续到了包含它的真实累积组。尚不能成立的是“它独自支配实际 AdamW 位移并导致 GMO 下降”。** O/F 的胜负只识别既定训练配方下的干预效用，不能单独完成后一项因果归因。

## 1. 源码对应的风险及归一化

固定一层和一个 anchor，把当前＋四个未来时域的原 coarse 有效体素共同记为集合 V。令前景概率 p_j、二值 GT y_j，并定义

\[
N=|V|,\quad G=\sum_jy_j,\quad S=\sum_jp_j,\quad I=\sum_jy_jp_j,
\quad J=\sum_j(1-y_j)(1-p_j).
\]

在两类都有 GT、分母非零、忽略 geo 的 ε 且概率处在 BCE 未饱和区间的实数近似下，

\[
P_1=I/S,\ R_1=I/G,\ P_0=J/(N-S),\ R_0=J/(N-G),
\]
\[
L_{geo}+L_{sem}
=-1.5\log P_1-0.5\log P_0-2\log R_1-2\log R_0.
\]

这由本地 `semkitti_loss.py` 的 geo 三项加 sem 两类平均直接展开，包含对前景召回和背景召回的重复约束。实际运行仍是原函数：geo 用 `1−softmax0`、分母加 `1e−5`；sem 用各类 softmax 并对不存在类跳过。PyTorch BCE 还有 log 下限与反向分母下限。因此上式用于解释内部区间，**不宣称精确重建饱和区的 FP32 梯度**。[PyTorch 2.1.2 CUDA BCE 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/native/cuda/Loss.cu)

“按风险归一化”必须先说明归一化对象。上述比值已近似对重复体素数不变：把同一组概率与标签重复 c 次，I/J/S/G/N 同乘 c，四比值不变。再次除以 N 或 G 会改变目标，不是消除一个遗漏的 batch 平均。加权 CE 则是

\[
L_{CE,s}=\frac{\sum_{j\in V_s}w_{y_j}[-\log p_{y_j}]}{\sum_{j\in V_s}w_{y_j}},\qquad(w_0,w_1)=(1,5).
\]

再对 anchor 平均，得到的是各自加权均值的平均，不是全数据自然分布的 proper NLL。Lovász 也在每个 anchor 的五时域 coarse 有效位置上展开排序。三个 decoder 层各计算四族，训练总和没有自动变为“最后层、四个未来时域、完整 GT 的全局 GMO”。[本地源码与轴证据](objective_metric_audit.md)

### 稀疏性为什么值得查，但不能单独推出梯度爆炸

对单项前景负 log recall，标签固定时

\[
\nabla_\theta[-\log(I/G)]=-\nabla_\theta I/I,
\quad \partial L/\partial p_j=-y_j/I.
\]

小软交集会放大概率空间敏感度；但 G 并不是直接留下的一个独立 `1/G` 参数梯度放大器。反例：若所有正例概率均为 q，且各正例有独立二分类 margin z，则

\[
\partial L/\partial z_j=-(1-q)/G\quad(y_j=1).
\]

softmax 的 q 因子抵消了 `1/q`；转到共享参数还要乘网络 Jacobian 并叠加其他项。故“GT 很少，所以参数梯度必然异常大”不是定理。当前大梯度由实际 VJP 证明，不能从低前景占比 alone 推定到其他场景。前景趋零、概率饱和、epsilon、softmax 浮点相消也不能混成同一个机制；上一真实统计已表明相消导致的总预测质量差很小，不能据出现次数宣布它是主因。

## 2. 全局比值与逐 anchor 风险不能互换

对某一未来时域 h，正式指标是

\[
Q_h=\frac{\sum_sTP_{sh}}{\sum_s(TP_{sh}+FP_{sh}+FN_{sh})},\qquad
Q=\tfrac14\sum_{h=1}^{4}Q_h.
\]

若 U_s 是该时域每个 anchor 的 union，`Q_h=Σ_s(U_s/ΣU)·IoU_s`。这既不同于 anchor 的等权 IoU，也不同于训练中的等权负 log 软 P/R。**U 含预测，所以它不是一个只由 GT 前景数决定的固定权重。** 对可微软 IoU 的相应恒等式，

\[
\nabla Q=\frac{1}{\sum_sU_s}\sum_s(\nabla I_s-Q\nabla U_s).
\]

用固定 `G_s` 乘逐 anchor loss，或把 `U_s` 当常数而忽略其梯度，都不会一般地得到这一全局梯度。硬 argmax 指标本身也不能按这条软导数反向传播。相应地，全局 recall 是按 GT 正例数加权的局部 recall，而全局 precision 按预测正例质量加权；逐 anchor scaling 则给极稀疏但非空 anchor 一个完整的负 log recall 惩罚。

一个纯代数反例（不是实验数据）：两个 anchor 的 GT 正例数固定为 1 和 100。模型 A 的 `(TP,FP,FN)` 为 `(0,0,1)` 与 `(90,10,10)`；模型 B 为 `(1,0,0)` 与 `(80,20,20)`。逐 anchor 平均 IoU 从 40.91% 升到 83.33%，全局 IoU 却从 81.08% 降到 66.94%。这证明聚合目标不等价，**不证明我们的每次更新采用了这个具体交换**。Lovász 原论文也明确区分图像/小批次与数据集 IoU；其名称不能提供当前三层、粗 GT、五时域损失到完整未来 GMO 的一致性保证。[Berman 等，CVPR 2018，§3.1](https://arxiv.org/html/1705.08790)

F 改变 GT 参与位置和插值梯度，仍保留逐 anchor 比率；O 去掉 scaling，但仍是加权 CE＋逐 anchor Lovász。二者都没有自动把训练目标变为严格全局 GMO 风险，也都不是新颖的“风险归一化方法”。

## 3. 从 raw gradient 到更新：不能跳过四层转换

设第 t 次更新中四个真实 train-mode 微步的梯度为 g_{ti}。原循环先反传 `loss/4`，形成

\[
\bar g_t=\tfrac14\sum_{i=1}^{4}g_{ti},\qquad
\tilde g_t=\alpha_t\bar g_t,\quad
\alpha_t=\min(1,35/(\|\bar g_t\|_2+10^{-6})).
\]

日志 `grad_norm` 是累积后的**裁剪前**范数。全局 clip 缩放整个总向量，基本不改变其方向，也不重新平衡四个样本的相对贡献。它会共同缩小同组另外三个样本的梯度，但我们没有这四个独立向量，不能由一个总范数反推出其贡献。[PyTorch 2.1.2 clip 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/nn/utils/clip_grad.py)

AdamW 随后用裁剪后的向量更新历史状态：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)\tilde g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)\tilde g_t^2,
\]
\[
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\widehat m_t/(\sqrt{\widehat v_t}+\epsilon).
\]

这是坐标级预条件化与独立 weight decay；该执行路径不会把 raw gradient 的 L2 比例直接转成参数位移比例。[PyTorch 2.1.2 AdamW 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/optim/adamw.py)、[Loshchilov 与 Hutter，ICLR 2019](https://arxiv.org/abs/1711.05101)

可由式子直接推导的反例是：忽略 ε，若完整梯度历史统一乘一个正常数 c，则 m 乘 c、v 乘 c²，自适应位移可保持相同；O 的改变通常不只是这种缩放，但“梯度小了，所以更新一定温和了”仍不成立。同一 LR/decay 下显式收缩系数相同；其相对自适应位移的强度是否改变需实测，不能一律称 O 增强了 decay。

最终 C checkpoint 的 `exp_avg/exp_avg_sq/step` 可刻画最终时刻预条件器，不能逆推出 512 个历史组或 rare 的家族贡献。若逐参数 step/原执行分支齐全，最后一次自适应位移可由最终矩计算；仍不能把最终矩与 M0 初值的 eval VJP 拼接，冒充 rare 出现时的真实 AdamW 更新。本轮不下载/运行大模型来补这个不可识别部分。

## 4. 已有日志新增的真实证据

本轮按冻结 train512 选择与 `RandomState(11)` 的四轮原始 permutation 重建 accum4 分组；C 日志 SHA 与 objective 协议的 `control_files_sha256` 精确一致。rare token `706d7015b80d4329b80d4a8c52139ab4` 是 train index 30，四次出现为：

| pass | update | 该组裁剪前 L2 | α 的标量重算近似 | 在本 pass 128 个组中的范数降序名次 |
|---|---:|---:|---:|---:|
| 1 | 67 | 238.9966 | 0.146446 | 1 |
| 2 | 229 | 331.0090 | 0.105737 | 1 |
| 3 | 275 | 390.5133 | 0.089626 | 1 |
| 4 | 395 | 515.8754 | 0.067846 | 1 |

C 总计 43/512 更新的记录范数超过 35。α 由已序列化范数用 float64 重算，未宣称复现 CUDA 浮点乘法的最后一位。四个组另有各自不同的三个训练 anchor；四次重复也不是四个独立 seed 或独立确认样本。

这把证据从“初始、dropout 关闭时 rare 的 raw 梯度 866.29”推进到“真实 train-mode、实际累积之后，包含 rare 的组在四轮都具有最大裁剪压力”。它是很强的定位线索，**不是** rare 单独导致该范数、梯度方向有害、AdamW 位移最大或开发 GMO 下降的证明。现有 VJP 还显示普通样本 sem＋geo 投影已占约 93%–96%，O 干预影响整个训练而非只修复一个异常样本。

## 5. F/O 后只做这一项便宜检验

**唯一建议：完整最终日志上的“固定累积组裁剪轨迹配对”，纯 CPU，不追加前向或训练。** 等 F/O 的 complete 与日志 SHA 绑定完成后，核同 512 更新、2048 exposure、相同四轮分组/LR，再对 C/F/O 的全部更新统一重算 α、每轮范数分位次及裁剪比例；单列预先固定的 update 67/229/275/395，并展示三个同组伙伴。保持全部样本与原性能指标，不根据 F/O 结果换 rare、改阈值或挑 pass。

这个检验直接区分两条目前都可能的解释：初值 VJP 异常是否在真实累积/clip 环节持续，F/O 的作用是否主要集中在这些已定位的组，或表现为所有组的普遍变化。它不能解开家族方向、二阶矩和后续轨迹的全部作用；因此不对四点作显著性检验，不据此裁定唯一因果，也不把较少裁剪当作新的晋级标准。

同更新号保证的是输入组、学习率与使用次数相同；C/F/O 从第一次更新后就可能有不同权重和 AdamW 状态。因此这是三条真实训练轨迹的配对描述，**不是在同一参数状态上切换 loss 的局部干预**。不能将其差值只归于当前四个 loss 的代数差。

- 若 O 显著减轻这些组的裁剪压力而 GMO 仍不改善，“缓解此压力足以修复性能”被反驳。
- 若 F 在保留强裁剪压力时改善 GMO，“必须先消除此压力才能改善”被反驳。
- 若 O 同时改善 GMO 和裁剪压力，仍可能是整体风险、梯度尺度、历史矩或决策边界的共同变化；只能形成一致的机制证据，不能独占归因。
- 若整体普遍减压而 rare 相对名次不变，更符合全目标尺度改变，不能写成特定 rare 归一化成功。

当前 F/O 只构成相对 C 的两个单独干预，没有 full＋去 scaling 的交互臂。无论胜负，都不能外推这种交互或把 O 当成自然 NLL 优化。本轮无需因机制解释而延长既定 F/O 训练或扩大验证预算。

对应实现为 [objective_clip_diagnostic.py](objective_clip_diagnostic.py)。它只接受三臂完成后的日志及严格 aggregate 收据；精确 checkpoint 内容/order 验证复用该已认证 CPU 证明，本脚本独立重算标量与分组，不重复读 pth。当前只在已完成 C 上验收了四组名次、43 次裁剪、完整日志拒绝门，尚未运行 F/O 联合结果。

## 来源与验证边界

已读：[损失梯度解读](loss_gradient_interpretation.md)、[原目标审计](objective_metric_audit.md)、[分割风险文献笔记](segmentation_risk_literature_note.md)。后两份中较早的“未保存分项/粗 GT 统计”等状态已被后续真实 probe 补充，不能当作当前仍然缺失。

- native `semkitti_loss.py` SHA：`e6a4b9accef1f47cf031f84e18fac925d2764b4105b8010486c8229e589abaa4`。
- 实际 VJP summary SHA：`e32c9f4e6dd2ff97909ada9aac7e90de891934a0205031dd1846e3ee2e188c5d`。
- C 完整训练日志 SHA：`0f7b650d3ab4f9fd0a1612943191e1411cb313586a2037e3de4b92b1a88de53f`；本轮逐行检查 512 更新并按原选择顺序重建上述四组。
- parent protocol SHA：`071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a`。
- 外部仅打开上列一手论文/官方实现；未采用文献结果替代本任务实测。公式是在明确条件下的本地推导，不是原 loss 的数值替换建议。
