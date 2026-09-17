# Sanyal 2018：非可分解目标与当前 O 路线

结论：可作为“任务指标驱动的深网微调”的直接先例；**不能把多时域比率优化、删除 sem/geo 或当前训练器的收敛性归给该文证明**。当前最有用的是辨清目标及总体统计量，不是复制旧代码。

文献：Amartya Sanyal、Pawan Kumar、Purushottam Kar、Sanjay Chawla、Fabrizio Sebastiani，*Optimizing non-decomposable measures with deep networks*，Machine Learning 107:1597–1620 (2018)，[期刊 DOI](https://doi.org/10.1007/s10994-018-5736-y)。实际细读公开 [arXiv v1 正文与附录](https://arxiv.org/pdf/1802.00086)；以下页码均为该 **17 页 PDF 的 1-based 页码**，不混用期刊页码。出版社页面为订阅预览，未声称读到其完整最终版。

## 原文定位摘要

保留简短英文技术摘要，避免长篇转述单一来源。

**Setting** (§3, pp.3–4): i.i.d. binary examples; sigmoid rewards replace hard TPR/TNR.

**DUPLE** (§4.1, Algorithms 1, pp.5–7; Eqs.2–3): concave Fenchel representation; alternate weighted-reward network ascent and dual updates. Theorem 1 assumes smooth rewards and a smooth concave link; constant step below (2/L) yields augmented-gradient stabilization. Appendix A (p.14) sketches batch analysis, mentioning minibatch error (O(b^{-1/2})).

**DENIM** (§4.2, Algorithm 2, pp.7–8): nested inner/outer dual variables; its separate convergence proof is omitted.

**DAME** (§4.3, Definition 1, Algorithm 3, pp.7–9): pretrain, freeze lower features, alternate estimating v=A/B and upper-network ascent on A−vB. Jaccard belongs to its pseudolinear family (§3.1). Theorem 2/Appendix B (pp.8–9,14–16) use batch fine-tuning, smoothness, A≤M, B≥m>0, κ=1+M/m, and constant η<2/(Lκ), claiming first-order stability within O(ε⁻²) inner iterations. The pseudolinear experiments concern F1 (§5.3,p.11; Fig.7,p.15). [原文](https://arxiv.org/html/1802.00086)

## 对本项目的独立推导：单比率与多比率

以下公式是按本项目二分类 GMO confusion 定义自行推导，不是该文给出的多时域定理。固定时域 h，在不含 ignore 的目标总体上令 p_h=P(y=1)，P_h=TPR，N_h=TNR。则

\[
J_h=\frac{p_hP_h}{p_h+(1-p_h)(1-N_h)}=\frac{A_h}{B_h}.
\]

可令 (A_h=p_hP_h)、(B_h=1-(1-p_h)N_h)。这确实是一个线性分式；但它仍是相对于正确总体/权重的指标。若用 (q=\mathrm{softmax}(z)_1) 替换硬预测，则

\[
A_h=E[yq],\qquad B_h=E[y+(1-y)q].
\]

所得为 soft IoU；原评价用 argmax 的 hard IoU，两者一般不相等。平滑分母、停止梯度、温度或采样权重都必须单独定义，不能称作原 hard IoU 的无偏优化。若批次全无前景，不能照搬“前景比例固定且正”的条件；加入 epsilon 也改变目标与梯度。

当前主评价量是各未来时域**先汇总全部 anchors 的 confusion，再分别求 IoU，最后平均**：

\[
J_{\mathrm{macro}}=\tfrac14\sum_{h=1}^4 A_h/B_h,
\qquad
\nabla J_{\mathrm{macro}}=\tfrac14\sum_h B_h^{-1}\bigl(\nabla A_h-J_h\nabla B_h\bigr).
\]

这说明三项常见误接：

1. 将时域混为一个 (A/B)，得到的是 pooled 目标，不是 macro。
2. 只累加四个 (A_h-v_hB_h)，遗漏 (1/B_h)，一般不等于 macro 梯度。
3. 对每个 anchor/batch 直接算比率后平均，一般不等于整个训练总体先累计再算比率；(E[\hat A/\hat B]\ne E[\hat A]/E[\hat B])。梯度累积也不会自动消除这一差别。

这些是代数边界，不构成新颖性。四比率之和并不能仅凭“每项都是 pseudolinear”就继承单比率定理；若再做最坏时域或 CVaR 风险，目标集合、风险权重和证明都需另行建立。DENIM 的“嵌套凹”名称也不能自动包住 IoU 的这类组合。

## O 实际优化了什么

按本地冻结源，O 只保留三层各自的 CE[1,5] 和 Lovász，共六项；原粗 GT、ignore=255、五时域一起展平仍保留。Lovász 默认 `per_image=False, classes='present'`，在每层的展平集合上按出现的类别平均；不是明确的未来四时域 GMO 单类 macro IoU。sem/geo 仍前向计算，但不进入优化返回字典。见 [adapter](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_adapters.py:32)、[native loss](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:195)、[时域展平](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:596)。

因此 O 和 DAME 的交集是“保留预训练表征再调整上层目标”；O 并没有动态 valuation 变量。O 若成功，只能支持当前损失组合在当前训练配置下更有效，不能倒推出“比率目标均不稳定”“sem/geo 恒有害”或“直接 IoU 优化已实现”。当前 AdamW、裁剪、非平滑 Lovász、相关体素/时序样本，也不满足仅靠引用上述定理即可核定的条件；低 raw-gradient 更不等于小 AdamW 更新。

## 作者代码核实与实现建议

出版社 Note 3 指向 [作者仓库 DeepPerf](https://github.com/purushottamkar/DeepPerf)；README 又指向第一作者同名仓库，两者本次解析到 commit `c012be18ee0db9d6f7cd7ad24d2a7161be7754a0`。DAME 实现为 Python 2 风格的 Theano/Blocks/Fuel，非即用 PyTorch 实现。

固定版本 [ANNAMP.py](https://github.com/purushottamkar/DeepPerf/blob/c012be18ee0db9d6f7cd7ad24d2a7161be7754a0/deep_non_decomp_src/DAMP/ANNAMP.py) 存在必须披露的静态差异：L53–56 实际预训平方误差；L149–161 使用阈值门乘概率的奖励与 Adam；L181–194 初始化 `value` 后循环只更新 `nu`，没有刷新传入下一轮的 `value`；L209–212、231–234 的严格 `<` 分支会漏写最后一批。这里的奖励既非纯 hard confusion，也非上文无门控的 soft confusion。这些事实不否定论文全部实验，但足以拒绝未经修复审计就把当前仓库当算法标准答案。此文件 SHA256：`f8dc9cc832533faa4290f703b60537ff9c48250f53c8a3c0f96bde7d6ce61a0c`。

本次只发现 `seq2seq-attn/LICENSE`，未发现根目录或 `deep_non_decomp_src` 的独立许可证/锁定依赖文件；不能用子目录许可推定整仓许可。未安装或执行作者代码。

可执行的下一步仅是设计建议，不是已启动实验：先固定完整训练总体、每时域原始有效体素与权重契约；用冻结模型得到训练集的 (A_h,B_h)，在一组固定样本上验证上式梯度与直接 quotient autograd 的一致性，再评估分批/累计近似误差。若使用停止梯度的总体统计量，应记录其更新时间、偏差和陈旧程度；不能把一次精确梯度等式提升为整个优化过程的保证。稀疏场景仍保留，t0 必须独立报告。不得用 partial full 结果选温度、权重、阈值或停止点。

强对照至少需要 O 原目标、相同时域集合的“分时域后平均”目标，以及明确匹配总体统计的 macro-ratio 目标；同初始化、训练数据/预算/seed11。先区分收益来自时域重新加权、去除项、GT 粒度还是比率统计，避免一次叠加全部改变。只按任务指标重加权、缓存冻结特征、分时域比率或使用 sigmoid 奖励，都不能包装成 ICLR 新贡献。真正的贡献需有新机制或新的、适用且经过验证的风险/估计理论，并得到冻结完整评价支持。
