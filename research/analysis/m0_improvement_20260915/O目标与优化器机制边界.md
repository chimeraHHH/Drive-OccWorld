# O 目标与优化器机制边界

2026-09-15。**目前能排除“在已检查点，O 只是原目标的一个正标量缩放”，但不能把 O 的开发收益拆成目标组成、时变裁剪和 AdamW 历史各自的因果份额。** O−C 的开发差为 +0.9254 pp；它识别的是固定预算配方的整体作用。本篇只补充已有梯度/clip 报告缺少的数学边界和可由真实记录重算的量，不读取部分 full 成绩、不改冻结源、不运行 GPU 或训练。

## 已有证据及本次新增重算

原生循环把四个样本各自的 loss 除以 4 后反传，累积完成才对整个 future head 做 global L2 clip35，再执行 AdamW；日志 `grad_norm` 是裁剪前范数。C/F/O 都是新建零初始 moments 的 AdamW，相同 512 updates、精确样本次序和 LR；实际最终 checkpoint 审计记录 β=(0.9,0.999)、ε=1e−8、weight_decay=0.01、amsgrad=False。O 仍前向计算原 12 项，但只有 CE/Lovász 六项进入优化；没有通过少算前向改变 dropout 调用顺序。[训练循环与新建 optimizer](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:220)、[O 选择原损失张量](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_adapters.py:65)

**同状态方向证据。** 原无 optimizer 预检已经保存四族 VJP 的范数、内积和它们对总梯度的内积。令 `u=g_CE+g_Lovasz`，直接用 Gram 数据计算 `||u||²=||g_CE||²+||g_Lovasz||²+2〈g_CE,g_Lovasz〉`，以及 `〈u,g_C〉`，得到：

| 原 M0 / coarse / eval 样本 | 推导的 `||u||` | `||g_C||/||u||` | `cos(u,g_C)` | 最佳正标量拟合的相对残差¹ |
|---|---:|---:|---:|---:|
| 普通 1 | 2.10278 | 13.7445 | 0.97756 | 21.07% |
| 普通 2 | 2.83196 | 24.6644 | 0.99001 | 14.10% |
| 事后 rare | 0.03098 | 27963.1 | 0.96865 | 24.84% |

¹ `min(c>0)||g_C−cu||/||g_C||=sqrt(1−cos²)`。这些量是**已测家族向量统计的代数重算，不是新执行 O backward**；原重复 VJP 的浮点加和审计也并非逐位相等。普通两点和事后 rare 不是总体代表样本；没有 accum4、train-mode dropout 或 optimizer。可以排除这三点的纯同向缩放，不能据此声称三族目标冲突，或把残差当 AdamW 位移。范数比例差别也提示样本间权重关系会改变，不能只用一个总体 loss 系数复现它。

**优化历史证据。** 完整真实日志已经显示 C/F 的大范数组，而 O 从未接近阈值；第一步 C=22.9842、O=1.15913，二者当时均不触发 clip。至少这项原始范数差异在 clip 发生前就存在，但它不证明第一步 AdamW 位移按此比例不同。

| 臂 | 日志 `r_t>35` 次数 / 512 | 裁剪前范数中位数 | 最大范数 | 推导的最终二阶矩对角和 `Σ_j v̂_512,j`² |
|---|---:|---:|---:|---:|
| C | 43 | 15.0536 | 515.8754 | 355.7142 |
| F | 41 | 14.4760 | 589.3169 | 346.0052 |
| O | 0 | 0.9060 | 7.37255 | 1.57824 |

² 新增量仅是标量近似重建。设 `s_t=min(1,35/(r_t+1e−6))`、`h_t=s_t g_t`，则零初始 moments 下

`Σ_j v̂_T,j = (1−β₂)/(1−β₂^T) · Σ_{t=1}^T β₂^(T−t) (s_t r_t)²`。

最终实际 payload 审计的 per-parameter step **最小值和最大值均为 512**；PyTorch 只有 grad 非 None 的参数才增加 step，故所有已进入 state 的参数参加了每步更新，始终无 grad 的参数可作零贡献。旧审计没有记录 state 条目数，本篇不追加“已证实恰有 130 份 state”的说法。这里用序列化 FP32 norm 在 CPU float64 中代入公式，**不是读取或逐位重建 GPU moments**。按同一近似，C 的 43 个大范数组贡献最终对角和约 27.10%；含固定 rare 的四个 accum4 组贡献约 2.64%，不是单个 rare 样本的贡献，更不是 2.64% 的预测作用。没有各步梯度方向，不能重建一阶矩、逐坐标分母或实际位移。

## AdamW：常数缩放何时抵消

对实际喂入 optimizer 的裁剪后梯度 `h_t`，在实数算术中写成

```
m_t = β₁ m_{t−1} + (1−β₁) h_t
v_t = β₂ v_{t−1} + (1−β₂) h_t²
m̂_t = m_t/(1−β₁^t),  v̂_t = v_t/(1−β₂^t)
θ_t = (1−η_t λ) θ_{t−1} − η_t m̂_t/(sqrt(v̂_t)+ε)
```

`v` 是未中心化二阶矩，不是减去均值后的方差。上式对应实际非 AMSGrad 配方；原论文 Adam 的缩放论述明确在 ε=0 的推导语境下，AdamW 把 decay 与梯度 moments 分开。[Adam Algorithm 1 / §2.1](https://arxiv.org/pdf/1412.6980)、[AdamW Algorithm 2 / Proposition 2](https://arxiv.org/pdf/1711.05101)、[PyTorch v2.1.2 实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/optim/adamw.py)

若**全部历史步**都有 `h'_t=c h_t`，c>0 是常数，同 β、step、零初始 moments，则 `m̂'=c m̂`、`v̂'=c²v̂`，自适应项变成

`m̂ / (sqrt(v̂)+ε/c)`。

因此固定 ε>0 时不普遍精确抵消；ε=0 且分母定义良好，或同时令 ε'=cε，才有这一理想等价。若 `sqrt(v̂_j)` 远大于 ε 和 ε/c，则可近似不变，但总 L2 范数大并不证明每个坐标都满足该条件。若中途改尺度而保留未相应缩放的旧 moments，关系也不成立。相同参数路径还需要相同初值、随机序列、LR/decay 等；有限精度不保证逐位复现。

**decoupled decay 不会仅因 loss 乘常数就自动增强。** 在上面成立的同参数路径下，其 `(1−ηλ)θ` 完全相同；不能套用 SGD 或耦合 L2 的直觉。本项目三臂实际 `Ση_t=.002816`，实数公式中的齐次 decay 乘积 `∏(1−.01η_t)=.9999718404`，即约 0.002816% 收缩，各臂相同。这不是实际权重变化测量：FP32 乘法舍入、自适应项及后来不同的 θ 路径未包含。O 的 decay 相对自适应位移可能不同，但现有日志没有测出其方向或大小。

## 为什么时变 clip 不能逐步约掉

实际 global clip 对所有坐标用同一个 `s_t`，但该系数随 batch、参数及历史变化。原始梯度乘 c 后，相对原裁剪后梯度的系数是

`d_t = c·min(1,35/(c r_t+δ)) / min(1,35/(r_t+δ)),  δ=1e−6`。

两条路径始终不 clip 时 d_t=c；理想 δ=0 且两者始终 clip 时 d_t=1；跨阈值、混合状态一般使 d_t 随时间改变。实际 δ>0 时，两者都 clip 也仅近似相同。这个 δ 与 Adam 的 ε 是不同机制。[PyTorch v2.1.2 clip 源码，62–75 行](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/torch/nn/utils/clip_grad.py)

对任意时变标量 a_t，第一矩积累 `Σβ₁^(t−k)a_k h_k`，第二矩积累 `Σβ₂^(t−k)a_k²h_k²`，通常不能提出同一公因子；clip 改变历史，后续即使不再 clip 也可能继续影响更新。当前 β₁/β₂ 的指数权重半衰期约 6.58/692.80 updates，后者超过整个短训长度，说明长记忆机制可能相关，**不是它造成 O 收益的实证**。clip35 也不是 AdamW 参数位移的上界。

所以既不能说“范数减小约 17 倍，更新就减小 17 倍”，也不能说“AdamW 尺度不变，所以裁剪和尺度完全无关”。F 的 clip 次数和二阶矩对角和接近 C，但开发指标仍比 C 高约 0.4760 pp；这也提醒我们不能只据全局尺度给三条路径的成绩排序。F 同时改了监督位置，三条训练路径早已不同，仍不是 optimizer 因果隔离。

## 至多一个下一轮匹配对照：保留 C 方向，匹配 O 的逐步范数

仅提出一个新单 seed11 训练臂 N，暂不实现或授权。固定原 M0、coarse GT 原 12 项、512 anchors×4 passes、accum4、参数范围、原 dropout/RNG 顺序、512-update LR、AdamW β/ε/decay 和 clip35；在每个 accum4 结束后，取 N 当前参数处的原 C 梯度 `g_t^C(θ_t^N)`，设置

`g̃_t^N = r_t^O · g_t^C(θ_t^N) / ||g_t^C(θ_t^N)||`，

其中 `r_t^O` 是**已经完成的 O 同 update 训练日志**里的 512 个固定范数，仅使用训练轨迹，不拟合 dev 分数。零/非有限分母直接终止，不增添临时补偿。随后走原 clip35 和 AdamW。O 的最大目标范数 7.37255 低于 35，故在数值容差内可同时匹配输入 Adam 的全局范数、无裁剪状态，以及零初始情况下的整个二阶矩对角和历史。匹配应在真实预检中逐步验证，未通过则不把它当已匹配实验。

这保留了 N 当前点的 C 梯度方向，**并非在相同 θ 重建 C/O 梯度，也不是仍优化原均匀经验目标的完全相同算法**：逐步正标量会改变 batch 权重和优化动态。它不匹配逐坐标 moments、ε 敏感坐标、实际参数位移或已经分岔的状态路径。它是 O 结果之后设计的训练信号控制，不是独立预注册的新算法。

固定最终 dev200，比较 N−C、N−O；不挑 checkpoint、阈值、seed 或自动追加 full。如果 N 在预先冻结的实质等效界内接近 O，只能说明该全局尺度历史配合 C 方向在这轮预算下足以取得相近效用；不能证明 sem/geo 总体无用或某机制唯一成立。若充分匹配后 N 仍明显落后 O，可否定“仅该全局尺度/clip/对角和轨迹足以解释”的窄假说，不能排除逐坐标预条件与方向的耦合。没有显著差异不等于等效；等效界和比较规则需在这唯一新 run 前冻结。

若完整结果与后续诊断支持执行这一控制实验，先完成真实预检再冻结预算；日志应补 `target/actual global norm`、clip 系数、二阶矩对角和及实际 adaptive/decay 位移分量，避免再拿原 loss 或梯度范数替代更新。不会为本篇另行启动反事实 GPU 试验。

## 来源边界

- 本次先复用 [原梯度解读](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/loss_gradient_interpretation.md)、[完整 clip 诊断](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/objective_clip_diagnostic_v1/report.md)；这里新增的是 Gram 重组合、二阶矩对角和及 decay 的受限推导，不覆盖它们的原记录。
- 实际 VJP summary SHA `e32c9f4e6dd2ff97909ada9aac7e90de891934a0205031dd1846e3ee2e188c5d`；clip summary SHA `a87e6b293d001802d1b148a5e3bb29297730aff7e92adb36aed2122a6ff21f8f`；完整五模型 dev summary SHA `46c37b5b074e73896e357e0bc9744a538abb01b288521769a577f459be54ad6b`。均已核 complete 链；C/F/O 的实际完整日志、manifest、final receipts 与已认证 summary 的哈希逐项一致。
- C/F/O training.jsonl SHA 依次为 `0f7b650d3ab4f9fd0a1612943191e1411cb313586a2037e3de4b92b1a88de53f`、`34d2091b405ca4fd2823144b55c535359b4eb0506c309c79eec64a382266890b`、`c5895553f25f8f26fbcedf2479b4302bee314be68e21a42717fc6e63e93d9b70`。
- optimizer 参数/step 范围复用已认证的真实 CPU final-payload 审计，本篇没有再次读取服务器 `.pth`，也没有每步向量档案；不要把代数重建写成新增 optimizer 测量。公开来源仅使用原论文和 PyTorch 官方 v2.1.2 代码；没有使用 skills 或 mock 实验。
