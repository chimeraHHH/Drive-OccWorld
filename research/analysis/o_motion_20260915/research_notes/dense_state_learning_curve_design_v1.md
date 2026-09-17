# 固定状态读出的运动可学习性实验

2026-09-17。承接 `dense_task_state_fit_v2_结果与决策.md`，不是重新筛选数据或恢复 S3。

32 次更新已经证明位移会影响自有 decoder 的占据输出，但物理误差接近零运动参考。下一步先区分有限预算下的运动拟合和联合目标竞争，不能直接将短训结果解释为结构无效。

固定之前的 4 个训练样本 `[396,445,60,115]`，单 seed11；从认证的同一个 v2 shared codec 起步。三个臂均冻结 encoder/decoder，仅训练 dynamics、content_increment 和 velocity。每臂连续 512 次更新，在 0/32/128/512 做固定检查，无 best selection。

| 臂 | 优化目标 | 评分时读出 |
|---|---|---|
| P | 0.1 × 原物理 SmoothL1 | transport |
| J | 原单层 CE[1,5]+Lovasz + 0.1 × 物理 | transport |
| D | 与 J 相同 | identity |

同 AdamW、lr=3e-4、weight_decay=.01、clip=10，样本顺序相同。P 训练可跳过无关渲染，位移递推公式不变；所以参数和更新数相同，FLOPs 不相同。GT 只用于损失或事后评分，开发集不访问。

记录全网格原 fine-resolution GMO、原对象等权 EPE、静止/模糊/移动分组、位移幅度及 source 支持数量；记录置零/反向地址干预。固定时点对每个样本测 occupancy 与加权物理梯度的范数、点积和夹角，独立 velocity 头和全部动态参数分别报告；没有路径的梯度标 null。每步另记各模块实际参数更新范数和梯度，以识别零初始化的延迟及优化尺度差异。实际时间与名义 horizon 偏差继续记录。

P/J 同时改变 content 和 velocity 的学习，不能把它称为只干预运动头的因果实验。单个负 cosine 也不足以证明泛化上的任务冲突。若 P 能拟合而 J 不能，只能说明此协议下联合目标妨碍拟合，还要结合更新量和内容变化判断。若三者都能拟合，原 32-step 结论主要受预算限制；随后才值得进入较大训练集的正式协议。若都不能拟合，先审查目标、表征和优化，不以四样本失败断言信息不可辨识。

预算上限 2,400 s / 24 GiB，共 1,536 次更新；独立服务器 runner 托管，不修改 O/M0 或其他任务。协议：`dense_state_learning_curve_protocol_v1.json`。代码和协议在启动前推送 GitHub，完成后再追加结果和决策。

文献解释边界：[Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://arxiv.org/abs/2001.06782) 将梯度干扰作为多任务优化问题研究，但不能仅凭本实验的一个负夹角就断言同一机制成立，也不预先采用 PCGrad。[Adam 原论文（ICLR 2015）](https://arxiv.org/abs/1412.6980) 强调其对梯度重缩放的性质；忽略 epsilon 时单任务统一缩放可被一、二阶矩的比例抵消，因此不能从 P 的损失系数 0.1 推断其参数步长必然比 J 小。这里直接记录实际更新范数，判断仍依赖本实验测量。
