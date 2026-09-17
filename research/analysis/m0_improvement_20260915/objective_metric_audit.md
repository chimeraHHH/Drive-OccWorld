# 原生训练目标与 GMO 指标审计

审计日期：2026-09-15。只读源码与已完成诊断；没有改变模型、训练、GT、阈值或评测。结论适用于固定训练缓存前 16 个场景各首个 anchor 的事后诊断，不能外推整个训练集，也不是新确认集。

**目前最强证据是：双记忆臂的平均复合损失下降主要来自一个极端损失样本的 scaling 项恢复；其余 15 样本的平均损失略升。三个训练臂在完整分辨率上的召回率提高、误报更多，最终 GMO IoU 下降。** 不能据此把下降简单归为过拟合，也不能把总 loss 下降视为整体预测质量改善。监督粒度与指标聚合方式确有明确差异，但它们各自造成多大性能损失尚未被干预实验识别。

## 1. 真实数值：哪一部分在下降

来源为 [诊断 summary.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/memory_train_diagnostic_v1/summary.json) 和同目录四份 `*_records.jsonl`。已核对 summary 和四份记录均匹配 `complete.json` 的 SHA；独立重算逐样本损失均值、损失族之和及混淆矩阵。四模型每个 token 的五时域 GT 行计数相同，12 项 loss 全部有限。

下表每个损失族均为三个 decoder 层之和，再对 16 个 anchor 算术平均；它不是自然 NLL。

| 模型 | 总 loss | 加权 CE | sem_scal | geo_scal | Lovász | Future macro GMO % |
|---|---:|---:|---:|---:|---:|---:|
| M0_fp32 | 17.849831 | 0.127612 | 7.049702 | 9.599995 | 1.072523 | 20.8198 |
| native1 | 15.163167 | 0.115214 | 5.800293 | 8.179272 | 1.068389 | 20.2624 |
| persistent2 | 15.616486 | 0.122376 | 5.979365 | 8.422164 | 1.092580 | 18.5760 |
| rolling2 | 11.088764 | 0.125713 | 4.001038 | 5.864574 | 1.097438 | 18.4390 |

native1 总 loss 减少 2.686664，sem+geo 减少 2.670132，占净降幅 99.38%。persistent2 和 rolling2 的 sem+geo 分别减少 2.248167、6.784084，超过总净降幅；两者 Lovász 反而增加 0.020057、0.024915。三个 decoder 层的总 loss 均降低，不能解释成“只有前两层辅助损失下降，最后层没有改善”。不过这些都是**标量损失贡献，不是梯度范数或更新方向贡献**；现有记录没有保存分项梯度。

### 单个样本主导平均数

固定序号 16 的 token `706d7015b80d4329b80d4a8c52139ab4`、scene `dd61533869aa48f2aaa7e8a6418bdbe6`，M0 loss 为 150.383162，其余 15 样本范围为 5.943555–12.219310。这个样本不是事后剔除对象，也没有证据证明其标注错误。

| 模型 | 该样本 loss | 该样本贡献的总净 loss 降幅 | 其余 15 样本平均 loss | 全部 16 样本中 loss 下降数 |
|---|---:|---:|---:|---:|
| M0_fp32 | 150.383162 | — | 9.014276 | — |
| native1 | 111.965383 | 89.37% | 8.709686 | 13/16 |
| persistent2 | 113.483625 | 103.26% | 9.092010 | 8/16 |
| rolling2 | 39.304209 | 102.68% | 9.207734 | 8/16 |

百分比的分母为所有 16 个样本 loss 减少量之和；超过 100% 表示其余样本合计恶化。去除单例的数字仅用于描述集中程度，**不替换原正式均值，不构造新的筛选样本集**。persistent2/rolling2 的逐样本 loss 减少量中位数分别为 −0.093534/−0.067462，即典型样本没有同样的大幅改善。

该样本原生完整 GT 在当前、0.5/1/1.5/2 秒的 GMO 体素数为 `[192,109,127,138,112]`，每帧总计 10,485,760 体素且没有 255：它是极稀疏前景样本，并非全空标签。M0、native1、persistent2 的五时域 TP 均为 0；rolling2 只在 2 秒得到 51 个 TP，同时该时域 FP 从 M0 的 138 增至 2,625。它的 sem+geo 总值从 148.877777 降至 rolling2 的 37.811676，CE 则从 0.005336 增至 0.007333，Lovász 仅从 1.500049 降至 1.485201。大幅降低负对数软统计惩罚，完全可以伴随很小的硬标签改进；目前没有保存其降采样后的 GT 数量、soft intersection 或各分母，不能把具体原因进一步认定为某一种下采样边界或数值溢出。

### 完整分辨率的误差变化

以下 precision/recall 来自四个未来时域合并混淆矩阵；主 GMO 仍是逐时域汇总后再平均，二者定义分别保留。

| 模型 | Precision % | Recall % | Future FP | Future FN |
|---|---:|---:|---:|---:|
| M0_fp32 | 24.1949 | 60.0350 | 970,157 | 206,131 |
| native1 | 22.8591 | 64.3066 | 1,119,294 | 184,099 |
| persistent2 | 20.6565 | 64.9332 | 1,286,429 | 180,867 |
| rolling2 | 20.5190 | 64.5532 | 1,289,699 | 182,827 |

在这批相同标签上，“增加 TP、减少 FN，但 FP 增加更多而 IoU 下降”是混淆矩阵直接支持的结果。它不能独自区分阈值偏移、排序变差、空间边界错位或特征质量下降；也不能把 GMO 当成实际运动目标。已完成 dev200 同样下降，但未记录 dev 原生 loss，不能补造训练/开发 loss gap。

## 2. 原生训练链的精确定义

**入口和轴：** [compute_occ_loss](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:596) 将预测转为 `[3层,5时域×batch,2类,200,200,16]`，包含当前和四个未来，平面轴转换与原生 evaluator 相同。GT 先从七帧中切去两帧历史，再按同样 frame-major 顺序展平。微批量 1 时，损失每次看到的是同一 anchor 的五个时域，不是五个独立 scene。

**监督粒度：** [loss_voxel/loss_occ](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:195) 将每层原始 logits 三线性插值到 `256×256×20`，`align_corners=False`。[_downsample_occ_target](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:15) 对原 `512×512×40` GT 的每个 2×2×2 block 作特殊 mode：全空 block 保留 0；非全空 block 中的每个 0 被赋予互不相同的负编号，mode 后负编号转成 255。

这不是普通多数投票、max pooling 或软标签平均。对于只含 0/1 的 block，源码可严格推出：

| 原 8 个体素 | 粗 GT | 与完整 GT 的关系 |
|---|---:|---|
| 8 个 0 | 0 | 保留背景 |
| 1 个 1、7 个 0 | 255 | 整块不参与四项损失，包括原本有效的单个 GMO |
| 2 个 1、6 个 0 | 1 | 25% 前景的块成为硬前景标签 |
| 8 个 1 | 1 | 保留前景 |

若同时有 255，需比较原 class1 和 255 的计数：重复次数相同且至少为 2 时，小类别 1 获胜；不能将其简化为“有未知就全忽略”。这些是原函数语义的离散反例，不是对真实 GT block 频率的测量。粗标签的前景保留/忽略策略与完整体素风险不同；它为边界误报提供了可检验解释，但不证明这就是当前性能下降的主要原因。

**12 个损失与权重：** [S0 loss 配置](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py:123) 中四个损失系数均为 1；[head 初始化](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:61) 固定类别权重 `[1,5]`。每一 decoder 层计算一次加权 CE、semantic scaling、geometric scaling 和 Lovász，再由原训练器 `sum(losses.values())` 相加，三层没有额外缩放。配置 `loss_weight=[[1],[1],[0]]` 和 `per_frame_loss_weight=(1,)` 在这条 `loss_occ→loss_voxel` 路径没有被读取，不能解释成“第三层不训练”或已应用指定时域权重。

CE 是按目标类权重归一化的 cross entropy，忽略粗标签 255；它不是自然分布下的等权 proper NLL。其他三项先汇合五个时域的有效体素，再形成软统计或排序，因此也不是“四个未来时域分别算 IoU 后等权平均”。梯度累积四次只平均四个 anchor 各自的非线性损失，不等于把四个 scene 合成一份全局 IoU 损失。

**scaling 的目标：** [geo_scal_loss/sem_scal_loss](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/losses/semkitti_loss.py:70) 先 softmax，再对 soft precision、recall、specificity 分别计算目标为 1 的 BCE。二分类中 geo 的 nonempty 就是 GMO；sem 对两个类平均，且和 geo 存在重复惩罚。若两个类都存在、概率分母非零，忽略 geo 的 `1e−5` 稳定项，可写成：

`geo + sem ≈ −1.5 log(P1) −0.5 log(P0) −2 log(R1) −2 log(R0)`。

这里 P/R 是粗 GT 上的软 precision/recall；不是最终 argmax 的统计。这解释了为何对接近零的软交集改善可以大幅减小 loss，同时硬 IoU 仍很低。实际梯度还受 softmax、分母、权重共享等影响；当前标量分解不能证明训练更新被这个样本或某一项支配。

**Lovász 的范围：** [实现](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/losses/lovasz_softmax.py:160) 使用 softmax 概率、`classes='present'`、`per_image=False`，对五时域展平后的有效体素、存在的二类求 surrogate 均值。它优化的是这份粗标签/小批次上的代理目标，不能因名字是 IoU loss 就认定与完整 GT 的 future macro GMO 完全一致。

## 3. 评测链与已排除/未排除项

[evaluate_occ_records](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:712) 只取最后 decoder 层；**直接从原生 logits** 三线性插值到完整 `512×512×40`，不经训练时的 `256×256×20` 中转。然后在 class 轴 argmax，GT 有效条件为 `0≤GT<2`，即排除原 GT 的 255；没有可见性 mask、运动 mask 或训练粗网格新产生的 ignore mask。二分类中该 argmax 与在插值后的 logits 上做 softmax、再采用 0.5 决策相同。

[固定统计定义](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/aggregate_memory.py:69) 是每个未来时域跨样本池化混淆矩阵，再计算 GMO IoU，最后对四个未来时域算术平均。当前时域、前两层辅助头、粗标签 class1/255 的重新分配均不进入这个主指标。训练/评测的平面轴转换在本链一致；另行坐标契约问题不在本审计混合归因。

**已有证据排除**“所有样本都均匀受益”“总 loss 下降等价于 natural NLL 或 full-resolution IoU 改善”“仅末层之外的 loss 掩盖退化”。**尚未排除**局部过拟合、优化步数/学习率影响、几何偏差、概率操作点变化、排序变差、粗监督边界以及部分异常梯度。Train16 都属于已用于拟合的样本，诊断时 dropout 关闭，不能与训练日志中 dropout 开启的 loss 数值直接配对。

## 4. 后续最低成本且可反驳的次序

1. **本次 reference 单槽 A 保持原 loss。** 与 native1 C 比较时只改变坐标契约，避免把几何修正和监督目标修正叠在一次干预中。本审计不触发新训练、提前停止或改写协议。
2. 若开展下一份独立只评测诊断，先在相同固定 16 个训练 anchor 记录每时域原/粗 GT 的 0/1/255 数量、`n_GMO∈{0,1,2…8}` block 频率和粗标签覆盖的原生类计数，并分别计算最终层“粗 GT/粗 logits”IoU 与原完整 GT IoU。直接检查发生了多少忽略和边界标签改变；不从性能重新挑样，不改正式 mask。
3. 在固定训练 anchor 上拆出每层/每时域 scaling 的 soft intersection、prediction mass、GT mass、precision/recall/spec；极端样本与其余样本都保留。必要时仅做无 optimizer 的分项梯度范数/夹角诊断，才能从“标量集中”推进到“更新方向冲突”。没有这些量，不能宣称大 loss 使所有训练被异常样本拖偏。
4. 只有另行冻结一个目标实验后，才比较**同一模型结构/初始化/seed11/样本/更新数**的原粗监督与原生分辨率监督。最直接、变量最少的版本仍保留原 loss 家族和层/时域范围，仅把监督节点移到 evaluator 的完整 logits 插值及原有效 GT；它检验监督粒度，不能同时更换层权重、去掉 scaling 或新增空间 refiner。全体素 Lovász 排序的内存/时间成本必须先 profile；不能因不够资源便把采样 Lovász 冒称同一精确目标。
5. 若资源只允许采样 CE，均匀原生有效位置或有正确重要性权重的分层采样可以估计相应全体素 CE，但**不能直接得到无偏的全局 ratio/排序 Lovász**。应另立受限目标对照，并明确无权重 CE 与 `[1,5]` CE 的区别。损失改变属于已知工具，需要由真实 dev200、随后独立完整协议评价证明收益，不作为方法新颖性本身。

已有 Lovász 一手论文明确区分 image/batch IoU 与 dataset IoU，指出平均比值与总体比值并不等价，并以 present-class heuristic 处理批次问题；这直接支持本审计对聚合层次的限制，而不保证换 loss 必然提高本模型。[Berman 等，CVPR 2018，§2–3.1](https://arxiv.org/html/1705.08790)。PointRend 在连续位置插值提取特征并作点级预测，说明高分辨率监督/边界计算可以作资源受限设计；其训练采用有偏不确定性采样，因此不能拿来证明无偏全体素 NLL，更不能直接证明 3D GMO 收益。[Kirillov 等，CVPR 2020，§3–4](https://arxiv.org/html/1912.08193)。两篇均已打开作者一手全文并读取相关方法段；这里不引入其新网络。

## 5. 可回溯来源

- 诊断 summary SHA256：`e589ff427824d90b40a286ec3b9ba77301c42b9be31d0b7cb4edea49791642d2`。
- M0_fp32 / native1 / persistent2 / rolling2 逐样本记录 SHA256 依次为：`ac08a5ccc3e9e1b0a8909f47de591ab6cf29c3e3a31804beba7c648823776113`、`acaf8e507854b147ccbc211088a6f061a54d3d737a99381f9628abbd095a034d`、`bec67ad9d80e85dc477e05926596f65e65419b70cc4ca1687791990d85fdb6cc`、`e9ee6ddd8768e09a6e798d3f920cdace7cd52c6ac7f73ad85b0709b361ea4a3e`。
- 固定协议 SHA256：`071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a`；原生 detector `67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5`；world_head_v1 `4738c03a28e353efd4dec68adf6c9407aa73b9273bd13c2f67021795d6240a7a`；semkitti_loss `e6a4b9accef1f47cf031f84e18fac925d2764b4105b8010486c8229e589abaa4`；Lovász 实现 `046a43d610d34cdd6d4b5f791aa5df0723e9cae3cbff4e34de086574b62cf8be`。
- 实际损失诊断在 `eval/no_grad` 下完成，所有模型参数前后不变；本报告仅重新读取与计算，不新增 checkpoint 或模型前向。
