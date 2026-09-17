# 共享刚体状态：真实原生预检

2026-09-16。H200任务 `shared_rigid_native_preflight_v1` 已正常退出，return code 0；runner 72302、child 72303 均确认不存在。两组各完成4次AdamW更新、16个固定训练样本；无保存权重、无开发集评分。**这证明接入和短步优化可运行，尚不证明超过O或运动更准。**

## 已完成的实际检查

- 固定两个训练样本上，初始五时域、三个decoder的全部logits与原O逐字节一致；完整640000点物理场与原预测框恒速CV一致。原完整GT混淆也一致。
- 两组第一轮accum4的初始logits/物理场一致，原六项CE/Lovász按既定数值口径通过；16个样本前后RNG配对一致。原O非future-head状态保持不变。
- 两组各130个future-head参数张量与11个新增模块参数张量的AdamW状态均到step4；记录中的loss、梯度、参数和一二阶矩有限。正式训练必须从原O和新seed11初始化，不能继续这些已丢弃的临时权重。
- 在第三次更新之后的固定样本ordinal394上，两个loss均到达共享对象encoder和pose head。物理loss不连接原native future head，符合此次结构；Fix的占据梯度仍可经值分支到达pose，不能解释成完全断开运动。

| 第三次更新后，同一固定样本的原始梯度范数 | Cpl | Fix |
|---|---:|---:|
| occupancy → object encoder | 9.629e-6 | 9.625e-6 |
| physical → object encoder | 6.821e-4 | 6.821e-4 |
| occupancy → pose head | 5.366e-6 | 6.647e-7 |
| physical → pose head | 0.126155 | 0.126155 |

这是一个样本、一个短训时点的未裁剪VJP，不是全训练集结论或AdamW更新比例。两组此时的参数已分别更新，梯度范数差不能单独归因于地址边。占据到位姿的作用目前很弱，非零不等于任务已经形成有效协同。

在各组自己的固定终点权重下切换Cpl/Fix地址，值向量和完整物理场保持字节一致，当前帧logits也不变，未来logits有变化：最大绝对差两组均为 **2.6703e-5**，平均绝对差约 **1.605e-7**。这个结果验证了真实forward连接；没有测IoU/EPE，不能把微小数值响应当作性能收益。

## 运行成本与后续动作

全流程122.16秒，其中16份完整当前几何78.63秒，native初始化9.11秒，初态对齐3.35秒，四次配对accum4更新24.32秒，地址干预1.80秒。峰值allocated **19.263 GiB**、reserved约21.166 GiB，低于冻结的32 GiB上限。预检的配对更新约5.71–6.49秒，包含审计开销；这些是实测短窗，不能直接承诺正式ETA。

下一步准备全部固定train512/development200的可验证几何缓存及正式配对训练代码。缓存只压缩完整当前预测几何的存储，不根据GT支持删点或改变float64 CV。正式两组各512更新，之后只评固定终点；继续比较原O完整GMO/common-change风险、CRN-CV和D的完整材料点误差，以及Cpl−Fix。只有真正共同改善才讨论扩大验证；共同运动评价与覆盖约束保持原约定。当前正式资源、缓存complete和训练源码尚未冻结，正式训练未启动。

根侧已核全部产物ledger、24个源SHA、两组4×4样本顺序和初态/RNG记录；未重新计算未保存的原始梯度或优化器张量。实际完整终点：`server_results/training/shared_rigid_native_preflight_v1/complete.json` SHA `198384ba7e768901c3c996504c897f5e7154bfdb4ddf61d3eb4da089a76d8212`；summary SHA `59ad77236568ae7d9933a1ac9a6630da576b85c68e524bfd7667a6361e3252cc`。独立审查另存，不能用文件名代替其实际审查结论。
