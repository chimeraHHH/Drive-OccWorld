# 最终 V checkpoint 的速度分配干预草案

2026-09-16。**未执行；不训练、不修改当前 V/G 源码或协议，不新增本轮晋级门。** 仅在固定 512 更新及全部开发评价完成、最终资产可认证后考虑实施；不能依据本诊断改写冻结主结果。本草案只读核对实际 conditioner、状态读取器和原生评价路径，没有读取进行中的性能结果。

## 问题与三个条件

对象状态经 32 个全场景 latent 再被 native queries 读取，速度能够改变输出，只证明计算通路存在。本诊断问：**在同一最终 V 权重内，保留速度集合时，原始对象—速度分配是否比破坏分配更有预测价值？**

根任务补充的空间接口核验：当前 conditioner 的 query 是 `bev_embedding.weight + prev_features[:, -1]` 经投影与 LayerNorm 后的结果，具有格点身份和历史特征，不能说它“完全没有空间信息”或只输出场景均值。但是 conditioner 本身不接收 native 的 `bev_pos`、`tgt_points` 或 `ref_points`；这些坐标／位置编码在后续原生 transformer 才进入。对象的数值中心、速度及 `p0+h*v` 因而需要通过学习建立到 query 的联系，**派生未来中心并不等于已把对象放到对应未来物理位置**。这定位了一个待检验的关联接口，不证明它已经失败，也不允许在本轮中途改成空间散射或修改评价支持。

固定最终 V 的 entire future-head、conditioner、physical readout，以及原 observer 和 D，三条件均 `eval/no_grad`、相同原生精度和输入：

| 条件 | 输入改动 | 回答的问题 |
|---|---|---|
| N：原分配 | 原 centered CRN 状态 | 重现本轮最终 V |
| P：同帧置换 | 只将有效 GMO 对象间的三维速度向量重新分配 | 原分配相对同一速度多重集的价值 |
| Z：零速度 | 仅三维速度槽清零 | 同一 V 权重对速度信息的整体依赖 |

三者全部通过 `use_velocity=True`；Z 清零输入后，模块自然令各时域派生中心等于当前位置。**V(Z) 不等于已独立训练的 G**：V−G 是训练干预，本表是同权重推理干预。既有 O/G/D/CRN-CV 主结果可并列作上下文，但不得混成该三条件的因果比较。

## 实际接线与可重复的置换

`DevelopmentPredictionInputs.get()` 已按原导出顺序返回 `states[1,N,27]`、`valid` 和 `retained_original_box_indices`。读取的是已转换几何中心的开发预测；只取当前 `G0`，不再次加半高。固定槽 `0:24` 为中心／尺寸／旋转／类别／score，`24:27` 为同一 t0 LiDAR R 中的速度除以 20。**在 pack 之后 clone 并只交换 `24:27`**，保留全部前三维分量，包括 global 水平向量旋转到 R 后可能出现的 z 分量；不重新估计速度、不接触 GT。

建议事先固定一种、每 anchor 一次的无固定点映射：对原有效 GMO 框索引，以 `SHA256("velocity-assignment-v1|" + sample_token + "|" + original_box_index)` 排序，再令排序后每个对象接收下一个对象的速度，末个接收首个。这样 N≥2 时 donor index 均不同；规则与框位置、速度数值、GT 和结果无关，不调用或改变模型全局 RNG。N=0/1 时 P 原样保留，仍计入完整评价；重复／全零速度可能让索引换了但数值未变，必须同时记录“重分配对象数”和“实际速度改变对象数”，不能强行剔除这些样本。只用这一预先固定映射，不寻找破坏最大的置换或较好看的置换种子。

全四时域共用同一 t0 分配。`ObjectStateConditioner.forward()` 每次从原中心重新算 `p0+h·v`，因此置换同时改变速度及它派生的未来中心；不能单独保留原未来中心，那是另一干预。整个四步 native replay 必须重跑，允许先前生成状态自然影响后续 query，不能复用 N 的未来 memory 给 P/Z。整体重排完整 token 会保留对象—速度关系，且集合 attention 对此原则上置换不变（浮点归约可能有微差），**不是本诊断**。

## 最小推理和记录路径

沿正式 trainer L402–443：`input_only → D(t0 BEV) → get(states) → condition_future_head → capture_native_terminal → readout(features,D)`。D 每 anchor 计算一次，N/P/Z 共用；每次 `native.replay` 自行 clone 可变输入，hook 独立上下文并退出清理，帧序固定 1–4。不更改原 ego action、几何、valid mask 或 sparse 源点。推理时保持相同 RNG 状态与模型 mode；权重和输入前后 digest 不变。

原固定 dev200／100 scenes 全覆盖，N 先重现原最终 V 的逐样本完整 confusion 与对象误差记录；重现不一致时先查加载／精度／输入，不能解释为干预效果。三条件预测全部生成后才读取 GT／sparse labels 做评价。物理标签仍是原全部 16,074 对象—时域支持，不按 CRN 覆盖、速度实际改变或检测成功筛掉对象；原缺失标注和无唯一源点缺口照旧披露。

最小新增记录是：最终三模块与 D／observer 的 SHA、原预测资产 SHA、每框 donor 原索引、除末尾三维速度外全部状态／valid 的一致性、速度多重集一致性、实际变化数及幅度、四帧 hook 回执。N/P/Z 的 t0 在同一 V 权重下应自一致，不能要求它们等于 O 的 t0。保存各时域 native logits／terminal-feature／physical-field 的差异摘要，另保存原整数 confusion、八组和 transition 计数、原对象 EPE；差异幅度只证敏感性，不是准确性评分。

统计复用原定义：四未来 pooled GMO 平均、全五时域及 FP/FN／全部组／transition；四时域各组 XY/XYZ EPE mean、median、p90。主要干预差为 P−N 和 Z−N：GMO 为负、EPE 为正表示破坏后变差。仍按原场景成对 bootstrap 10,000 次 seed11；它只反映场景抽样不确定性，**不涵盖换一套置换的随机性**，更不是新训练 seed。没有新阈值、最佳时域或诊断 pass 分数。

### 经论文与源码核对的位置查询先例

2026-09-16 补读 [MotionPerceiver 原论文 §IV-C 式(11–12)](https://arxiv.org/html/2306.08879v2#S4.SS3)：其占据 emission 用位置编码作为查询读取场景 latent。作者固定 commit `cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f` 的 [PerceiverDecoder](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/perceiver_io.py#L333) 同时实现 learned 和 Fourier 两种分支；不能只凭 Python 默认值认定论文配置使用 learned。此次核对的 [no-ctx](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/cfg/no-ctx.yml#L34)、all-ctx 和 occ-flow 配置均明确选 `position_encoding_type: fourier`，32 频带。decoder 由规则位置网格生成编码、作为非持久 buffer，再跨注意力查询 latent；输入适配器对对象位置／朝向作频率编码并拼接速度等额外状态。五个实际源码／配置文件 SHA 与既有 Git-blob 来源记录逐一相符。

**由来源得到的待验证推论：** 全场景 latent 压缩与显式位置查询可以共存；“使用了 Perceiver 式 latent”不足以保证数值对象状态与原生格点已建立相同空间联系。我们的 conditioner 使用原 BEV embedding 与最近特征，原 native transformer 仍有自己的位置编码和参考点，因此不能称整个模型缺失空间信息。若最终结果及同权重干预提示“速度有响应却无定位收益”，可优先检查这一接口的关联学习，而不是直接推断需扩大 latent 数量。该先例不证明换成 Fourier 就会改善本模型；任务、状态质量、输出维度和训练预算均不同，且原 WOMD 配方的标签过滤不能迁入本任务。

## 可区分的现象及不能越过的界限

- **N 优于 P 且优于 Z，尤其物理和完整占据均如此：** 支持原始速度—对象属性关联在该 checkpoint 上有用；不证明学到了唯一实例对应，也不证明任意检测误差下稳健。
- **N≈P，而 N 优于 Z：** 与利用速度总体统计相容，也可能是重复速度、局部混合、影响较弱或原指标对变化不敏感。结合实际变化数与输出差异判断信息量，不能直接判为“只用场景均速”。
- **N/P/Z 输出不同但准确性相近，或只有一类输出变好：** 通路存在，预测价值或共享利用尚未得到联合支持；不能把敏感性冒称正确运动关联。
- **P 比 N 差，但 Z 不差／更好：** 错配速度有破坏力，不等于原速度提供净收益。N≈P≈Z 也不能把 V/G 的共同增益唯一归因于几何；继续训练和新增容量仍是混杂解释。

最大限制是 **P 属于人为错配输入**：它保留帧内速度多重集，却打破速度与位置、类别、朝向、置信度等联合关系，可能出现不合理的行人／车辆速度组合。因此性能下降包含 OOD 敏感性，不能视为交换性成立的随机化因果检验或自然检测误差模型。零速度同样可能 OOD。当前方案不额外添加按类别／置信度分层置换来挑结果；若确需分离这些因素，应另列后续问题。

全场景 latent 允许单对象干预影响远处 query；不能用“变化落在预测框外”自动判错，也不能把 hidden token 直接当作已严格固定 R 的物理格。attention 图、latent 范数或单个漂亮场景都不替代完整支持的实际误差。该诊断只解释当前模型如何使用状态，不改变本轮正式 V/G 对用户“超过 O 且运动更准确”的原结论。

## 本次核对的冻结来源

- `object_state_conditioner_v2.py`：SHA256 `dad88c05105d85d9b2a6094c687630f1d661cd7a4e35be42a15a5d01b94a8f36`；pack、速度清零／未来中心、32 latent、四帧 hook。
- `object_state_prediction_inputs_v1.py`：`55e39715fdf19e5ae1f227a943a843b194eabed08e788ccb2694a040d6f18256`；`current_states_numpy` 的全局向量到 R 转换。
- `object_state_forecast_development_inputs_v1.py`：`7cd498d13d45d5960c6f494ebf7b750718b397d90720ecd745a8c527b0dace06`；200 样本身份、完整框、一次原点适配、当前 pose 边界。
- `object_state_forecast_train_v1.py`：`2841aeaa533b4883e635014fa74d22520d94890ef1939a136f75beb8366ad162`；最终 payload 和原生 dev 路径。
- `future_state_motion_v1.py`：`96e796dd5bb6d33949a03a6c451919ef2e19c75b81eaea4b10e2c98f8ecc176e`；共享未来 features 读出。另读 `object_state_forecast_preflight_v2.py::intervention_probe` 及 `native_state_cache.py::replay`，仅作为实际入口依据；本草案不修改或重新执行它们。
- 原生 `world_head_base.py`：`8e7a7adabb968a67ddc52c46a8927a90d3286572fc8395dd8fa98d7ffe40e4cf`；根任务核对本地字节与当前运行绑定的 `runtime_source_contract.json` 一致。`rollout_prior` 加到原 BEV query 后，原 transformer 仍接收自身的 `bev_pos`、目标／参考点及动作条件；本次未修改该链。
