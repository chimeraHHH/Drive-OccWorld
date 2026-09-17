# 对象状态条件化未来预测：下一实验草案

**状态：正式协议已冻结。官方 CRN train512 状态推理及完整校验完成，v2 四次原生更新预检通过根任务与独立审计；V/G 单种子正式训练已在 H200 启动，最后核验两臂各 7/512 次更新（北京时间 2026-09-16 06:40:43）。以下保留设计依据，实际运行以 object_state_forecast_training_protocol_v1.json 为准；暂无 V/G 效果分数。** 2026-09-16。CRN 导出框的底面中心／几何中心约定已由实际运行源码确认；仅作确定的中心转换后，完整支持下四时域运动误差均低于 D（2 秒 3.386739 对 4.208240 m）。这支持输入状态具有可用信号，尚未证明它能改善 O 占据。该草案回答一个窄问题：相同检测几何条件下，**可部署的预测速度作为未来状态的前向输入，能否同时改善原生完整占据和同源点位移？** 不以新阈值、只保留检测成功对象或只报有利时域来定义成功。

## 依据与尚缺证据

K/B 已证明物理梯度确实能进入原 future-head，但其占据增益几乎相同，物理误差没有可信增量；这不证明未来特征无运动信息。R/R0 保持 O 冻结、在末端以运输特征修正输出，主要降低 FP，却损失动态召回，未证明运动连接的额外价值。历史 GT-CV 说明“知道过去对象状态后有可预测信号”，不是传感器恢复这些状态的证据；CRN 诊断检验其中可部署的一部分；修正原点后的 v2 结果已完成，原始 v1 结果保留，不把坐标修复当成算法收益。

MotionPerceiver 已有“位置／速度／尺寸 token → 场景状态传播 → 占据查询”的直接先例；不能把该一般连接包装成新颖性。其对象状态输入、二维任务、仅已观测对象的标签过滤不能迁入我们的评价分母。[已核论文／源码依据](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/object_state_conditioned_occupancy_sources_v1.md)

## 唯一两臂干预

暂称 **V（预测速度）／G（几何对照）**。两臂从同一最终 O 和相同新模块 seed11 初始化，保留同一冻结 D。检测框中心、尺寸、朝向、类别、置信度、顺序、有效集合及容量完全相同；V 输入预测 `[vx,vy,0]` 经 t0 旋转后的速度，G 在同一输入槽置零。CRN 未提供的角速度／加速度不虚构，首轮只比较现成速度。零是消融输入，不宣称场内所有物体真实静止。

不增加分数阈值或 GT 匹配选择，不按框裁剪占据输出。当前已确定的 GMO 类别过滤和原导出顺序一致；不因计算方便事后选 top-k。若所有检测框注意力不可承受，先压缩固定数量的场景 latent，而非删低分框。两臂都训练原完整 future-head、相同条件模块及相同位移读出；原 O 的观测编码器与 D 保持冻结。不得只让 V 获得额外框几何或额外损失。

## 轻量接口：条件进入 transition，不改末端 logits

```text
CurrentPredictedStates:
  center_R[B,N,3], size_wlh[B,N,3], rotation_R[B,N,3,3],
  class[B,N], score[B,N], valid[B,N], velocity_R[B,N,3]
  # R=t0 LiDAR；只用当前预测与 G0；不含 GT ID、future pose/dt。

StateConditioner(states, nominal_horizon, native_query_features)
  -> query_delta[B,200*200,256]

native_future_head(memory, unchanged_action_geometry,
                   query_features + query_delta)
  -> generated_states[3,B,200*200,256]

native occupancy decoder(generated_states) -> original occupancy logits
physical readout(t0_state, generated_future_state, nominal_horizon)
  -> frozen_D + delta_displacement[B,4,3,200,200,16]
```

当前 v2 保留 27 维当前状态，并在每个名义时域附加预测中心 p0+h*v 的 3 维数值，G 在推导前先把速度置零；状态 MLP 输入为 30 维。全部对象经 64 维编码与 cross-attention 汇入 32 个场景 latent，原生 query 通过低维 cross-attention 读取，再投回 256 维。共享时间偏置仅称时域偏置，不称其本身为对象动力学；v1 这条路径的可分离性已由实际 CPU 代数检查确认，见《object_state_temporal_interaction_诊断与决策.md》。输出投影零初始化；总参数 95,872。实际原生预检已确认初始 O/D 一致、更新后 V 的速度响应及 G 的零速度不变性、两类 loss 到状态 MLP 和未来主干的梯度连接；这仅确认实现可训练，尚不证明效果。容量不按开发分数选择。

优先利用原 `WorldHeadTemplate._get_next_bev_features` 在 transformer 前已有的 `rollout_prior` 加法入口：仅把新 `query_delta` 接进去，不修改原 action、SE(3) grids、memory 更新或 readout。原 detector 中每步 `pred_feat[-1]` 继续进入下一步 memory，因此条件可改变后续预测状态，而非只修当前 horizon 的分类分数。具体函数／边界需实施时再次按冻结 O 源码确认。

占据和位移都必须读取**同一条经条件更新后的 future-state 图**，位移头不得另接一条绕过 future-head 的 CRN-CV 输出捷径。可沿 K/B 使用同容量的 D＋零初始化修正读出，但两臂都保留状态梯度；本次变量是运动信息输入，而非 detach。两项损失都能作用于共享 future-head／conditioner，物理读出仍只在原合法 sparse 点受监督。

不把原生 hidden token 理想化为严格 t0 物理网格：既有审计表明显式查询／memory 几何与监督帧之间存在复杂关系，旧坐标修正路线未晋级。首轮不把预测框硬 rasterize 到未来 hidden 索引，不做 GT 位置采样或重新 warp 既有权重；状态位置用明确 R 数值编码，由原生 query 内容学习条件关联。若该关联学不好，应作为机制局限报告，不能据此断言对象状态本身无用。

## 数据、训练与预检边界

目前只有固定 dev200 的 CRN 当前帧全框导出，**train512 尚无对应预测资产**。需用同一官方 checkpoint／源码在原 512 个 train anchors 上只做推理，保留 CRN 原历史输入、全部框及原顺序；不重训检测器。记录原始导出／checkpoint／输入时间／坐标约定和样本哈希。标注框、GT 速度、实例对应、未来位置不得进入 conditioner；GT 仅用于原损失和评价。CRN 比 O/D 多出的检测器容量、预训练、历史帧和推理成本另行披露，V−G 只隔离在这些条件固定后的速度增量。

拟沿原固定 train512／dev200、四次样本排列、accum4、512 updates、seed11 作筛查；占据仍为原 CE[1,5]＋Lovasz 六项，物理仍为原 group/object-mean SmoothL1，权重均为 1，不把 focal 或仅 observed-agent 标签搬过来。新的条件模块优化器与学习率须在正式前明确；当前不写成已冻结配方。

资源先测两个不同 train scenes 的真实前向，再以固定前16 train anchors 做四次成对更新，临时权重丢弃、不读 dev 分数。必须实际确认：零初始化占据／D 一致；非零更新后速度输入能改变共享状态及两类输出；占据／物理 VJP 能到 conditioner 和 future transition；两臂相同 RNG／orders／优化次数；空检测集合仍可沿 O 稠密通路预测；真实显存与暖态耗时。零初始化首步上游梯度为零可能是预期，不能当断路，也不能只靠最后投影梯度宣称完整连接。CRN 推理及两臂训练成本分别计量，不能用旧 K/B 耗时充当本方案 ETA；预算在预检后冻结。

## 揭晓前解释边界

全原 GT／ignore／ROI、四未来 GMO、FP/FN、全部共同速度组和 transitions、t0 代价全部保留；物理按原合法源点及未来有效性评 D/V/G，并列出既有 CRN-CV v2 强对照的四时域 XY/XYZ mean／median／p90，完整保留未检测覆盖点，沿原 paired-scene bootstrap，不以体素数冒充独立样本。

- V 同时胜 O（原任务）、D（物理）且优于 G，才支持“预测运动状态的前向连接有额外效用”；仍是单 seed、历史开发集筛查，需必要完整验证。
- V/G 都提升而彼此接近，支持几何条件／容量／继续训练的共同作用，不能声称速度机制成立。
- 物理更好而占据无增益，说明状态可用于位移但未转化为完整任务收益；占据提高而物理或动态覆盖变差，也未达到用户联合目标。
- 两臂均无收益，只否定此输入质量、条件化位置和预算下的实现。它不能区分检测噪声、集合编码丢失、帧关联、优化不足与状态信息无用；不要自动加层、延长负候选或重启旧分支。下一动作应由实际覆盖、条件敏感性和共享梯度证据决定。
