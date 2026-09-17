# Persistent-reference 两槽实验：独立设计审计

2026-09-15。基于根 agent 提供的候选协议、`goal_state.json`、`native_catalog.py`、时间建模文献报告，以及本地 `Drive-OccWorld-sota-p2` detector/head/decoder 的只读检查。没有实现、连接服务器或启动训练。当前结论：**设计值得做，但只在原生回放、标签时间对应、两槽初始化与预算匹配通过后，才进入机制比较。** 未见完整冻结的实验 preregistration；以下为待落实的设计要求，不是已通过认证。

## 1. 可以识别什么，必须有哪些对照

四个条件不可合并：

| 条件 | 初始化与可训练部分 | 作用 |
|---|---|---|
| M0-native1 | 原 checkpoint，冻结 | 原始真实基线；相同原生 radar、GT、evaluator 与样本 |
| Continue-native1 | 同 M0，完整 `future_pred_head` 训练 | 控制额外训练/数据重放收益 |
| Rolling2 | 两槽扩展，同一初始化，完整 future head | 控制额外记忆容量与参数 |
| Persistent2 | 与 Rolling2 完全相同，只改变队列策略 | 检验持续读取参考观测的策略 |

Persistent2 = `[F0, Fhat(h−1)]`；Rolling2 = 最近两状态。初始均 `[F0,F0]`。在完全相同权重、相同前序输出、无随机扰动的回放下，**h=1 与 h=2 两者都应相同，策略差异从 h=3 才出现**。训练后共享参数已经不同，早期时域不再必须相同；不能把训练后的 h=1 增益自动归因于多读了 t0。

同容量并不能单独证明“真实观测信息”是唯一因果中介：persistent 同时改变记忆年龄、坐标变换与反向传播路径。主结论应先限定为**记忆策略**。若主比较通过，再做预算很小的来源替换/关闭读取敏感性诊断，说明该测试可能出分布；不要仅用 attention 权重证明模型利用信息。无需为每个诊断再训练一整组模型。

原理已有 StreamingT2V/WorldMem/OccProphet 先例，不能包装为首次长期锚定、首次双记忆或首次防止累积误差。可以探索的科学问题是：在原生未来占据契约下，有限容量记忆应保留观测还是生成状态，以及收益是否来自更好的时序证据访问。

## 2. 最容易造成假阳性/假阴性的管线风险

**记忆数量与 GT 历史长度必须分开。** 本地 `compute_occ_loss` 通过 `future_pred_head.history_queue_length` 切 `occ_gts`；`future_pred` 还每步裁 `cond_norm_dict['occ_gts']`。实际 S0 的 history_queue_length=2（真实历史两帧），memory_queue_len=1；令真实 history 长度跟随 memory 槽数变化，会造成标签时间错位。两槽必须只是 latent memory；真实 history_queue_length 继续为 2，GT 帧顺序、损失帧选择、时间间隔仍照原生契约。每个监督输出保留 `(scene, anchor_token, target_token, actual_timestamp, horizon)` 收据，不以数组位置代替时间对应。

**持久锚的变换与条件也要持久。** 固定槽保存 F0、自己的 reference-to-history 变换和时间身份；生成槽保存对应生成状态的变换。每轮只更新生成槽。不得给 F0 使用最近生成帧的 pose；也不能把静态 ego 对齐声称为对象运动补偿。`prev_render_neck`、语义/ego normalization、frame embedding、ref_points 的槽轴都要一起核对；若原生训练条件允许 GT 语义，必须按来源帧配对，评价时禁止未来 GT 作为预测条件。

**缓存只能截断在冻结部分的确定性边界。** F0 可能已包含历史视觉融合和原生 radar，不应称作“只有 t0 单帧传感器”。不要用 R2 的 common/去晚点/重算雷达缓存代替 native 输入；也不要复用旧 future feature 缓存，它们是此次可训练 head 的输出。冻结 backbone/radar/参考 encoder 及 BN/dropout 状态，缓存特征、必要坐标/action 元信息和 provenance。不能先调用含 future label 的训练路径生成缓存，再认为 detach 已清除了泄漏。

**缓存改变训练分布，要对所有训练臂一致。** 单视图确定性缓存通常取消在线图像增强；这是“冻结表示上的 future head 继续学习”，并非原始端到端训练复现。Continue-native1、Rolling2、Persistent2 必须用完全相同缓存和增广策略。若使用几何增强，坐标、radar、action、GT 同步变换；不能只翻转特征。

**参数匹配不等于函数匹配。** 两槽初始化复制 frame embedding、sampling-offset 输出行和 attention-logit 输出行，并在正确的 head×level×point 轴复制。重复相同槽后 softmax 权重总和应保持原函数。必须实际核对原生→缓存单槽的全时域 logits/confusion，单槽→双重复槽的输出误差，以及 Rolling2/Persistent2 的初始化前两时域等价。限定 dtype 误差容忍并验证不会改变报告级混淆矩阵；不能仅检测“能运行”。

**损失与梯度预算必须匹配。** 保留原生有效体素、ignore mask、GMO 类映射、完整未来时域和原损失。不要增加槽时把逐槽辅助损失简单求和而意外翻倍；记录各项分母、head 参数清单与实际有梯度参数。回传所有原本应该回传的生成步；不得仅某一臂 detach 历史或只训练最后分类层。支持无效帧/scene reset 的样本不要跨场景拼接队列。

**未来 ego/action 是明确的条件边界。** 为公平超过 M0，本轮可以保留原有真实未来 ego/action，但必须逐臂相同，并把结论限定为这种条件下的预测；它不是完全自主闭环预测。未来图像、radar、占据 GT 不能进入缓存或推理条件。dataset `future_metadata_only` 名称不是证明，需检查实际被读的字段。

## 3. 先量成本，再锁定分阶段预算

Preflight 只用于运行/成本/正确性，不用于调候选参数。先测：若干合法训练和开发场景的原生编码与磁盘读写；三种可训练结构各完成初始化、warm-up 与覆盖四个未来时域的完整前后向；最后测真实评价与 GT 加载。记录稳态/尾延迟、峰值显存、缓存 bytes/anchor、训练 step 的样本数与有效时域。参数复制后的单步梯度有限且 optimizer 真正更新 future head，才说明训练路径可用。

预算写为 `缓存成本 + Σ(实际更新数 × 各臂稳态步耗时) + 每次完整开发评价 + 最终全验证评价 + 预留`，而非先拍定“1200 步/一小时”。三臂以相同样本曝光次数为主，报告不同 wall time；若强行等 GPU 秒导致数据曝光不同，那是另一个效率问题。可缓存只需的 anchors，不能未经估算就把全部训练集密集特征落盘：假设 200×200×256、FP16，仅一个 F0 就约 20.48 MB/anchor，FP32 加倍；实际以 preflight 为准。

建议阶段：

1. **正确性/可学习性门**：少量 train anchors 的全 head 更新和确定性回放。不看开发成绩选架构；低成本过拟合可检验损失/梯度链。失败只能说明实现或预算不足，不能否定记忆假设。
2. **中等样本机制筛选**：预先按官方 train 场景分层或 SHA 选覆盖充分的场景，再在每场景选多个时间分散的 anchors。一个可讨论的起点是 64–128 个 train scenes、每 scene 4–8 个 anchors；具体数量由成本确定。开发集仅来自预指定 development scenes，各取多个分散 anchors，不用 32 个相邻/同一 scene 的 anchors 假充独立样本。三臂样本、顺序、seed11、更新次数一致。预注册少量检查点和选择指标，输出全部曲线。
3. **扩训练/扩评价**：当三臂训练稳定、有足够有效 scene/foreground 支持，且 Persistent2 相对 rolling/continuation 显示有意义方向，或 pilot 区间宽到无法排除有意义增益时，扩大训练覆盖/训练曝光；不要把“尚不显著”当提前否定。若 train loss 未稳定、关键子域几乎没有样本，先解决覆盖/优化，不能继续同一小集刷 epoch。
4. **最终 full5119**：模型与 checkpoint 选择规则在 development 上冻结后，一次评价完整 5119 个原生 validation anchors；M0、Continue-native1、Rolling2、Persistent2 用同一 catalog。四臂均报告，不能只展示最佳候选与旧 M0。full5119 是原生验证任务的最终效果判断，不是 pilot 的放大打印，也不是只评一个好看的子集。

以上阶段是建议，不是固定样本量保证。每场景多个 anchors 改善时间覆盖，独立单位仍是 scene。短序列的多轮重放不能称作几千独立样本；`steps × batch / chosen_anchors` 才是本轮样本曝光 epoch，必须另报 unique scenes/anchors/完整时域体素覆盖。小集 20 epoch 和全训练集 20 epoch 不同，不能仅凭 epoch 数判断充分训练。

## 4. 切分与停止规则

官方 train/val scene 必须严格不交叉，catalog 对合法未来帧逐项验证；同 scene 的相邻 anchor 不跨 train/dev。开发检查、选择 checkpoint 与 full validation 的历史暴露要如实保留。locked50 可用于本轮不参与选择的核对，但如果以前完整 val 曾被查看，不能改称新的未见测试集。full5119 包含用于开发的场景，场景配对 bootstrap 也不能消除模型选择偏差。

建议主效应预注册为平均四时域原生 fine-GMO 的 Persistent2−Rolling2，同时以 Persistent2−Continue-native1 和 Persistent2−M0 验证实际效用；h=1/2、h=3/4 单独报告以考察记忆差异开始时域，不能看到结果后挑 horizon。NLL、Brier、AP 和 FPR/recall 是解释与风险边界，不能替换主任务。GMO 可移动类别不能直接叫实际运动子集。

允许的否定结论应分级：

- Pilot 区间宽、训练尚未稳定：**信息不足**，不说 persistent 无效。
- Persistent 和 rolling 均超过 continuation，但互相无稳定差异：支持两槽容量/优化收益，尚未支持观测锚策略。
- Persistent 优于 rolling、但不优于 continuation/M0：机制比较可能为真，实际改进目标尚未完成。
- 预算匹配、覆盖充分、实现通过，仍没有超过 M0：否定**本配置及预算下的收益**；不是所有锚定记忆或全部雷达信息无效。
- 全验证主差值区间排除一个预登记的最小有意义提升，才可说该提升幅度被本轮数据否定；仅区间跨零不是等效证据。

## 5. 什么足以宣称超过 M0

需要真实可重放 checkpoint、不可回填的源/数据/缓存/参数收据；原生 5119 合法样本全完成；原始 fine-GMO 主指标优于 M0，且相对同预算 continuation 和 rolling 对照仍有收益。报告场景配对效应区间、各时域、必要 FP/FN 分解及计算成本；没有用评价集调阈值、重映射标签或改 GT 分辨率。建议在冻结协议时定义“有意义”的最小增益，而不是看结果后把任意正数认定成功。

一次单 seed 的真实提升可以满足当前工程/科研推进目标，但不能宣称跨训练种子的稳定性；场景 bootstrap 仅反映固定训练模型的场景差异。进一步的 ICLR 贡献需要说明该原则为何在不同主干/预测条件下成立及何时失效，不能仅靠 M0 一个配置的 +IoU 完成新颖性论证。

审计补充（根 agent 提供的冻结事实）：selection_v1 SHA256 `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d`；256 个官方 train scenes×2 anchors=512，100 个 development scenes×2=200；preflight 为其中两个不同 train scenes。身份已经冻结，训练/驻留预算尚未冻结，本报告的样本规模范围仅是此前设计建议，不替代该已冻结选择。
