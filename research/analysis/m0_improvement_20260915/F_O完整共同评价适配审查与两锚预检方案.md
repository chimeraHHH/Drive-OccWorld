# F/O 完整共同评价：只读适配审查与两锚工程方案

2026-09-15。本轮只读审查并准备后续方案；没有修改评价器、训练器或协议，没有联网、上服务器、启动推理或生成候选成绩。正式 F/O 仍按已冻结训练协议运行。此文不授权或触发 full validation。

## 结论

现有联合评价的数据路径适合复用：每个 anchor 只用原 M0 编码一次原生图像/雷达观测，捕获完整 `future_pred` 边界，再向各固定未来 head 提供该边界的独立副本，使用原完整 GT evaluator。**必要科学变化仅是模型集合及其最终 checkpoint 身份，以及对应的预登记比较；F/O 的监督与损失改动不进入推理。**

旧工具不能直接换文件夹启动。它们硬编码两槽模型、父训练协议及旧七个比较；必须另写独立的 F/O 评价 revision，保留本轮冻结文件。现有检查大多会直接拒绝 F/O，不能删掉检查来“兼容”。新的候选必须先完成固定最终训练、dev200 和五模型配对收据，随后才有可冻结的最终模型清单。

当前未找到已完成的 `joint_native_evaluation` 真实 pilot/full 收据。既有真实缓存、frame 与 objective 预检能支持其局部接口，**不能替代原始 loader→五模型联合流→分片合并的真实验收**。以下是源码审查，未穷尽 CUDA/原始数据 I/O 的运行风险。

## 本次读取并核对的固定身份

| 文件 | 当前 SHA256 |
|---|---|
| `joint_native_evaluation.py` | `280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578` |
| `merge_joint_evaluation.py` | `e54e1eda2647f487f9b3f1bb1441307f7ebd7c7125e16296a29bc0508279bf24` |
| `objective_supervision_protocol_v1.json` | `275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d` |
| 父 `protocol_v2.json` | `071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a` |
| `joint_preflight_selection_v1.json` | `e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d` |
| `full_validation_selection_v1.json` | `60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391` |
| `objective_supervision_aggregate.py` | `9e2327d51dd34e3f207d216b0966177531ffdc4e6ecdddddf0f06004ace5e256` |

full selection 当前确实有 5,119 个唯一 `official_index=0..5118`、150 scenes。仍沿用这两个事先冻结的 selection；不按 F/O 表现删样或重选 pilot。selection 内历史 `launch_status` 不需要修改，可在新运行协议另记本轮状态。

## 必要改动与现有问题

| 位置 | 已确认事实或适配阻断 | 最小处理 |
|---|---|---|
| producer `MODELS/ARMS`、`run` 加载循环 | 旧集合为 M0/M0_fp32/native1/persistent2/rolling2；`mode=name` 会将 F/O 传给只接受 native1/persistent2/rolling2 的安装器 | 新集合固定 M0/M0_fp32/native1/F/O；四个重放模型均 `install_observation_memory(...,'native1')`，F/O 只严格加载最终 `future_pred_head`。不调用 objective loss adapter，也不安装 frame adapter |
| producer `frozen_arms`、`contracts` | 强制全部 arm 来自父协议、source 字段恰好三项、migration.mode 等于 arm；F/O 实际为新协议且 migration.mode=native1；C 与 F/O 目录也不同 | 显式区分父协议与 objective 协议、`control-run` 与 F/O runs-root；复用已审 C/F/O 完成收据与实际 payload 核验，绝不复制/改名旧结果以伪装来源 |
| producer checkpoint 加载与 `loaded_models.json` | 旧版本校验最终 update/pass/order、严格 `load_state_dict` 和每个 tensor 相等；其记录 tensor SHA，但旧臂 manifest 没有本轮 F/O 的 final tensor SHA 门 | 保留全部旧门，并要求实载 F/O head digest 等于其 `complete.final_head_state_sha256`；实际参数名、形状、dtype、容量均与 C/native1 一致。新 common contract 固定所有 checkpoint 文件 SHA 与最终 tensor SHA |
| producer `pilot_reference` | 引用旧三臂 dev JSONL，FP32 只从 native1 frozen reference 取得 | 保留同一 native dev200 cache 和 C 的 frozen FP32 reference；F/O 对照其各自固定最终 dev200 JSONL。必须先通过本轮目标协议/receipt/hash/身份核验 |
| merger `validate_arm_receipts/validate_loaded_models/collect` | 硬编码父协议、旧 source 集、persistent/rolling 容量对照、旧 producer SHA；不能识别新目标协议 | 新独立 merger 同时绑定父与 objective 协议、另冻结新 producer SHA，核 M0/M0_fp32/C/F/O 的真实最终身份与单槽 migration；不放宽其他 hash/完整覆盖门 |
| merger full 分支 | 直接调用 `aggregate_memory.aggregate`，其模型名与七个比较硬编码旧三臂；仅改 merger MODELS 会报错，映射旧名还会误标比较 | 只复用 `aggregate_memory.metric_arrays`，以及已冻结 objective 聚合器的**纯数学** `aggregate`/场景 bootstrap；不调用其 dev200 加载 main。五模型与六个比较固定，四主比较为 F−M0_fp32、F−C、O−M0_fp32、O−C |
| pilot provenance hash | `pilot_reference_sources.json` 在旧 merger 中读取并记录 digest，但旧 shard complete 未直接绑定该文件；内容另有若干来源交叉核验 | 新 revision 可将该小文件 SHA 直接加入 shard complete，让 pilot 参考来源也进入完整 hash 链；这是证据闭环，不改变推理或指标 |
| 峰值/时间与进程治理 | 旧 producer 同时保留 5 份完整模型；`model_seconds.M0` 包含原数据准备及编码，variants 包含重放/evaluator/CPU摘要；没有 allocator cap 参数，默认 max_seconds 不代表已审 full 预算 | 保持首版模型驻留方式，先实测整体峰值，不同时重构为共享 backbone。新增运行层预算/两卡空闲门和已有受控进程组规则即可；不得据旧训练或缓存时间宣称 full 联合吞吐 |

原数据/推理部分不应改动：`cfg.data.test`、只读 5-sweep radar、`future_metadata_only`、annotation SHA、`dataset.usable_index` 和 `_prepare_data_info(data_info_index,rand_interval=None)`；当前 `_selected_rows` 已逐项核原索引/token/scene。`Drive_OccWorld.forward_test` 每个输入自带历史队列并重新计算 history BEV，而不是跨 anchor 复用上一项 `prev_frame_info`；因此源码上允许按 selection ordinal 模 2 分片。仍需实际分片 pilot 证明实现路径。

M0 原始编码/未来预测保持 TF32=true；M0_fp32/C/F/O 的未来 head matmul TF32=false、cuDNN=true，所有模型 `eval/no_grad`，无 optimizer，所有参数冻结。t0 原观测、未来 ego/action 和物理空间对齐全部相同。GT 只交 evaluator，不能进入未来预测条件。F/O 训练改变的只是 final head 参数，其推理图与 C 相同；因此无需保留 12/6 loss 计算或完整分辨率训练插值。

## 启动前必须具备的输入

1. 本轮 F/O 完整 campaign/summary 收据以及两臂 `manifest`, `training_complete`, `complete`, `latest.pth`, `training.jsonl`, `development_records.jsonl`, `control_reuse.json`。两臂必须为 4 passes、512 updates、2,048 examples，实际 4×RandomState11 sample orders 与 C 相同，无 failed/interrupted 状态；不接受正在写入或未完成评价的 checkpoint。
2. 旧 native1 七个冻结控制文件和 M0 epoch24 checkpoint；C 和 F/O 同一 M0 初始化、13,274,016 个可训练未来 head 参数、同原优化预算和未来数值策略。只用新的 full 评价协议固定候选，不改训练协议。
3. 原 config、父与 objective 协议、两份 selection、官方 annotation 及可读原始图像/occupancy/radar 资产；full 模式依然读取原始 loader，不拿 dev200 cache 当全量输入。pilot 额外需要完整 dev200 cache 的两条实际 inputs/GT/native_preds 文件。
4. 新 evaluator/merger 的独立源码 SHA 与依赖 SHA、全部最终 checkpoint SHA、固定模型集合/比较/数值策略、完整 evaluation selection SHA、工程 pilot PASS 收据、明确的单次 wall-time/显存预算。所有路径必须在实际环境核实，不凭本地镜像推断可访问性。

在 full 读取任何候选评分前，冻结该清单与四主比较；保留 C−M0_fp32、M0_fp32−M0 作为控制。无阈值拟合、R4 校准、best checkpoint、新训练 seed 或 F+O 合并。

## 仅两条实际 anchor 的工程验收方案

固定复用已有 pilot selection，分别是：

| official index | sample token | scene token |
|---:|---|---|
| 2036 | `297c52902a384b84ba179f3160ac54e9` | `aedcd3cf7c4a49d7a4a43ab7443a9eb1` |
| 2998 | `01076466d815421c857d2c8cf8034488` | `7bd098ac88cb4221addd19202a7ea5de` |

建议一次 **2 shards，每 shard 1 anchor**，使用两个已核空闲 GPU，完整测试同一 common contract 的排他创建、跨 shard 一致性与合并；只评价这两条、无优化器。单次短时上限由根任务冻结（可采用每 shard 600 秒的工程级上限），无失败换样或自动重试。两个都是各进程首样本，资源结果应明确为含冷启动的工程数据，不能称稳定尾部吞吐。

必须同时通过：

- **真实 loader parity**：新捕获的完整 typed input tree 与旧 cache inputs 摘要相等；所有 7 帧 GT 逐位相等；M0 原生 5 时域×3 层 logits 与旧 native_preds 逐位相等。`box_type_3d` 继续使用已审 module+qualname+source SHA 标签，未知 metadata 类型拒绝。
- **各模型身份**：M0/M0_fp32 的初始 head SHA 固定；C/F/O 实载 tensor SHA 等于各自冻结 final state。五模型均为原单槽架构、eval、无优化器、无 objective/frame 推理干预。前后 head SHA 不变。
- **共同样本/标签**：五模型使用同一边界的独立副本，重放前后 CPU/GPU typed tree hash 不变；各模型 `(5,2,2)` confusion 的 GT 行和等于原 7 帧计数中的最后 5 帧。当前＋四未来分别核对，不只看未来均值。
- **已完成 dev 记录 parity**：五模型全部 5 时域 confusion 分别精确等于上述两个 token 在其冻结 dev200 JSONL/cache 中的结果；F/O 不必等于 M0，只须等于自身固定最终权重的已完成评价。
- **联合 hash 与覆盖**：两 shard 中 token/ordinal 互斥，合并后恰好两条官方顺序；contract→manifest/loaded_models→index→sample complete→records 及 pilot reference ledger 全链一致。缺一条、缺 shard、不同 checkpoint/source 或 GT 行列转置必须拒绝。
- **输出边界**：只给 parity/resource PASS 或失败原因，`performance_metrics_computed=false`、`performance_success_claim=false`，不报 pilot 提升；保存小型 JSON/日志，不保存完整大 logits/GT 或任何 checkpoint。

本轮已有的 train-mode objective 预检使用的是两条 **train** anchor，并且没有原图像/雷达 loader；不能将其 PASS 当作这两条 validation anchor 的联合验收。若新 pilot 在来源/bitwise 上失败，保留证据后查原因，不放宽精度、改 GT 或替换样本以过门。

## 资源估计依据与不能外推的部分

以下均为已回传的真实旧收据，读取位置分别是 `server_results/cache/campaign_cache_v1`、`server_results/campaign_memory_v1/runs/native1/complete.json` 和 `server_results/objective_supervision_preflight_v1/summary.json`：

| 实测任务 | 数据 | 适用范围 |
|---|---|---|
| 原 native dev200 cache 抽取 | 总 777.00 秒；样本段 mean 3.220 秒、p95 3.893、max 4.452；其余约132.94秒；原抽取峰值约4.143 GiB | 包含原数据准备、M0 forward、落盘和 parity replay，不能拆成纯编码时间 |
| 原 native train512 cache 抽取 | 总1,779.50秒；样本段 mean3.356、p95 4.212、max5.406 | 同样含缓存写入与校验，也不是 full joint 流推理 |
| C 固定 dev200 cache 评价 | 45.412秒，即0.2271秒/anchor | 是一个缓存未来 head 的加载/推理/完整 GT 评价；没有完整原图像编码，且不含联合5模型驻留与全部摘要成本 |
| F/O train-mode 工程预检 | forward约0.0583/0.0584秒；实际训练峰值33.139/15.916 GiB | 说明同推理图的 head 量级；backward、完整 loss 与 AdamW 不应计入推理，也不能用训练峰值替代联合驻留峰值 |

四个缓存重放的粗数量级是 `4 × 5119 × 0.2271 ≈ 4,649 GPU-resident 秒`，即约1.29小时，**尚未包括一次原生图像/雷达编码/GT加载**，也不是严格下界，因为联合流会共享输入加载而新增摘要开销。原抽取已经包括一次 parity replay 和大缓存落盘，直接将3.220秒乘full条数同样会混入本轮不需要的工作。只有新的联合 pilot 才能测其真实完整路径。

后续预算用同口径记录估计：单卡 `setup + 5119 × joint_seconds_per_anchor + final_hash/merge`；双卡每 shard分别为2,560或2,559条，墙钟取较慢 shard，GPU驻留总成本取两者之和。两条首样本只能作为粗筛；不能假定两卡同读原数据时线性加速或代表最坏 I/O。先读取 pilot 的 `M0` 与四个 replay 时长分解、加载全部模型后的峰值，再由根任务冻结一次 full 截止。此报告不先承诺“几分钟跑完”。

单个原生 logits 数组为76.8MB，7帧uint8 GT约73.4MB、GPU long GT约587.2MB；现有工具每样本顺序处理并只持久化摘要，可继续沿用，勿将5,119条输入或预测全集驻留。五模型×5,119×5时域×2×2的uint64 confusion原数组约4.10MB，末端保存 raw confusions 的开销很小。完整模型深拷贝、临时数组/摘要和真实数据 loader 才是应实测的内存部分。

## full 结果的科学边界

主指标仍为每个未来时域先合并全部样本 confusion、计算 foreground class1 GMO IoU，再对0.5/1/1.5/2秒算术平均；另报 pooled future GMO 和 binary mIoU，不可互换。bootstrap改为实际150个场景，保留每场景全部自然数量 anchors，10,000次seed11配对抽样；不能套用dev200的100scene×2anchor加载断言，也不能先平均每scene IoU替换原主指标。

全量5119包含已使用的开发场景和更多历史验证曝光，因此是完整原生验证确认，**不是新的盲测或训练种子稳健性证据**。四项主比较逐项95%区间未作多重比较校正；模型选择/历史探索不确定性也不在区间内。full成绩必须同时呈现两个固定候选与对照，不靠删掉较弱臂改善叙述；无论结果如何，不自动宣告论文或总科研目标完成。
