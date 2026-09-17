# 提名候选的原生联合评价接口

这是接口说明，**不是冻结协议或评价授权**。新 producer 为 `objective_joint_evaluation.py`，merger 为 `objective_joint_merge.py`。旧 evaluator、父八文件及正在运行的 F/O 训练包均不修改；此文不填写尚未完成的候选结果，也不启动推理。

## 提名规则与完整开发证据

模型顺序固定为 `M0, M0_fp32, native1`，再追加按 `F,O` 顺序排列的**全部**合格臂。同一候选对 `M0_fp32` 和 `native1` 两个主比较均须满足 `future_macro_gmo.delta_pp >= 0.5` 且 `ci95_pp[0] > 0`。区间下界不要求达到0.5。零个合格臂时拒绝 pilot/full；不得只选择其中看起来最强的一个。

提名来源始终是已完成的五模型 dev200 聚合；它的完整 summary、complete 及所有输出 SHA 固定，F/O 两臂结果始终披露。只让合格臂进入完整原生评价是事先确定的计算节约规则，不是删去较弱开发结果。筛选后的 full 区间也不包含历史选择不确定性；四主比较未作多重比较校正，full5119不是新盲测。

不可变 evaluation protocol 的 `nomination_rule` 必须精确为：

```json
{
  "metric": "future_macro_gmo",
  "comparators": ["M0_fp32", "native1"],
  "minimum_point_delta_pp": 0.5,
  "minimum_ci_lower_pp": 0.0,
  "ci_lower_strict": true,
  "include_all_qualifying_in_F_O_order": true
}
```

这是规则常量示例，不是候选结果或协议文件。producer 与 merger 分别按固定五模型开发结果重算合格名单。

## 不可变 evaluation protocol

- `schema`: `objective-joint-evaluation-protocol-v1`。
- `status`: `FROZEN_BEFORE_OBJECTIVE_JOINT_PILOT`。pilot 后不编辑此文件，full 用另外的授权文件接续。
- `parent_protocol_sha256`: `071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a`。
- `objective_train_protocol_sha256`: `275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d`。
- `candidate_names`: 依据上述规则由完整 dev summary 得到的非空有序 F/O 子集；现在不能根据某一臂的部分结果提前填写。
- `config_sha256`, `m0_sha256`, `numerical_policy`: 与父和 objective 训练协议完整一致。
- `runtime_source_sha256`: 与 objective 训练协议完整一致。原 loader、坐标、GT、原 evaluator 与 loss 源码均未变。
- `source_sha256`: 至少绑定新 producer、new merger、冻结 `joint_native_evaluation.py`、`merge_joint_evaluation.py`。旧 producer SHA 固定 `280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578`。其他父/训练/helper 源码继续通过已冻结训练协议逐文件核验。
- `development_summary`: `{directory, complete_sha256, summary_sha256}`，指向完成的 `objective_supervision_aggregate.py` 输出；`directory` 可绝对或相对本协议目录。
- `final_sources`: 键恰好为 `native1` 加 `candidate_names`，每项含 `{directory, files_sha256, final_head_state_sha256}`。C 的 `files_sha256` 来自开发 summary 的 `source_receipts.native1.reuse_audit.files_sha256`；F/O 来自各自 `source_receipts[arm].files_sha256`。最终 tensor SHA 全部来自 `source_receipts[arm].checkpoint.final_head_state_sha256`。它们是已完成源的实际 SHA，不是占位。
- `selections.pilot/full`: 每项 `{file,sha256}`。分别原 `joint_preflight_selection_v1.json` SHA `e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d` 和 `full_validation_selection_v1.json` SHA `60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391`。不重选 pilot，不跳过 full 样本。
- `resources.pilot/full`: 每项至少 `{max_seconds,max_allocated_gib,shard_count:2}`。它们是在 pilot 前冻结的资源**上限**。full 实际预算随后由真实 pilot 资源证据决定，并写入独立 authorization，不能超过这里的上限。
- `historical_validation_exposure=true`, `new_blind_test=false`, `multiple_comparison_adjustment=false`, `training_seeds_added=0`。

## 独立 full authorization

pilot 评价及合并通过后，根任务才可写新文件：

- `schema`: `objective-joint-full-authorization-v1`。
- `status`: `AUTHORIZED_AFTER_PILOT_PASS`。
- `evaluation_protocol_sha256`: 原不可变 evaluation protocol 的 SHA。
- `pilot_merge`: `{directory,complete_sha256,summary_sha256}`，绑定此次选定候选的真实工程合并 PASS。整个 pilot 文件 ledger 也会复核。
- `resources`: **恰好** `{max_seconds,max_allocated_gib,shard_count:2}`，均不大于 evaluation protocol 的 full 上限。这是 pilot 后确定的实际预算。

full 的 CLI 必须显式传 `--max-seconds` 与 `--max-allocated-gib`，数值与 authorization **精确相等**。两 shard 的 common contract 绑定同一 authorization SHA 与实际 resources，不能各自松动。authorization SHA 在开始捕获并于结束复核。`--deadline-unix` 可用作共同提前截止，不允许延长实际预算。

## 保留的推理与源码派生

新 producer 在独立 function globals 中编译旧 `Evaluation.run`；原模块 globals 不修改。MODELS/ARMS 只在私有 namespace 替换为冻结名单；`frozen_arms` 与 `pilot_reference` 分别绑定新最终来源核验和本轮 own-dev 参考，不改变原始数据的运算表达式。

源码树恰好七处变换：

1. common contract 构造改为新双训练协议、提名、资源与最终权重身份。
2. 所有重放模型的 memory mode 固定 `native1`。
3. sample schema 改为 `objective-joint-sample-v1`。
4. pilot flag 改为 `all_models_5_horizons_full_GT_confusion_equal`，不再写死5模型。
5. CUDA device 设置后插入 allocator cap。
6. wrapper 创建前插入实载 head SHA 与 migration 核验。
7. shard complete 加入 `pilot_reference_sources_sha256`，full 显式为 null。

计数必须各1；逐项逆变换后 AST 必须恢复完整原 run，才编译执行。当前原 AST SHA 为 `99b2b7021deb8587e937f8253a7406e0f54697ace844233df7e673c30a94af7c`，派生 AST 为 `3e8131656cdabd04d8646f7b94b04cf6159b8f26ef1e7764195bac45d8a573a7`；receipt 同时记录 source SHA、计数与 `inverse_ast_restores_original_run=true`。这约束原 loader/capture/replay/GT/evaluator 表达式，不能代替真实 CUDA pilot。

四个或五个模型仍共享一次 M0 原生图像/雷达观测编码。M0 全原生 TF32=true；M0_fp32/C/合格候选仅未来 head matmul=false，cuDNN=true。候选只加载其最终 head 参数，不安装任何 F/O loss adapter、frame adapter或新阈值。实际 final payload 的 update/pass/orders、shape/dtype、optimizer group 和文件/tensor SHA 均先在 CPU 核验；GPU strict load 后再次逐 tensor 相等及最终 digest 核验。评价结束 head SHA 必须不变。

## Producer CLI

每个 shard 都是新目录、无 retry/resume；共享新的 evaluation root 中的 common contract 和 samples 目录，由两个新 shard 目录排他拥有。外层负责核两卡空闲、明确 GPU 分配及受控进程组，不终止其他作业。

```bash
python objective_joint_evaluation.py --evaluation-protocol EVAL_PROTOCOL \
  --objective-protocol OBJECTIVE_TRAIN_PROTOCOL --protocol PARENT_PROTOCOL \
  --config CONFIG --checkpoint M0 --repo REPO --out NEW_PILOT_ROOT \
  --mode pilot --shard-index 0 --device cuda:0 --parity-cache DEV200_CACHE
```

另一个受控进程使用相同 root/protocol，`--shard-index 1`，另一张已核空闲 GPU。pilot 恰好原两个 validation anchors，分别 index2036/2998；每 shard一条，完整输入/GT/M0 5时域×3层 logits 与旧 cache 逐位相等，各模型5时域 confusion 与自身固定最终 dev 记录精确相等。pilot 不计算效果分数或提升结论。

```bash
python objective_joint_evaluation.py --evaluation-protocol EVAL_PROTOCOL \
  --objective-protocol OBJECTIVE_TRAIN_PROTOCOL --protocol PARENT_PROTOCOL \
  --config CONFIG --checkpoint M0 --repo REPO --out NEW_FULL_ROOT \
  --mode full --shard-index 0 --device cuda:0 \
  --full-authorization FULL_AUTHORIZATION \
  --max-seconds AUTHORIZED_SECONDS --max-allocated-gib AUTHORIZED_GIB
```

full 不允许 `--parity-cache`；它读取全部5119原生输入。第二 shard同授权实际资源。`--selection` 可省略使用协议路径；如显式传入，producer 要求等于协议绑定路径。merger 可对本地镜像提供相同 SHA 的 selection，不改协议文本。

## 输出给 merger 的精确约定

`contract.json.schema=objective-joint-evaluation-contract-v1`。核心字段：`mode`, `models`, `candidate_names`, `evaluation_protocol_sha256`, `objective_train_protocol_sha256`, `parent_protocol_sha256`, `selection_sha256`, `script_sha256`, `config_sha256`, `source_sha256`, `parent_source_sha256`, `objective_source_sha256`, `runtime_source_sha256`, `m0_sha256`, `final_checkpoints_sha256`, `final_head_state_sha256`, `development_summary_sha256`, `nomination_evidence`, `shard_count`, `selected_samples`, `numerical_policy`, `horizons`, `optimizer_steps=0`, `sample_partition`, `historical_validation_exposure`, `full_authorization_sha256`, `pilot_gate`, `data_recipe_receipt`, `resources`, `allocator_cap_bytes`。

`final_*` 两个映射只包括 C 和提名候选；`models`/实际 `loaded_models.head_state_sha256` 包括 M0/M0_fp32 与它们。`loaded_models.json` 沿原 schema，含实际 migrations、head digest、native M0 provenance、torch/CUDA/GPU版本、`training=false`、`optimizer_created=false`。

shard manifest 继承完整 common contract，并含所分配的真实 `selected_records`、`arm_source_receipts`、PID/PGID、实际时限。每臂 source receipt 为 `{directory,files_sha256,manifest,final_head_state_sha256,checkpoint_audit}`；checkpoint_audit 可与已完成开发 summary 的同名模型 checkpoint 审计完整比较。

每样本 `records.json` 保留原 identity/input/GT/observed-BEV/radar/full-logits 摘要、7帧 class counts、各模型5时域原 confusion、时长和峰值；只更换 sample schema 与动态 pilot flag。旧 index/sample complete 的文件 SHA 链保留。

shard complete 保留 `status=COMPLETE_JOINT_NATIVE_EVALUATION_SHARD`，完整绑定 index/manifest/loaded_models/common，并新增 `pilot_reference_sources_sha256`；full 为 null。结束仍执行原 source 重核和所有实际 head digest 不变门。不输出任何训练 checkpoint、大 GT/logits 缓存或候选性能摘要。

pilot 合并 summary 须使用 `objective-joint-pilot-merge-v1`、`PILOT_ENGINEERING_PARITY_RESOURCE_PASS`，顶层 `candidate_names`、`model_names`、2 samples/2 scenes，`performance_metrics_computed=false`、`performance_success_claim=false`、`merger_sha256`；sources 包含三协议、pilot selection及两个最终权重映射。complete 沿 `status=COMPLETE,mode=pilot,samples=2,summary_sha256,files_sha256`。这些是 full_gate 要读取的实际字段。

merger 的 full 数学对实际4/5模型显式参数化，不改其他模块模型常量，不调用 dev200 加载入口；每个候选保留对 M0_fp32/C 两主比较，另列 C−M0_fp32 与 M0_fp32−M0。所有150场景自然 anchor 数一起配对 bootstrap，原四未来 macro GMO、pooled副指标和 binary mIoU 清楚区分。只核验完整结果，不自动判科研目标完成。
