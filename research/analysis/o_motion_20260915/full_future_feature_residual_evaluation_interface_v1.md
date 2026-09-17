# O/R/R0 原生 full 流式评价接口

代码已准备，尚未运行 R/R0 full 或新的 GPU pilot；没有读取尚未完成的候选结果。实际执行、晋级与预算由根任务决定。

- producer：`full_future_feature_residual_evaluation_v1.py`，SHA `da0480e78351b1be9442ef1e3d65928ccb5dce1630d467f299ea8f158c3d91f2`。
- 严格 loader：`load_future_feature_residual_v1.py`，SHA `b97d660db64b37f2e1196896df35128d33e4f331757a46004197bbe12e132e86`。
- 原 native stream：`3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef`。
- 固定 R/R0 训练协议：`future_feature_residual_protocol_v1.json`，SHA `45048560948fae975945685e5151db158933b81b94b53b543551213b29a79a56`。

每个样本只走一次原始观测 pipeline；`stream.fetch(ordinal)` 已完成原生 TF32-on 观测与 TF32-off O 重放后，调用 `capture_O_terminal(stream.model, frame['sample'], native)` 再重放 future head 取得末层特征，不重跑观测 backbone。新增重放输出必须与 stream 已认证的 O 五时域三层 logits 逐位相同，t0 特征必须等于 measured BEV；输入、GT、特征、D 位移和模型状态保持不变。

R 与 R0 使用各自严格 final512 参数及同一 O 上下文；R 传入固定 D 位移，R0 传入零位移，复用原核心 `compose_prediction`。模型推理不读取 raw box 内容；原 dataset 提前加载的完整 GT 只交 evaluator。全部三模型保留原 fine histogram、八个 GT 正类运动组、四种占据变化及全局 FP/FN；不重复物理 sparse EPE，不输出“原生 O flow”结论。

## CLI

```text
python full_future_feature_residual_evaluation_v1.py
  --mode pilot|full
  --protocol future_feature_residual_protocol_v1.json
  --protocol-sha256 45048560948fae975945685e5151db158933b81b94b53b543551213b29a79a56
  --connected-protocol connected_motion_protocol_v2.json
  --connected-training-run <原 completed connected_motion_train_v2>
  --training-run <完整 R/R0 fixed512/dev200 run>
  --training-complete-sha256 <实际完成文件 SHA，不能使用预检回执>
  --stream-sha256 3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef
  --stream-engineering <已完成 native_full_o_stream_preflight_v1>
  --config <原 S0> --checkpoint <原 M0> --o-checkpoint <原 O> --repo <原 repo>
  --runtime-contract runtime_source_contract.json
  --selection full_validation_selection_v1.json
  --reference-dir <原 objective_joint_full_v2/summary_v1>
  --raw-labels <完整 full_motion_targets_v1>
  --out <新目录> --device cuda:0
  --shard-index 0 --shard-count 1|2
  --max-seconds <根任务明确预算> --max-allocated-gib 32
```

pilot 另外要求 `--pilot-selection joint_preflight_selection_v1.json`，固定两个 official ordinals 2036、2998，单分片且不超过 600 秒；full 单片或两片连续覆盖 `[5119*i//n, 5119*(i+1)//n)`。失败后停止，无自动重试、替换样本、阈值选择或模型晋级。

24 个 Python 源的精确清单在 `receipts/full_future_feature_residual_evaluation_v1_cpu_checks.json.sources_sha256`；打包 producer 之外的全部 23 件，包含旧 `full_connected_motion_evaluation_v1.py`，其用途仅为冻结的 schedule 与 native-stream 工程回执校验。静态非 Python 资产为上述两个训练协议、runtime contract、full selection，pilot 再加 pilot selection；实际 checkpoints、completed run 全套文件、raw labels、原 full reference 和 stream engineering 都使用已有真实目录，不生成替代数据。

## 合并器契约

纯 CPU 接口保持：

```text
imports(protocol) -> (contract, stream, modules, sources)
stream_engineering(root, stream) -> 原 c35…完整工程文件绑定
training_metadata(root, complete_sha, protocol, trainer, selection)
  -> (root, done, manifest, previous_by_sample_token)
labels_contract(rawroot, selection) -> (root, descriptors)
```

每个 `OUT/shard_XX` 产出 `manifest.json`、`loaded_models.json`、`records.jsonl`、`summary.json`，完成文件对四件逐一 SHA 绑定；无 physical objects 文件。schema 为 `full-future-feature-residual-evaluation-v1`，full 状态 `COMPLETE_FULL_FUTURE_FEATURE_RESIDUAL_SHARD`，pilot 状态 `PASS_FULL_FUTURE_FEATURE_RESIDUAL_PILOT`。

`loaded_models` 含 `native_O`、`O_full_helper_list_shape_state_sha256`、严格 D loader 完整 `D` receipt、严格 R/R0 loader 完整 `arms` receipts。records 保留 `native_stream_audit`、`hist_by_arm`、`metrics_by_arm`、官方 identity/order、`is_training_development_intersection`、`training_final_hist_exact`；训练 dev200 交集三模型五时域历史混淆必须精确相同。新增 `captured_O_replay_all_layer_logits_exact`、`captured_t0_features_equal_measured`、`captured_context_and_D_displacement_unchanged` 及特征/位移小哈希。

complete 保留原非物理 source/protocol/training/stream/selection/shard/ordinal/parity 字段；summary 只提供计数与本分片范围，不做 bootstrap 或结果选择，根任务合并时用已冻结 v3 统计一次。

## 实际 CPU 验收边界

24 源 SHA、Python 3.10 AST、CLI help、原 5119/150 身份与单片/双片覆盖、原 full O reference、完整 raw metadata、两个真实 pilot raw payload 以及已完成 c35…stream 回执均已实查通过。验收文件 SHA `ed0d4a9b0194c962deb4152bde020e78e87c716cb54b75b9d5c4be65dff696fd`。

没有导入 Torch、加载新的 final checkpoint、读取 R/R0 成绩或执行新的 GPU 推理；真实新模型 tensor 加载、捕获重放字节门与完整推理资源仍需执行时验证。
