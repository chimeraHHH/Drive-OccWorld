# R / R0 特征残差预检交接

训练器已冻结：`train_future_feature_residual_v1.py`，SHA `2226681ee8b6b0b7ffe40ee40540947619217f2f3cdd83d034f101ba6706cbde`。核心为 `future_feature_residual_v1.py`，SHA `e427b74698ad4537f135f6dab03df9787f8661abac666164e21e48997caa541d`。独立审查无阻断，未据此声称原生 GPU 训练通过。

`future_feature_residual_preflight_protocol_v1.json` 当前为 **REVIEW_REQUIRED**；候选 SHA `7b1b6f4bc73942f477e8855559e0907fdb2e602ee5bc6ab9d927ef22206c5b26`。根代理冻结状态后必须使用新实际 SHA，不把候选 SHA 当作最终协议 SHA。预检仅 600 秒 / 32 GiB，建议外层 runner 660 秒；`resources.train=null`，无法用该协议启动正式训练。

完整结构化实际命令、21 文件 package 清单及其 SHA 位于 `future_feature_residual_preflight_preparation_v1.json` 的 `command` / `package_files`。其中 17 个科学 Python 源通过协议绑定，另有新预检协议、原 J/D 协议、原生 runtime source contract 和 job runner。所有路径来自现有真实任务收据；预检本身未派发。物理 GPU 由根代理选择空闲卡，脚本使用该单卡环境内的逻辑 `cuda:0`；使用系统驱动库，禁止旧 driver override。

执行轮廓（准确服务器 argv 以准备 JSON 为准）：

```text
python -B train_future_feature_residual_v1.py
  --protocol future_feature_residual_preflight_protocol_v1.json
  --connected-protocol connected_motion_protocol_v2.json
  --training-run ACTUAL_COMPLETED_CONNECTED_MOTION_TRAIN_V2
  --config S0 --checkpoint M0 --o-checkpoint O
  --repo NATIVE_REPO --runtime-contract runtime_source_contract.json
  --train-cache NATIVE_TRAIN512 --out NEW_PREFLIGHT_DIRECTORY
  --device cuda:0 --preflight --max-seconds 600 --max-allocated-gib 32
```

R / R0 各 4 次 accum4 AdamW 更新，同 seed11 / 初值 / 原训练排列 / LR；只训练新 41,137 参数。O 和实际 final D motion 固定，旧 D gate 仅严格载入校验后丢弃。预检不传开发集、raw box 或 sparse motion labels，不保存任何权重。原固定顺序前两个不同场景样本核零残差初始五时域×三层 logits 与原 O 位级一致、native histogram 相同。后续核末层、projection 和 trunk 曾获得非零任务梯度、AdamW 状态有限、O / D 状态不变。所有步保留 t0 / 中间层 / background logits 原样；允许残差修正 W=0 区域，不复用局部影响域门。

输出顶层 `manifest.json`、`loaded_models.json`、`summary.json`、`complete.json`，逐臂 `runs/{R,R0}/manifest.json`、`training.jsonl`、`complete.json`。成功状态为 `PASS_FUTURE_FEATURE_RESIDUAL_PREFLIGHT`，updates=4、examples=16、evaluated_samples=0、final_checkpoints={}；失败写 `failed.json`，不重试。资源报告包含初始化、16 个微步及 4 个双臂更新耗时、真实 AdamW 显存峰值；冷微步包含额外初始 native parity 开销。

真实独立 CPU 工程任务 `future_feature_residual_cpu_v1` 已 EXITED_ZERO：7 tests / 0.555 秒，外层 2.2244092 秒。实际 launch / state / output SHA 已绑定预检协议的 `engineering.core_cpu_tests`。这仅验证核心解析行为，不等同真实原生 GPU 4-update 通过，更不是科研效果。

正式计划仍为各 512 更新 / 2048 暴露 / 同固定开发200，原 O 的 CE[1,5]+Lovasz 六项，原 12 项公式仍照算；不改原 GT、分辨率、时域或 evaluator。正式需另冻结资源与真实预检 complete SHA，重新 seed11 初始化，不能复用预检权重。开发结果输出 `hist_by_arm` 与 `metrics_by_arm`，包括 O / R / R0，原 O histogram、CPU/native histogram、t0 全部严格核对。D motion 未改变，因此不报告新物理 EPE。
