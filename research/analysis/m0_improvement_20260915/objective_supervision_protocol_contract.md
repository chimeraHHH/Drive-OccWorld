# F / O 固定训练协议接口

此文件是接口说明，不是冻结协议，也不包含待测资源的替代数值。根任务应在真实四更新工程预检 PASS 后填写并冻结 `objective_supervision_protocol_v1.json`。本包不自动生成或更新协议。

## 协议必须字段

- `schema`: `objective-supervision-training-protocol-v1`
- `status`: `FROZEN_BEFORE_F_O_TRAINING`
- `arms`: `["F", "O"]`
- `parent_protocol_sha256`: `071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a`
- `training`, `numerical_policy`, `config_sha256`, `m0_sha256`, `selection_sha256`：与父 `protocol_v2.json` 完整相等。512 train anchors、200 development anchors；seed11、4 passes、accumulate4、512 updates、AdamW LR1e−5、weight decay .01、clip35、原 warmup50/cosine 全部保持。
- `cache_index_sha256`: `{ "train": <SHA>, "development": <SHA> }`，复用原已冻结两缓存。
- `control_files_sha256`：复用已完成 `native1` 的七文件绑定，字段为 `manifest.json`, `training_complete.json`, `complete.json`, `latest.pth`, `training.jsonl`, `development_records.jsonl`, `frozen_native_fp32_records.jsonl`。
- `source_sha256`：至少绑定 `objective_supervision_train.py`, `objective_supervision_adapters.py`, `objective_supervision_preflight.py`, `objective_supervision_aggregate.py`, `run_objective_supervision_campaign.py`, `full_resolution_supervision_preflight.py`, `frame_train.py`, `frame_consistent_adapter.py`, `frame_aggregate.py`。父八文件仍按父协议独立验 SHA。后面三个旧 frame 文件只复用已审控制核验、原训练 AST 的规范化与 checkpoint/log 核验；**不安装任何坐标干预**。
- `runtime_source_sha256`：至少完整包含父原生源码绑定；包括全部实际使用的原 loss 源码，不能修改 sem/geo/Lovász。
- `engineering_evidence`：仅两个键 `full_resolution_vjp`, `objective_trainmode`。每项含 `directory`, `complete_sha256`, `manifest_sha256`, `summary_sha256`, `source`。分别引用真实 `full_resolution_supervision_preflight.py` 与 `objective_supervision_preflight.py` 的 PASS 目录。路径可为服务器绝对路径或相对新协议目录的路径。
- `resource_policy.arms.F` 与 `.O`：各自 `{ "max_seconds": <真实预检后冻结的正整数>, "max_allocated_gib": <冻结上限> }`。每臂计时覆盖控制核验、建模、训练与最终 dev200 评价。allocator 限额不包含 PyTorch 外显存。
- `resource_policy.aggregate_max_seconds`：末端 CPU 配对汇总时限，正整数。
- `evaluation.primary_contrasts`: `[["F","M0_fp32"],["F","native1"],["O","M0_fp32"],["O","native1"]]`。
- `evaluation.minimum_meaningful_effect_pp`：根任务登记的阈值（单位 pp）；未登记可为 null，不自动判断目标完成。
- `evaluation.historical_validation_exposure`: `true`；开发集已被历史实验使用，当前不是新盲测。
- `evaluation.multiple_comparison_adjustment`: `false`。四个主比较共享同一场景重采样；其 95% CI 为逐项区间，不声称 family-wise 覆盖。

## 实验和工程证据边界

F 从相同 M0/native1 开始，仅移除原 coarse GT 并把原 loss 插值尺寸改为完整 512×512×40，三层、五时域和原 12 项损失保持。O 从相同 M0/native1 开始，保留 coarse GT 和原 12 项计算，仅返回 CE[1,5] 与 Lovász 六项进入优化目标；sem/geo 仍有前向成本，detached 标量仅用于有限值审计。F 不移除 scaling，O 不改变监督分辨率，二者不叠加。

工程证据一要求 `PASS_NO_OPTIMIZER_PREFLIGHT`、零更新、完整 GT 实际核验和配对预测/原 evaluator hist 精确一致。工程证据二要求 `PASS_OBJECTIVE_SUPERVISION_ENGINEERING_PREFLIGHT`，F/O 各 4 次真实 AdamW 更新、16 样本次、两条固定训练 anchor，native_fp32/F/O 初始预测与原完整 GT evaluator 精确一致，完整 head 梯度有限且参数实际变化。其 constant LR1e−5 用于工程检查；正式训练仍是父 warmup/cosine。探针权重不被训练器读取。

两臂不重复零更新 dev200。原 `native1` manifest 的 `frozen_precision_reference` 与 JSONL 实际 SHA 完整复核后，在新 manifest 以 `reused_frozen_precision_reference` 记录。两个训练臂都从 M0 checkpoint 初始化，绝不从 native1 最终参数或探针参数初始化。模型构造/适配结束后，在创建新 AdamW 前重置 seed11。

## CLI 与输出

独立单臂（CUDA_VISIBLE_DEVICES 由外层明确指定）：

```bash
python objective_supervision_train.py --objective-protocol OBJECTIVE_PROTOCOL \
  --protocol PARENT_PROTOCOL --repo REPO --config CONFIG --checkpoint M0 \
  --train-cache TRAIN_CACHE --dev-cache DEV_CACHE --control-run NATIVE1_RUN \
  --arm F --out NEW_F_RUN --max-seconds FROZEN_F_CAP
```

双 GPU 自动完成各自训练/固定最终评价，均成功后做 CPU 汇总：

`--gpus` 只接受物理索引 `0,1` 或 `1,0`，按 F/O 顺序分配。启动前只读查询两卡 UUID 与 compute PID；两卡均空闲才启动，记录 `manifest.gpu_before`，不终止其他作业。训练子进程继承外层 `job_runner` 的进程组，避免外层停止后遗留独立训练进程。

```bash
python run_objective_supervision_campaign.py --root NEW_CAMPAIGN \
  --objective-protocol OBJECTIVE_PROTOCOL --protocol PARENT_PROTOCOL \
  --repo REPO --config CONFIG --checkpoint M0 --train-cache TRAIN_CACHE \
  --dev-cache DEV_CACHE --control-run NATIVE1_RUN --gpus F_GPU,O_GPU
```

每臂输出 `runs/{F,O}/manifest.json`, `control_reuse.json`, `training.jsonl`, `progress.json`, `latest.pth`, `training_complete.json`, `development_records.jsonl`, `complete.json`。原训练循环每 pass 保存一次 `latest.pth`；仅最终 pass4/update512 可进入汇总，无 best 选择或自动 resume。失败时允许新目录内保存 `interrupted.pth`，明确 `INTERRUPTED_NOT_FINAL_NOT_RESUMABLE`，不能进入最终评价或自动继续。

新 manifest 除原 scope/source/cache 字段外，包含 `objective_adapter`, `parent_protocol_sha256`, `parent_source_sha256`, `engineering_evidence`, `control_reuse`, `training_block`, `reused_frozen_precision_reference`。`migration.mode` 保持 `native1`。F 的训练 `loss_mean` 是完整分辨率 12 项之和；O 是粗分辨率 6 项之和，不可据绝对值横向比较目标优劣。

`complete.json` 保留原 `status=TRAINED_AND_DEVELOPMENT_EVALUATED`, `updates=512`, `examples=2048`, `checkpoint_sha256`, `evaluation{samples,sha256}`；追加 `objective_protocol_sha256`, `parent_protocol_sha256`, `manifest_sha256`, `control_reuse_sha256`, `training_log_sha256`, `final_head_state_sha256`, `training_seed=11`, `initial_development_repeated=false`。

配对汇总 CLI：

```bash
python objective_supervision_aggregate.py --objective-protocol OBJECTIVE_PROTOCOL \
  --protocol PARENT_PROTOCOL --cache DEV_CACHE --control-run NATIVE1_RUN \
  --runs-root NEW_CAMPAIGN/runs --out NEW_SUMMARY
```

汇总只使用五个已固定模型 M0、M0_fp32、native1、F、O。主指标是逐未来时域先合并全部 anchors 的 confusion，再将四个 foreground class1 IoU 算术平均；不是 binary mIoU，也不是把四未来 confusion 合在一起的 pooled IoU。全部 100 场景、每场景 2 anchors 保持配对，10,000 次 seed11 场景 bootstrap。保留五时域 confusion、GMO/binary mIoU、FP/FN、四主对比与 C−M0_fp32、M0_fp32−M0。单训练种子与历史开发集曝光必须披露，不自动触发 full validation 或新的训练。
