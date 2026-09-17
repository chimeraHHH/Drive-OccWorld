# C/O 未来特征与共享读出：只读 2×2 诊断协议草案

状态：方案与执行脚本准备；本文件不是已冻结授权，不含虚构结果。正式协议由 root 绑定实际来源后另写 JSON。当前完整验证任务完成并退出之前，不启动本诊断。

## 科学范围

固定 C＝已完成 native1、O＝已完成 CE[1,5]+Lovász 目标的最终 512-update checkpoint，训练 seed 都为 11。固定原 development 200 anchors / 100 scenes，每场景两 anchors。CC、CO、OC、OO 的第一位是 future features 来源，第二位是完整三层 `bev_pred_head` 读出来源。

原代码的未来状态递推只用特征，不回流 t0 语义 logits；关闭规划等分支时，readout 只在五时域特征生成之后调用。诊断从 C/O 各执行一次原生 native1 回放，在三个 `nn.Sequential` readout 分支安装只读 `forward_pre_hook`，复制三个 `[5,1,40000,256]` 输入张量。移除 hooks 后，把三层输入恢复成 `[5,3,1,40000,256]`，直接调用原生 `forward_head` 完成四组合。没有修改模型参数、替换模块、改 logits/坐标、拟合 stitching 或安装训练损失适配器。

每个 readout 来源覆盖整个 `bev_pred_head` 的全部参数和 buffers，不仅最后 Linear。每个分支为 Linear、LayerNorm、ReLU、Linear。C/O 原生 future head 均为 13,274,016 参数，全部只读；两模型都从相同原 M0 初始化再严格加载各自最终完整 future head。

## 每条样本的强验收门

1. CC 与 OO 重放 logits 必须逐位复现同次原生 C/O 输出，覆盖全部五时域、三个层、全部 coarse-output voxel；字节比较包含 signed zero。历史开发记录没有全层 logits 缓存，因此不声称复现了历史全层 logits。
2. CC 与 OO 的每样本五时域完整 GT 混淆必须精确复现已冻结 `development_records.jsonl`。所有组合用原 evaluator、完整 `[1,7,512,512,40]` GT、相同身份、未来 ego/action 条件、坐标、argmax；不改阈值或插值。
3. 三层 t0 features 必须 C=O。所有三层、所有 voxel 的 t0 logits 必须 OC=CC、CO=OO；不能仅比较 IoU。未来 features 是否不同不是验收门。
4. 输入完整 typed tree、GT tensor、捕获的 features 和原始 logits 在使用前后保持不变。原生 replay 深拷贝输入并统一设 `valid_frames=[]`，输入标签分离。此任务四组合使用同一缓存，不采用 inactive metadata 排除例外。
5. 原生单槽、`soft_weight=False`、planning/flow/M3/semantic-normalization 关闭；所有模块 eval、所有参数 frozen、所有参数 grad 为 None。matmul TF32=False、cuDNN TF32=True、benchmark=False。
6. 源文件、原生方法 AST、冻结 native1 no-GT 派生方法 provenance、参数/缓冲区实际完整 SHA、来源 checkpoint、开发记录、输入/GT payload SHA 全绑定。加载与退出都核模型状态；退出重核代码与原 C/O 文件。任何门失败终止，不降低门槛、不自动重试。

## 两阶段与资源

`pilot` 固定取 dev 顺序中最前两个不同场景各第一 anchor（当前 ordinals 0、2）。不按成绩或 GT 选择，不输出性能成功主张。它检验真实 hook、精度、身份、完整 GT/输出一致性与资源开销。

`development` 必须读同一冻结诊断协议的 PASS pilot，核完整产物哈希、两模型实际状态及两 pilot 身份。需要 pilot 后单独冻结的实际预算授权。先前完整验证 job 的完成/退出门由独立服务器 wrapper 负责；脚本不主动查看部分完整验证成绩。

root 提议固定天花板：pilot 300 s / 32 GiB，development 最多 1500 s / 32 GiB，总计最多 1800 s。实际 development cap 按 pilot `pre_sample_initialization_seconds` 与 `max_sample_seconds` 测量：`max(300, ceil(1.75 * (init + 200 * max_sample) + 60))`。其中 init 覆盖到首 anchor 前的来源审计、CPU checkpoint 校验、导入、建模和日志初始化；`model_initialization_seconds` 另单列纯建模时间。若超过 1500 s、32 GiB 或 pilot 实际时间加 development cap 超过 1800 s，拒绝授权，不能静默截断或扩容。两样本是工程观测，不能保证所有场景同样快；末端重哈希和 CPU bootstrap 也有固定成本，预算公式只是估算，正式 deadline 和 allocator cap 仍强制执行。

不构建 optimizer，不 backward，不训练、不新增种子、不挑 checkpoint 或阈值、不把混合组合晋升为方法。输入完整数据在 GPU 上短暂存在；只持久化小型 confusion/hash/identity/resource JSON(L)，不持久化 features 或 logits。静态检查不能证明 GPU 显存/时长；实际 pilot 尚待授权与运行。

## 输出与解释

每个 new output 目录只含 `manifest.json`、`records.jsonl`、`summary.json`、`report.md`、`complete.json`；失败另留 `failed.json`，不写 COMPLETE。`complete.files_sha256` 正好绑定前四件。

development 报告五时域 GMO，以及四未来时域分别池化 class1 混淆算 IoU、再算术平均的 future macro GMO；另报 future pooled GMO 与 binary mIoU，不能混称。保存每组合整数 confusion。

令 J 为同一指标，报告所有以下量：

- 总差：J(OO)−J(CC)。
- 路径一：读出效应（C features）J(CO)−J(CC)，再 features 效应（O readout）J(OO)−J(CO)。
- 路径二：features 效应（C readout）J(OC)−J(CC)，再读出效应（O features）J(OO)−J(OC)。
- 交互：J(OO)−J(OC)−J(CO)+J(CC)。

两条路径代数上都等于总差；交互是所选非线性指标尺度上的差中差。潜空间重参数化和共同适应可能影响跨读出兼容性，不能把 cross 降低说成 transition 信息损失，不能唯一分配贡献百分比或作统计因果识别。无需强制 cross 差为正。

development 使用 10,000 次固定 seed11 场景配对 bootstrap，同一场景两个 anchors 一起抽样、四组合共用抽样。区间未作多重比较校正，只体现固定模型场景不确定性，不含训练种子、历史模型选择或跨模型识别不确定性。所有量为 posthoc，无新预注册主性能检验或晋级门。始终保留此前完整五模型开发披露的哈希，不选择性隐藏 F。

## 待 root 填真实值并冻结的 JSON 字段

诊断协议：

| 字段 | 固定语义 |
|---|---|
| `schema`, `status` | `readout-transition-diagnostic-protocol-v1`, `FROZEN` |
| `parent_protocol_sha256`, `objective_protocol_sha256` | 实际父 v2 与 F/O 训练协议 SHA |
| `source_sha256` | 新脚本、所有父/目标 frozen helpers、wrapper、`objective_full_dependency.py` 的真实 SHA；文件放同包 |
| `config_sha256`, `m0_sha256` | 原生 M0 来源 |
| `cache_index_sha256` | 原 dev200 index SHA（单字符串） |
| `development_summary` | `{summary_sha256, complete_sha256}`；原完整五模型 summary/complete |
| `model_sources.C`, `.O` | `{files_sha256, final_head_state_sha256}`；分别取原 summary `native1.reuse_audit.files_sha256`、`O.files_sha256`，及各自 `checkpoint.final_head_state_sha256` |
| `combinations` | `['CC','CO','OC','OO']` |
| `pilot_selection` | `first_anchor_of_first_two_distinct_scenes_in_frozen_development_order` |
| `resources` | `{total_max_seconds:1800,max_allocated_gib:32,pilot:{max_seconds:300,max_allocated_gib:32},development_ceiling:{max_seconds:1500,max_allocated_gib:32}}` |
| scope booleans | `posthoc:true, training:false, model_selection:false, historical_validation_exposure:true` |

development 授权：`schema='readout-transition-development-authorization-v1'`、`status='FROZEN'`、`diagnostic_protocol_sha256`、`pilot_complete_sha256`、`pilot_summary_sha256`、`resources={max_seconds,max_allocated_gib}`。实际 CLI 的两个资源数值必须与授权完全一致、处于固定 ceiling 内，不能只填一个未来未知 placeholder 进行执行。

producer complete：`schema='readout-transition-complete-v1'`；pilot status=`PASS_READOUT_TRANSITION_ENGINEERING`，development status=`COMPLETE_READOUT_TRANSITION_DIAGNOSTIC`；还有 mode、diagnostic_protocol_sha256、samples、scenes、optimizer_updates=0、seconds、files_sha256。

summary：`schema='readout-transition-summary-v1'`；mode、顶层及 `sources.diagnostic_protocol_sha256`、selected_identities、loaded_models、all_engineering_gates_passed、resources；资源含 `elapsed_seconds`、`pre_sample_initialization_seconds`、`model_initialization_seconds`、`max_sample_seconds`、`anchor_seconds`、`peak_allocated_bytes` 和完整 allocated/reserved 峰值。development 才有 models/effects/bootstrap。

## CLI

```bash
python readout_transition_diagnostic.py --mode pilot \
  --repo "$REPO" --config "$CONFIG" --checkpoint "$M0_CHECKPOINT" \
  --protocol protocol_v2.json --objective-protocol objective_supervision_protocol_v1.json \
  --diagnostic-protocol "$FROZEN_DIAGNOSTIC_PROTOCOL" \
  --cache "$DEV_CACHE" --native-runs-root "$MEMORY_RUNS" --runs-root "$OBJECTIVE_RUNS" \
  --development-summary "$OBJECTIVE_SUMMARY/summary.json" \
  --out "$NEW_PILOT_OUTPUT" --device cuda:0 --max-seconds 300 --max-allocated-gib 32
```

development 同公共参数，改 `--mode development --out "$NEW_DEVELOPMENT_OUTPUT"`，增加 `--pilot "$COMPLETE_PILOT_OUTPUT" --authorization "$FROZEN_ACTUAL_AUTHORIZATION"`，并把 `--max-seconds` 设为该授权实际值。脚本不会自行写出授权、启动下一阶段或改冻结源。
