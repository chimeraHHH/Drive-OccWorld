# Readout / transition v2b 打包独立审查

**结论：实际候选清单通过本次打包审查，可由根代理以新 job `readout_transition_v2b`、新输出 `campaign_readout_transition_v2b` 执行原 pilot→dev200 序列。** 这是补齐静态资产的全新任务；既有 v1／v2 失败不得覆盖，源码、冻结协议、科学设置和预算均不变。本审查没有派发、修改源码、运行 GPU 或重新评价 full。

审查依据为[实际完整打包计划](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/readout_transition_v2b_package_plan.json)，最终 SHA-256：`c1b462017d3e1588cde51677d3789e8f80c4bcb7f434dfab8de3d4430a60dbc4`。其 26 份文件在本地均存在，实际 SHA 全部等于清单。

**真实错误与此次修复边界。** [v2 pilot 回溯](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v2/pilot.log:6) 指向 producer 第 181 行 → `aggregate_memory.load_cache` 第 146–147 行：`Path(protocol_path).with_name('selection_v1.json')`，该 sibling 文件不在 v2 包内，故 SHA 读取时报 `FileNotFoundError`。失败记录为 pilot 1.486047 秒、optimizer updates 0；外层 job 2.129083 秒、`FAILED`／return code 1。尚未构建诊断模型或执行两 anchor 推理，因此不能解读为 parity 失败或科学阴性结果；也不是先前两处 `Path` 修复失效。

以原 v1 真实 `launch.json` 的 26 文件为基准，分别替换 producer 名称／SHA、wrapper 名称／SHA及诊断协议名称／SHA，所得字典与 v2b 清单精确相同。对失败 v2 的真实 25 文件逐项比较，全部原值保留，**唯一新增是 `selection_v1.json`**；没有额外文件或非预期源变化。核心绑定仍为：

| 文件 | 实际 SHA-256 |
|---|---|
| `readout_transition_diagnostic_v2.py` | `489fccce38cb418b02d6de18164b70c192b8c76d8e9654c4d8293ce9e7bf0b06` |
| `run_readout_transition_campaign_v2.py` | `e3ec99f45c7e91355cf4d2eef7ace6a92ef94036d0915f61bf0fc23b3f6a7756` |
| 已冻结 `readout_transition_diagnostic_protocol_v2.json` | `deba0647052014de50a00e373582806e6dfa814c16f9c4eaa524d93b66bf77cf` |
| `selection_v1.json` | `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d` |

完整 26 文件由已冻结诊断协议的全部 22 份 Python 来源，加三协议和 selection 组成，实审清单如下；每项完整哈希以以上计划为准：

```text
aggregate_memory.py
frame_aggregate.py
frame_consistent_adapter.py
frame_train.py
full_resolution_supervision_preflight.py
job_runner.py
joint_native_evaluation.py
memory_experiment.py
native_state_cache.py
objective_full_dependency.py
objective_supervision_adapters.py
objective_supervision_aggregate.py
objective_supervision_preflight.py
objective_supervision_protocol_v1.json
objective_supervision_train.py
observation_memory.py
posthoc_train_diagnostic.py
protocol_v2.json
readout_transition_diagnostic_protocol_v2.json
readout_transition_diagnostic_v2.py
run_campaign_card.py
run_objective_supervision_campaign.py
run_readout_transition_campaign_v2.py
selection_v1.json
test_observation_memory.py
wait_for_native_cache.py
```

**非 Python 依赖闭合。** 本次核对实际 producer 活跃调用链及 wrapper 的文件读取，而非只枚举 `source_sha256`：三份协议通过 CLI 读取；selection 是父协议同目录的固定文件，其 SHA 同时等于父协议及 objective 协议的 `selection_sha256`。`aggregate_memory.load_cache` 第 146–158 行还会核完整开发顺序及训练／开发场景互斥。本地直接调用原函数、读取真实开发缓存 index／complete 和原 selection，通过 **200 anchors／100 scenes／混淆数组 [200,5,2,2]** 校验；未读大体素或模拟预测。

其余活跃资产是既有外部数据，不能因“不在 package 内”误判成遗漏：`--cache` 提供 index／complete 和逐样本 input／target 文件；`--native-runs-root`、`--runs-root` 提供 C/O 最终 checkpoint、manifest、训练日志及评价回执；`--development-summary` 提供原五模型汇总及其绑定产物；`--config`、`--checkpoint`、`--repo` 提供原配置、M0 与运行库。`objective_supervision_train.validate_engineering` 第 94–101 行允许相对协议路径，但**当前冻结协议的两套工程证据 directory 均为绝对路径**，没有待打包的相对证据目录。完整 full 依赖由外部 `--full-campaign`／`--full-job` 读取，development 授权与 pilot 产物由新 campaign 运行时生成。没有发现本活跃链还缺其他 package 相对的非 Python 静态资产；未调用的训练／缓存抽取入口不是本次执行分支。

原方案曾将 dispatch 包装回执的 SHA 命名为 launch SHA。现计划已改为 `launch_provenance`，分别列原 v1／失败 v2 的 dispatch 包装回执与真实远端 launch 文件的本地路径、原字节 SHA。本次四份实际哈希全部核对，且两个 `json.loads(dispatch.stdout)` 分别与对应远端 launch 对象精确相同；该歧义已关闭，不使用未说明的 canonical JSON hash。

**名称、资源与执行条件。** 实际计划固定新 job／输出名称，原 v2 目录不被复用；仍使用冻结 `deba…` 协议及 `489…`／`e3ec…` 源码。pilot **300 s／32 GiB**、development ceiling **1500 s／32 GiB**、诊断总上限 **1800 s** 均不变；外层 **2220 s** 保留原等待上限 300 s 和清理余量，不能转作扩大诊断预算。仍为零 optimizer updates、无自动重试；两 anchor 全部工程门通过并正常退出、按原资源公式生成授权后才进入 dev200。根代理随后生成实际 argv 时，只需刷新等待 deadline 和新 job／package／输出路径，必须保持此 26 文件清单、原外部数据路径及全部资源约束；本报告未声称尚未生成的实际 argv 或远端 staged package 已核验。

本次是静态资产／本地缓存元数据审查，不保证剩余实际 checkpoint 加载、GPU forward 或 parity 必然通过；这些仍须由真实新 pilot 验收。原 full summary SHA `199347570b9d34130c6366ef3fb138bbc08c1310a50010a1f200061fe428f463` 本次重核未变，已有 full 结果与解释不受此工程修复影响。
