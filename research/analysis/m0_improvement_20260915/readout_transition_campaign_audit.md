# Readout/transition 队列独立审核

2026-09-15。结论：**当前 wrapper 与 dependency helper 的只读工程审核无阻断；不代表真实 GPU pilot 或完整开发诊断已通过。** 本审查未连接服务器、未启动任务、未修改这两个源文件，也未构造完整验证或诊断的成功结果。

## 审核字节与证据

| 对象 | SHA-256 | 范围 |
|---|---|---|
| `run_readout_transition_campaign.py` | `bdb73316059c00530a5f48d7a98c8d1d06cd46b9498679e8333fe6752d1be6af` | 本次完整控制流审核 |
| `objective_full_dependency.py` | `55939464aee53067227e4ff98e4cd66890616141a33f54a97f671e25014eddde` | 完成门、进程绑定、有限等待 |
| `readout_transition_diagnostic.py` | `e943479e17f4ff1e4c7dee7eaf948556463d8b3ef64c99c9d9709261f4358f0c` | 当时接口候选，仅核对 CLI、收据和初始化计时字段；producer 科学实现另行审查 |
| `receipts/objective_full_dependency_live_probe.json` | `5b928e1b43e7fe75421a58c18af0522bc060aef65e0356bb7e93e95954660bd7` | Root 实际采集的一次 live WAIT 回调，本地只读复核 |

## 已核对的控制链

1. **依赖完整验证成功退出。** Helper 固定当前 full campaign/job 路径，核对 launch/spec、full CLI、协议/授权与包内八个源码/配置哈希，并固定 runner 与 child 的 PID/UID/startticks。只有 job `EXITED_ZERO`、returncode 0，且 campaign 与严格合并产物的哈希、四模型、5119 anchors/150 scenes 都满足合同，才返回。最终文件验收后再次要求终态完全相同且没有 failure marker。完整产物 marker 单独出现不会放行。
2. **退出竞态只等待，不重启。** RUNNING child 刚退出时，仍存活且身份正确的 runner 可获得 45 秒 finalization 宽限；STOPPING、FAILED、TIMEOUT、身份改变或超过截止时间均停止。deadline 必须是有限的未来时间。等待只读取进程与状态，不读取 partial full 性能，不据 full 的改善程度选择是否运行诊断。
3. **只使用已核验空闲的 GPU0。** 完成依赖后才调用 `nvidia-smi`；要求 GPU0 UUID 为 `GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe` 且无该 UUID 的 compute PID。development 前再次查询。通过 UUID 设置 `CUDA_VISIBLE_DEVICES`，producer 使用逻辑 `cuda:0`。这是启动前快照，未宣称锁定或预留 GPU。
4. **pilot 成功后才授权 200 anchors。** Stage 必须 exit 0，四件产物逐文件哈希一致，complete/summary 的 schema、mode、协议 SHA、样本/场景数和零 optimizer updates 均匹配；`all_engineering_gates_passed` 必须为真。pilot 为 2 anchors/2 scenes，development 为 200/100。development authorization 绑定 pilot complete/summary SHA，producer 再独立校验该 PASS、固定模型与样本身份。
5. **资源推导不读取效果。** pilot 固定 300 秒、32 GiB；以 `pre_sample_initialization_seconds`（参数解析后至首次采样前，包括来源审计及主要导入/模型初始化）和 `max_sample_seconds` 计算：

   `development_cap = max(300, ceil(1.75 × (pre_sample_initialization_seconds + 200 × max_sample_seconds) + 60))`。

   超过 1500 秒或当前 1800 秒总执行 deadline 的剩余时间则停止。独立 `model_initialization_seconds` 仍可记录，但不再冒充全部固定启动成本。等待 full 的时长与初次 GPU 空闲查询在诊断执行计时之前；wrapper 分别记录诊断耗时与包含等待的总耗时。
6. **等待期间源文件漂移也会拒绝。** 除入口核验外，每个 stage 启动前重新核对整个冻结 `source_sha256` 字典与 diagnostic protocol SHA。所有命令为 argv 数组，不经 shell。输出使用新目录，科学步骤无自动重试。
7. **子进程受原 job runner 清理范围约束。** 唯一 `Popen` 不创建新 session/process group，无 `killpg`；子进程继承 wrapper 所在的 job runner 进程组。异常时只对持有的直接 child 发送 TERM，等待最多额外 20 秒的清理宽限，再 KILL/reap。**1800 秒是诊断执行截止，不包含这段最多 20 秒的 TERM 清理宽限，不能表述为清理后的总墙钟始终不超过 1800 秒。** 外层 runner 应保留收尾余量。

## 实际验证到哪一步

- 本地 CPU：两个审核源的 Python 3.10 语法通过；AST 检查确认唯一 spawn 继承进程组、没有组级 kill、stage 前源码复核先于 spawn，且预算使用新的完整初始化字段。
- 实际负例：将已经完成的 `server_results/campaign_objective_joint_pilot_v2` 传入 full 完成验收，得到 `Wrong full completion`，正确拒绝。未用伪造 full 数据测试成功路径。
- Root 留存的真实 live WAIT：helper SHA 与本次审核一致；runner PID 2209043、child PID 2209044 的 UID/startticks 均匹配，回调报告两者存活且 `partial_performance_read=False`。回调立即终止这一次只读探针，**不是依赖完成证明，也不是本审查重新执行远端操作。**
- 未验证：full 最终成功释放等待门、GPU 空闲门的本轮现场状态、真实 readout pilot 的数值正确性/资源占用、200-anchor 诊断的吞吐与完整输出。须由真实运行及对应收据判断，不从静态审核推出 PASS。

上述结论绑定所列字节；producer 若继续修改，应在最终冻结时复核接口，不能把本记录作为新字节或未运行 GPU 结果的凭证。
