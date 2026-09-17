# Readout transition v1 失败与 v2 最小工程修复

## 真实原因

只读取得服务器 `campaign_readout_transition_v1/pilot.log`，SHA `850eb3d764139f24a77cfb58dba93232adbaa03b5581ee2b8f4a66e076ef5034`。真实调用链为：

```text
readout_transition_diagnostic.py:463 main → load_evidence
readout_transition_diagnostic.py:192 audit.check_log(a.control_run, ...)
frame_aggregate.py:104 rows = jsonl(directory/'training.jsonl')
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

producer 第 187 行把 `a.control_run` 转成字符串；`frame_aggregate.check_log` 的接口直接要求 `Path`。下一行 `audit.check_payload` 接收相同字符串，其第 121 行 `directory/'latest.pth'` 也有相同缺陷。前面 `helper.validate_control` 能工作，是因为它自行执行 `Path(a.control_run)`；不能由前一个调用成功推断后两个接口兼容。

真实 pilot 在 18.278952419990674 秒失败，optimizer updates 为 0；失败位于 `load_evidence`，尚未进入 `make_models`、模型前向、两个 anchor 采样或任何组合指标计算。证据检查可能已经读取 CPU checkpoint，不等于尚未发生 checkpoint I/O。wrapper 保留 `FAILED_NO_RETRY`，job 为 `FAILED / returncode 1`。失败目录完整保留，没有重新使用。

远端实际部署的 producer、wrapper、frame helper 与 v1 协议四个文件均只读重新计算 SHA，与本地冻结字节完全一致，排除这一报错由本地/远端版本错配造成。证据目录：`receipts/readout_transition_v1_failure_evidence/`，其中 `source_receipt.json` 记录各原始路径、获取时间与逐文件 SHA，`deployed_source_sha256.json` 保存上述部署核对。先前静态验收没有执行这个真实函数组合，不能作为端到端通过证明。

## 唯一运行修改

新 `readout_transition_diagnostic_v2.py` 相对 v1 **只有两个表达式**：

```diff
- audit.check_log(a.control_run, ...)
+ audit.check_log(Path(a.control_run), ...)
- audit.check_payload(a.control_run, ...)
+ audit.check_payload(Path(a.control_run), ...)
```

新 `run_readout_transition_campaign_v2.py` 相对 v1 **只有一个字符串**：将子进程脚本名改为 `readout_transition_diagnostic_v2.py`。没有修改冻结 helper、模型代码、读取路径的实际值、输入/GT、权重、选择顺序、精度、指标、bootstrap、进程组或清理逻辑。

同类检查覆盖 producer/wrapper 的路径运算和外部目录接口：`metric.load_cache`、`metric.load_arm`、`helper.validate_control`、`aggregate.load_candidate` 均自行规范化为 `Path`；`make_models` 的目录已在构造时转换；`measure_anchor` 收到 `load_cache` 返回的 `Path`；两个外部 `audit` 调用是这条链中发现的全部未转换入参。不声称静态扫描能排除其他未执行的 GPU/运行时问题。

新协议 `readout_transition_diagnostic_protocol_v2.json` 当前为 **REVIEW_REQUIRED**，不能直接启动。数据结构仍兼容 schema v1；文件名、工程 revision、源 SHA 与新协议 SHA 区分 v2，避免无必要的输出 schema 改造。它保留全部科学字段，只替换两条 versioned source，并新增真实失败来源与工程变更记录。根任务完成独立审查后才冻结新协议；旧 v1 protocol 和源码完全不变。

候选源码 SHA：

| 文件 | SHA-256 |
|---|---|
| readout_transition_diagnostic_v2.py | 489fccce38cb418b02d6de18164b70c192b8c76d8e9654c4d8293ce9e7bf0b06 |
| run_readout_transition_campaign_v2.py | e3ec99f45c7e91355cf4d2eef7ace6a92ef94036d0915f61bf0fc23b3f6a7756 |
| test_readout_transition_path_v2.py | 391c3f4372c8d26927dd3c8f4c7eefa052309be8a737dcf5b84f07e6b177cd48 |

## 已完成的有界验证

`python3 -B test_readout_transition_path_v2.py` 的四项 CPU 检查通过：真实 C 的固定 512-update 日志在旧字符串调用下复现同一 TypeError；使用 Path 后得到的审计 receipt 与已认证真实五模型 summary 中 C 的 receipt 完全相同；第二个故障点只执行原 AST 的路径表达式验证，不模拟 Torch checkpoint 加载；精确逆替换后两份新运行源码均逐字节还原 v1，全部其他源 SHA 保持一致。另完成 Python 3.10 语法与两个 CLI `--help` 检查。

机器可读回执为 `receipts/readout_transition_v2_engineering_tests.json`。没有构造 mock 成功结果，没有运行 GPU、前向、优化器或新种子。真实 v2 checkpoint 全链与两 anchor 工程 parity **尚待 pilot**；原组合输出与模型/GT parity 门仍精确保留，不能以 CPU 修复通过代替它们。

## 资源与后续接口

新的、独立命名的 job/output 才可使用 v2；不得重启或覆写失败 v1。wrapper CLI 参数不变，入口改为 `run_readout_transition_campaign_v2.py`，`--diagnostic-protocol` 指向根任务随后冻结的 v2 JSON。依赖仍必须是同一 full job 的真实 `EXITED_ZERO` 完整回执；GPU0 UUID/空闲门不变。即使 full 已完成，也必须提供有效的有限未来 wait deadline。

预算保持 pilot 300 秒、development ceiling 1500 秒、诊断合计 1800 秒、32 GiB；TERM 清理宽限最多额外 20 秒。development 仍只能由真实 PASS pilot 的 `pre_sample_initialization_seconds` 和最慢 anchor 耗时按原公式推导预算，不能使用失败的 18.28 秒外推吞吐。没有增加训练或评测预算，没有自动重试。

原 `plot_readout_transition_results.py` 精确绑定 v1 协议/源；它不能冒用 v2 结果。若 v2 真正完成，绘图入口需由根任务另行明确进行版本绑定迁移，不能削弱其 source/receipt 检查。此修复没有改绘图文件。
