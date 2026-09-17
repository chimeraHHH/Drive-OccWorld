# Readout / transition v3 独立工程审查

**结论：限定修复无阻断，可由根代理冻结后用新 job／新输出执行原两 anchor pilot，再按原规则进入 dev200。候选协议目前仍是 `REVIEW_REQUIRED`；没有真实 v3 GPU／parity 通过结论。** 本审查未派发、冻结或改动任何实现／协议。

审查实际版本：producer `9921c211e477d2f05dc08848436c66d238cc81a27f13d5124f81bd359ffd675b`；wrapper `ddd8fd7536602ed5fc9e6ee66ce4c99f5bb7b8ebcf8e8009312e1f10bde02b05`；待冻结协议 `34424122d81424da7a36f7da188973568526dae8a3fcd7903385d926f0196333`。

真实 v2b 回溯停在 producer 第 204 行的完整来源回执比较。只读 CPU 证据 [readout_transition_v2b_receipt_types.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/receipts/readout_transition_v2b_receipt_types.json) 的 SHA 为 `14405b641d2872f8a5930aa511023032dea224b85331356ed62e9721fe3493a7`：C／O 各自唯一差异是 `$.checkpoint.optimizer_groups[0].betas` 的 Python tuple 与历史 JSON list，值均为 0.9、0.999；完整 JSON roundtrip 后所有字段相等。证据绑定的历史 summary、原 job state 和新协议所列失败文件实际 SHA 已独立核对。该诊断未调用 main／make_models、forward 为 0、optimizer updates 为 0、CUDA 未初始化，不能推断任何模型结果。

v3 唯一功能变化是第 204 行将完整实际 receipt 经 `json.loads(json.dumps(receipt, allow_nan=False))` 后，与完整历史 JSON receipt 做严格相等比较。没有选择性删字段、忽略 hash、设置数值容差、`default=str` 或改动 checkpoint／模型／输入。比较只处理已生成的审计回执，保留原实际 receipt 对象；JSON 所不支持的对象和非有限浮点数仍会失败。此修复解决持久化 JSON 对 tuple 的表示差异，未放宽模型输入的 typed-tree 或任何预测／GT parity。

已独立验证：逆替换该一个表达式后 producer 与 v2 全文逐字相等；逆替换 child 文件名后 wrapper 与 v2 全文相等；两者通过 Python 3.10 语法解析。协议撤销版本／来源／状态元数据及两源条目后，与冻结 v2 结构精确相同；前次修复 provenance 仍被保留。四组合、固定 C/O final、原数据／指标、全部工程门、无 optimizer、300／1500／1800 s 和 32 GiB 预算、失败停止均不变。

协议全部 22 份 Python 来源实际 SHA 匹配；加 `protocol_v2.json`、`objective_supervision_protocol_v1.json`、新诊断协议及 **`selection_v1.json`** 构成 26 文件闭合集合，精确等于真实 v2b launch 清单仅替换三份版本文件所得集合。selection 保留原 SHA `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d`，无新增相对静态资产依赖。本次核验的是候选闭合集合；根代理仍需据此构建实际新包，不应再次仅按 22 个 source 条目打包。

旧失败与 full 结果保持不变。真实 checkpoint 加载、两 anchor forward／全部 parity、资源实测及 dev200 尚须新任务验收；本地静态审查和已完成的只读 CPU 类型诊断不替代这些运行结果。无额外科学门槛或预算要求。
