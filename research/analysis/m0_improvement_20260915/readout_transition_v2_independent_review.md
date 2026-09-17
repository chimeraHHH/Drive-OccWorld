# Readout / transition v2 独立工程审查

**结论：本次限定修复未发现阻断；可由根代理冻结新协议后，以新 job、新输出目录执行真实两 anchor pilot，且仅在原有全部门通过后进入 dev200。当前状态是代码审查通过、协议仍 `REVIEW_REQUIRED`、真实 GPU pilot 未验收，不是已冻结或已实跑通过。** 本审查未编辑实现、协议或冻结源，未连接服务器、派发任务、训练或改动原 full 结果。

审查对象及本次实际 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `readout_transition_diagnostic_v2.py` | `489fccce38cb418b02d6de18164b70c192b8c76d8e9654c4d8293ce9e7bf0b06` |
| `run_readout_transition_campaign_v2.py` | `e3ec99f45c7e91355cf4d2eef7ace6a92ef94036d0915f61bf0fc23b3f6a7756` |
| `readout_transition_diagnostic_protocol_v2.json`，待冻结版本 | `6c04d7cfdbc078693f521618b9a9d64ce3aedc53cf29bffa714d88936cb00c28` |

**失败来源与必要修复。** [真实 pilot 回溯](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/receipts/readout_transition_v1_failure_evidence/campaign_readout_transition_v1/pilot.log:4) 指向 producer 的 `load_evidence` 第 192 行 → `frame_aggregate.check_log` 第 104 行，报 `unsupported operand type(s) for /: 'str' and 'str'`。原 producer 第 187 行将 `a.control_run` 明确设为字符串，随后直接传给使用 `directory/'training.jsonl'` 的函数，足以解释这次真实错误。

紧邻第 193 行的 `check_payload` 同样要求 Path：其第 121 行执行 `directory/'latest.pth'`。v1 尚未走到这里，因此它是源码确认的后续同类错误，不冒称第二个已观测失败。v2 在两个调用处分别加 `Path(a.control_run)`，保留 `a.control_run` 原有字符串值，未改变传给其他 helper 的参数约定。

同类边界已核：`frame_train.validate_control` 第 186 行内部转 Path；`aggregate_memory.load_cache`、`load_arm` 内部转换 cache／run 路径；`objective_supervision_aggregate.load_candidate` 第 120 行将 run root 转 Path；producer 的 `make_models`、最终源重核、JSON／SHA helpers 均自带转换。本次未发现还需修改的同类调用。

失败凭证记录 pilot 18.278952 秒、optimizer updates 0；外层原 job 状态 `FAILED`、return code 1，campaign `FAILED_NO_RETRY`。main 的执行顺序是 `load_evidence` 后才构建诊断模型并回放，因此这次没有两 anchor 推理／parity 结果。凭证中外层数千秒主要包含等待先前 full 任务，不能把它当成 pilot 的 GPU 时长。原失败目录和 job 必须保留，不得把新任务写回原目录或称为原 job 成功。

**改动范围核验。** 本地将 v2 两个新增 `Path(...)` 精确逆替换后，producer 全文与原 v1 逐字相等；将 wrapper 的 child 文件名逆替换后，wrapper 全文与原 v1 逐字相等。未变化的部分包括 C/O 最终权重和身份链、native1 架构、原始输入及 GT、数值策略、四组合、原 evaluator、bootstrap、所有 parity、无 optimizer、失败停止和完整性核验。

协议仅替换两份新源文件名／SHA，并新增 v1 失败来源、修订编号及准备时间等 provenance。去除这些明确修订元数据、恢复原冻结状态／时间及两源条目后，JSON 与 v1 结构精确相等。兼容序列化 schema 仍为 v1 是有意保留；新源码、协议实际 SHA 和 revision 2 区分两次工程版本。修订协议引用的原协议 SHA 与真实 v1 一致。现阶段不得将待冻结 JSON 当作已冻结输入；其 `load_policy` 和 wrapper 都会拒绝 `REVIEW_REQUIRED`。

**不变的执行门。** 新 wrapper 第 114 行仍要求完整 full 依赖完成并验证其退出状态，随后只读检查指定 GPU0 空闲；每阶段启动前重核协议及全部来源。pilot 必须为固定开发顺序中前两个不同场景各一个 anchor，不选性能样本。CC／OO 的五时域三层 logits 必须与同次原生回放逐位相同，且完整 GT 混淆精确重现历史 C／O 记录；t0 特征 C=O、三层全部 voxel 的 OC=CC 和 CO=OO；输入、GT、features、原生 logits、完整模型／readout 状态均不得改变。历史记录仅含混淆矩阵，不被误写成历史全层 logits 缓存。

pilot 子进程正常退出后，wrapper 才读取绑定四产物的 PASS 回执；dev200 还需新协议 SHA、pilot complete／summary SHA 绑定的资源授权。预算保持 pilot **300 s／32 GiB**，development ceiling **1500 s／32 GiB**，诊断总上限 **1800 s**。development 实际 cap 仍为 `max(300, ceil(1.75 × (pre_sample_initialization_seconds + 200 × max_sample_seconds) + 60))`；超过单阶段或剩余总上限即拒绝，不截样本、不扩预算。每次使用新输出目录，失败无自动重试；只保存混淆／身份／哈希小记录，不持久化 features 或 logits，不选择候选／checkpoint／阈值。旧 v1 的失败耗时独立保留，不隐去。

**本地实际验证与剩余边界。** 已独立通过 Python 3.10 语法解析、上述全文及协议逆变换比较、协议全部 22 份源码实 SHA 核验、拉回的九份失败证据 SHA 核验及修订协议的四项失败来源绑定。用真实 C 的完整 512 条训练日志调用原 `check_log(str(...))`，复现相同 TypeError；调用 `check_log(Path(...))` 后，返回字典与已认证五模型 development summary 的 C training receipt 逐字段相同（512 updates、2048 examples、全部 LR 相同、日志 SHA 相同）。没有模拟 checkpoint 或伪造模型输出来声称 GPU 成功。

本地未重新执行真实 checkpoint 张量加载、CUDA 模型构建、两 anchor forward／parity、峰值显存或 dev200；这些仍由新 job 的真实 pilot 和原严格完成门验收。本审查支持修复版本按原序列运行，不保证尚未运行的后续环节一定通过。若根代理仅增加冻结状态／时间，可冻结该候选；若源代码、科学字段或资源字段再变，应重新核对对应差异。原 full summary `199347570b9d34130c6366ef3fb138bbc08c1310a50010a1f200061fe428f463` 及其科研结论不受本修复影响。
