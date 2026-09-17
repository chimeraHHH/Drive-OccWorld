# Readout / transition v3 真实 pilot 与资源停止审计

**独立结论：固定两场景各一 anchor 的真实 pilot 工程验收通过；dev200 按原资源门停止，未运行。** 这是工程可行性证据，不提供两样本效能、贡献比例或机制结论；不追加预算或新任务。

已核 [pilot complete](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v3/pilot/complete.json) 的四件产物实际 SHA 全匹配，schema／状态为 `PASS_READOUT_TRANSITION_ENGINEERING`，样本／场景均为 2，optimizer updates 为 0。complete SHA：`b445a81450edc8c7a236747df0194ac4f10e704160925b0cd30abb24e8d8a328`；summary SHA：`6d97cb9a6010ac162c6a72a403b967a6e18c7420ec6ccd6795a349f4b81aaa8e`；其诊断协议 SHA 与实际冻结 `e760da6c6eb645b0f2862a58bd37a340d97ed94c51d254384fe084c71386695a` 一致，22 份来源实际 SHA 均匹配。

两条记录确为原开发顺序 ordinal 0、2，对应固定两场景，身份和 GT 绑定相同。已逐项核验四组合五时域 GT 行计数；CC／OO 与真实 C／O 历史开发记录的五时域完整混淆精确相等。真实回执还记录：对角五时域三层 logits 与同次原生回放逐位相同；t0 特征 C=O、全部三层 OC=CC／CO=OO；输入、GT、features 及模型状态不变。两模型使用固定 final head，全部 eval／frozen、原 FP32 策略、规划／flow／M3 关闭，没有安装损失适配器或构建 optimizer。历史记录只支持混淆复现，不被宣称为历史全部 logits 缓存。

实测 pilot 总耗时 **163.705206 s**，首样本前初始化 **151.759568 s**；两样本分别 **5.892220／5.485702 s**，allocated 峰值 **3.146122 GiB**。按未改变的公式：

`raw = 1.75 × (151.7595680099912 + 200 × 5.892220180016011) + 60 = 2387.8563070230884 s`。

所以实际所需授权 `max(300, ceil(raw)) = 2388 s`，超过 development 的 **1500 s** 上限；pilot 加该授权为 **2551.705206 s**，也超过 **1800 s** 总上限。停止原因是预先冻结的时间预算规则，不是显存超限或 parity 失败。不得为了继续而换均值、去掉安全系数、减样本或扩大上限。

campaign 的真实失败回执为 `FAILED_NO_RETRY: Measured development budget exceeds frozen ceiling; stop`；job 已以 `FAILED`／return code 1 终止。镜像中没有 development 目录、日志、授权或整个 campaign complete，不能写成 dev200 已完成。campaign 保留的 `RUNNING_PILOT` state 是最后阶段记录，不覆盖终态 job／failed 回执。仅 pilot 为 PASS 与 campaign 因预算未完成二者同时成立。

原 full5119 上 O 对 M0 的改进结论不由这次 pilot 决定，也不受这次预算停止推翻；读出／未来特征及其交互的总体机制仍未获得 dev200 诊断证据。本审查只读真实产物并复算绑定、计数相等与预算，未重新运行模型、训练或统计两样本效能。
