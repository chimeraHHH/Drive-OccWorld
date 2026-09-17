# Shared rigid 原生四次更新预检：独立审查

结论：**实际元数据与日志审查通过，未发现阻断；仅证明工程连接成立，不证明预测改善或两任务相互增强。** 本次只用本地 Python/NumPy 读取真实镜像，未导入 Torch、访问服务器、运行模型或读取大权重。详细文件清单与独立标量复算见同目录 `shared_rigid_native_preflight_v1_independent_audit.json`。

- 逐项核实 top complete、两臂 complete/manifest/training.jsonl 的完整哈希链，24 个生产源码及 29 个打包文件与实际本地源码、协议、launch 一致。终态为 EXITED_ZERO/0；既有 2026-09-16 01:16:50 UTC 检查收据记录 runner/child 均不存在，本审查未重新查询进程。
- 独立重建 seed11 原样本排列：两臂各 4 update × 4 microstep，共同 16 个训练 anchor/16 场景；身份、缓存、原始标签、稀疏标签与几何记录对应。16 对 RNG 及输入/支持一致；384 项 horizon/group 对象数与点数核对通过。原六项 CE+Lovász、物理组均值及 total loss 标量关系一致；FP32 汇加与双精度复算最大差为 1.10e−7。前四次微步输出哈希相同，CE 最大差 3 ULP，Lovász 精确相同，符合原数值契约。
- 初始两 anchor 的五时域三层 O 预测/原混淆及全场 float64 CV 一致性收据已交叉核对。两臂各 130 个原生 head 参数张量、11 个新模块参数张量的 Adam 记录均到 step4，日志值有限，权重摘要确有更新。输出无 checkpoint、开发样本数为 0、无性能评分。这里核的是运行时摘要及其绑定关系，**未重新计算原始预测、梯度或 Adam 状态张量**。

晚期 VJP 在第三次更新后、固定 ordinal394 上记录：Cpl 占据→pose 范数 5.3661e−6，物理→pose 为 0.12615；Fix 占据→pose 仍为 6.6466e−7，因为两臂都保留 learned-value 路径。物理梯度只进入 object encoder/pose，原生 future head 全部为 None，符合本设计。不能把两臂梯度范数相减解释为独立地址梯度，也不能把原始范数比解释为 Adam 更新占比、冲突或协同。

同一已更新模型切换 Cpl/Fix 地址时，learned values、物理场和 t0 三层 logits 保持精确一致；Cpl 权重下未来 logits 最大差 2.6703e−5、平均差 1.6046e−7（Fix 权重下平均 1.6052e−7）。这支持地址边非零，但效应很弱；变化元素很多不等于准确率改善，更不等于物理运动改善。

资源记录：外层 runner 驻留 **124.2396 s**；内部汇总 **122.1646 s**，其中全场几何 78.6307 s、四次配对更新 24.3214 s。峰值 allocated **20,683,954,176 bytes（19.2634 GiB）**，在内部 1200 s / 外部 1260 s / 32 GiB 上限内。几何及 VJP 开销包含在本次测量中，不能直接当正式训练 ETA。

证据哈希：

- producer：`99a043d557b4001259fc0067257177239262e5cac7af84d866f68852a55696d2`
- 冻结 protocol：`471d587d1254cd6ce6333a523ad18762bef0aa7ee6b07c068a8e8dd4b535cc56`
- 真实 summary：`59ad77236568ae7d9933a1ac9a6630da576b85c68e524bfd7667a6361e3252cc`
- 真实 complete：`198384ba7e768901c3c996504c897f5e7154bfdb4ddf61d3eb4da089a76d8212`
- 独立审查 JSON：`97ebf59a1fc4c7f071dd6b08c4065fcae94f91f7af613afcb7124298544bf5d0`
