# C/O 2×2 producer 独立代码审查

审查对象为实际 `readout_transition_diagnostic.py`（555 行），SHA256：
`e943479e17f4ff1e4c7dee7eaf948556463d8b3ef64c99c9d9709261f4358f0c`。
结论：本次有界审查未发现阻断；可按既定两 anchor 工程预检推进。这是代码审查结论，**不是实际 CUDA、逐位一致性或资源预检通过的声明**。未修改 producer 或任何冻结源。

## 实现与设计对应

- **特征边界和只读操作**：`capture_terminal`（290–313 行）只对三个完整 `bev_pred_head` 分支安装返回 `None` 的前置 hook，复制输入后在 `finally` 中移除。每分支必须恰好调用一次，输入严格为 FP32 `[5,1,40000,256]`；重组特征为 `[5,3,1,40000,256]`。原生输出固定 `[5,3,1,1,40000,16,2]`。四个组合调用真实 `forward_head`，包括完整 Linear/LayerNorm/ReLU/Linear 分支，而非仅替换最终分类矩阵；没有模型参数交换或输入修改。
- **读出没有暗中改变所捕获的递推特征**：原生路径把预测 latent 反馈入队列，终端 `forward_head` 在五时域汇集后执行。规划分支中的额外读出调用由运行时规划关闭门排除；soft-weight、flow、语义归一化和 neck 相关路径也被显式拒绝。因此这里的特征来源是固定源模型的全部非读出计算，不能缩称为唯一的 transition 参数效应。
- **强一致性门**：`measure_anchor`（335–382 行）同时要求 CC/OO 与同一次原生回放的五时域、三层全部 logits 逐位相等，并逐样本复现各自历史开发集完整 GT 混淆矩阵。后者没有被误写成历史全层 logits 复现。C/O 的 t0 输入特征逐位一致，并检验三层全部 voxel 上 `OC=CC`、`CO=OO`。字节视图比较保留正负零区别。完整输入、GT、复制特征和原生 logits 在诊断前后有独立哈希门；全部模型状态及 readout 状态在结束时重核。
- **来源和数值条件**：实际 M0 初始 head、C/O 最终 checkpoint/head、512 updates/4 passes、参数名和 13,274,016 个参数、训练历史及 cache/dev 身份均受冻结证据约束。模型重新构建后严格载入全 head，全部参数冻结并进入 eval，无 optimizer、loss adapter、梯度或阈值拟合。`seed_all(11)` 与门检查一致：FP32 matmul 不用 TF32，cuDNN TF32 开启、benchmark 关闭。动态 native1 `future_pred` 按已冻结 migration 与 synthetic filename 认证，同时绑定原 class method；源码路径 canonical 化不会把服务器 symlink 当作源码变化。
- **统计量**：第一位为特征、第二位为读出。六组系数实现两条恒等分解与 `OO−OC−CO+CC`，没有把非线性 IoU 的交互解释成唯一因果贡献。所有四组合用相同的 100 场景重抽样，每个场景两 anchors 一起保留，10,000 次 seed11 配对 bootstrap；指标先池化各时域混淆矩阵、算 IoU，再平均未来四时域，未混同 binary mIoU。逐时域效应的区间数组为 `[lower/upper, horizon]`。区间未调整多重比较，不覆盖训练种子或模型选择不确定性。
- **CLI 和资源**：pilot 仅取冻结 dev 顺序中前两个不同场景各一个 anchor；development 必须另外提供通过的 pilot 与冻结授权。输出目录必须新建，失败写回执后退出，不重试。绝对时限覆盖来源审计、构建模型、样本循环与 CPU 汇总；CUDA allocated 峰值和 allocator fraction 有门。此值是 PyTorch allocated 口径，不代表整张卡全部显存；物理 GPU 占用、外层驻留总预算和进程清理由独立 wrapper 管理。本 producer 不启动额外训练或子进程。

## 真实验证范围与解释边界

我本次独立完成了实际脚本及相关冻结 helper/原生调用路径的只读审查，并在本机 CPU 重核上述 SHA、Python 3.10 AST 解析及编译，均通过；没有执行 Torch/CUDA 模型、加载大 checkpoint、产生 hybrid 结果或构造 mock 数据。

实现者另报告：用已完成的真实 C/O 开发集记录复算 point 和 10,000 次配对区间，与冻结五模型汇总精确一致；CLI/import/拒绝路径检查通过。本次为避免重复，没有重新运行这些检查，不将其表述为本审查者独立执行。实际形状、同次 logits 逐位门、t0 门和资源可行性仍须由既定真实两 anchor pilot 验证；通过后也只说明工程适配成立。

本诊断是历史已曝光开发集上的固定模块替换效应。潜空间可重参数化及共同适应会改变交叉组合的兼容性，交叉失败不证明特征信息减少；相同初始化和固定 t0 只能部分约束此风险。不得将 CO/OC 提名为本次 full validation 新候选，也不得用此诊断声称研究目标成功。
