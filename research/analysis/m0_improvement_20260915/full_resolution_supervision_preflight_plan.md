# 完整分辨率监督：资源预检计划

2026-09-15；实现已完成，当前仅通过本地静态验收，真实 GPU 结果尚待派发验证。该预检与 reference 坐标实验分开：使用 **M0 初始化的 native1 单槽**，原生空间转换、真实未来 ego/action、相同 12 项损失、3 个 decoder 层及当前＋4 个未来时域。唯一干预是监督分辨率；原 GT、有效标签规则和 evaluator 均不变。保留单种子 11，不读任何 probe 更新后的权重。

## 最小源码变换

在独立运行时 adapter 中派生两个原方法，并以原文件 SHA、规范化 AST 及变换计数验收；不修改仓库、缓存或现有训练器：

1. `WorldHeadV1.loss_occ`：去掉唯一的 `target_voxels = _downsample_occ_target(target_voxels)`，直接把 `compute_occ_loss` 已完成历史切片和 frame-major 排列的完整 GT 交给三个层。原 `target_voxels_prepared=True` 及层循环保持原样。
2. `WorldHeadV1.loss_voxel`：将唯一的 `H,W,D = 256,256,20` 替换为完整目标的 `target_voxels.shape[-3:]`，并显式验收为 `512,512,40`。原 logits `F.interpolate(...,mode='trilinear',align_corners=False)` 保持原样；本配置原预测 200×200×16，必然进入该分支。

除此之外，原 `[1,5]` CE 类别权重、四个系数均为 1、255 ignore、三个层求和、五时域展平、Lovász `classes='present'/per_image=False`、两种 scaling 及原训练 loss 汇总全部不改。不能把“同损失家族与权重”写成数值目标不变：目标网格及原先被粗化/忽略的标签已经变了，这正是待检验变量。全局 `_downsample_occ_target` 不替换，不影响其他任务方法；当前预检只允许上述 `compute_occ_loss→loss_occ→loss_voxel` 调用。

**数值路径也属于控制变量：** geo 仍用 `1−softmax(class0)`，sem 仍用 `softmax(class1)`；geo 的 `1e−5`、原 BCE、浮点 dtype 均原样保留。不把相消稳定化、损失重加权或空类策略暗中叠入本实验。

## 固定两条工程样本及验收

复用原 train512 缓存顺序中前两个不同场景的首个 anchor，身份与已完成 `frame_preflight_v3/manifest.json` 一致：

| sample_token | scene_token | official_index |
|---|---|---:|
| `464fe0be05a74ec9852573cb9e3afd89` | `1ad821fcfc9f4c24bb879ec7d5ba0ec1` | 9899 |
| `1dacdffab5c240f1b20e05566b940ef6` | `d2db9f5df62c4d338d3bed43f616b954` | 23322 |

两者都为 `split=train`，不得按损失、前景数量或预测表现换样。冻结 parent protocol、config、cache index、inputs/targets、原 8 个源文件及新 adapter/preflight 的 SHA。用独立新输出目录，只保存小型收据和统计，不写 checkpoint 或大预测张量。

- **静态结构：** 恰好两处目标 AST 改动；参数名/形状/数量、M0 初始 future-head tensor SHA 不变；`memory_queue_len=1`、`history_queue_length=2` 不变。没有坐标 adapter、teacher、额外损失、threshold 或 mask 修改。
- **实际调用：** 观察 `compute_occ_loss` 的真实参数，验收 `[3,5,2,200,200,16]` 原生层/时域预测和 `[5,512,512,40]` 完整 GT；避免另写一份猜测的 permute。各层收到的 GT 必须逐位等于原 targets 去掉两帧历史后的原生排列，0/1/255 数量也相等。
- **评测不变：** 同初始化、同未来 head 精度（matmul TF32=false，cuDNN=true）下，安装 loss adapter 前后的五时域、三层原始 logits 应逐位相同。调用原 `evaluate_occ_records`，逐时域 full-GT 混淆矩阵逐项相同。不能和 observer TF32=true 的旧 cache logits 直接要求逐位相同。
- **损失真实性：** full 版本调用前保持原 logits。源码核验确认 loss 与 evaluator 使用相同轴、完整输出大小及 `trilinear/align_corners=False` 插值；实际运行核验三个层收到的完整 GT 与原切片逐位相同。本次不额外保存/比较两个完整插值张量，不能将这项未运行的 tensor-bitwise 检查写成 PASS。12 项名称/系数不变，但不要求 full loss 数值等于 coarse loss。粗对照继续调用完全原方法。

## 无 optimizer 的前向、反向与资源记录

分别为 coarse 与 full 加载相同初始 M0/native1。全部采用 `model.eval()` 关闭 dropout，但在 `torch.enable_grad()` 内直接调用原生 `future_pred`，固定 `valid_frames=[1,2,3,4]`，保留整个未来链的 BPTT。不能使用内部包着 `no_grad()` 的评测 replay。每次前向都重置 seed11；两条 normal anchor 在 coarse/full 条件下独立重放，必须满足完整原始预测 SHA 与五时域 full-GT evaluator 混淆记录精确一致。所有权重始终不更新，结束核初始 future-head tensor SHA 相同。

固定顺序为 coarse 的两条 normal 加一条显式后验 rare，随后 full 的同两条 normal。rare 为已完成 Train16 中最后一条 `706d7015b80d4329b80d4a8c52139ab4`，仅检验极端损失的梯度来源，不进入 full 资源配对，不作性能确认，也不从已有统计中删除。

每条 coarse 样本在同一前向图上，对 CE、sem、geo、Lovász 各自跨三层之和依次作四次 `autograd.grad(...,retain_graph=True)`，最后对原 12 项之和作 `retain_graph=False` 的总 VJP 释放图。full 只作总 VJP。梯度覆盖所有 13,274,016 个 future-head 参数；`None` 显式记录并按零计，单族零范数允许且 cosine 为 null，完整总梯度必须有限且非零。不创建 optimizer，不调用 `.backward()` 或 `.step()`，不累积参数 `.grad`。

仅在 CPU 临时保存 float32 梯度向量，以 float64 分块点积记录四族 L2/两两 cosine/与 total 的 cosine、full 与同锚 coarse 的 total cosine，以及 rare 与两条 normal、两条 normal 相互的 total cosine；磁盘只保留小型标量统计。三样本方向比较不能外推训练总体，也不等于原四样本累积、clip 或 AdamW 后的更新方向。

同步计时单独报告 forward、包含 GT 核验开销的 loss、total VJP、额外四族 VJP，以及 CUDA peak allocated/reserved。coarse 峰值包含重复且保留图的分项 VJP，不能和 full 总 VJP 的峰值一起冒充同等训练步骤资源。若触及预算，保留已完成记录并失败停止，不自动延长或换样。

## 进入正式训练前的资源门

根任务已确定本次单 GPU **600 秒程序截止**、**64 GiB allocator 上限**，外层 job_runner 最多 660 秒；框架外显存不受 PyTorch allocator 限额约束。这是工程预检预算，不是正式训练预算。OOM、非有限 loss/总梯度、身份/完整 GT/指标 parity 失败，均保留失败收据并停止；不得自动增显存、重试、降分辨率或换采样损失。

本次为 eval/no-dropout 的 VJP 工程探针，不能直接称为训练峰值或吞吐：没有参数 `.grad` 累积缓冲、训练 dropout 和 AdamW moment。正式 AdamW 两份 moment 约 `2×13,274,016×4` 字节，另有梯度及临时 buffer；额外预留 1 GiB 也只是估计，不构成充分资源保证。正式累积 4 微步应逐个 backward 释放图，不能同时保留 4 份 loss 图。

正式 512 更新对应 2,048 个 anchor 前反向。两条 full 样本的 `forward+loss+total VJP` 只作成本筛查，不能直接当训练单步；还需考虑训练模式、初始化、200 个完整评测、checkpoint I/O 和样本差异。**在根任务为新训练冻结总时限并完成必要资源确认前不启动；若未来采用 3 GPU 小时上限，成本筛查已明显超出 10,800 秒，或峰值加必要余量超显存门，则本方案不进入正式 512 更新。** PASS 不会触发自动训练，两样本也不能证明全数据最坏资源上界。

若 full Lovász 排序成本不可接受，应结束这份等损失家族的资源路线；任何 uniform/分层采样 CE、采样 Lovász、去 scaling 或仅监督最后层都要另立明确目标。尤其采样排序/ratio 不等价于原始全体素 Lovász，不能为了通过资源门替换后仍声称“只改监督分辨率”。
