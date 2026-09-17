# JEPA-WMS predictive-state audit（2026-09-17）

## 范围与固定版本

本笔记审查的是论文 **What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?** 的 arXiv v4（`2512.24497v4`），以及官方仓库 `facebookresearch/jepa-wms` 的 `main` 提交
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`（2026-04-11，提交页见
[`13cf1d9`](https://github.com/facebookresearch/jepa-wms/commit/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0)）。这不是 arXiv `2602.18639` 的 bisimulation 论文。以下结论来自论文正文/附录和该提交的四个源码文件；未下载权重，也未运行训练。

论文：[arXiv 摘要与版本记录](https://arxiv.org/abs/2512.24497)，[v4 全文 HTML](https://arxiv.org/html/2512.24497v4)。

## 已核实的机制

**视觉 encoder 默认冻结，但配置上可切换。** 论文 §3（“Training method”）明确写 `E_phi^vis` remains frozen；训练的是 action encoder、可选 proprio encoder 和 predictor。仓库 [`app/vjepa_wm/train.py`](https://raw.githubusercontent.com/facebookresearch/jepa-wms/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/train.py) 在约 86–89 行把 `freeze_encoder` 默认设为 `True`，约 553–561 行传入 optimizer 初始化，约 642–644 行只有在不冻结时才给 encoder 包 DDP。具体参数筛选位于未纳入本次四文件审查的 utility，因此源码层面最稳妥的表述是“入口默认冻结并按 flag 传递”，论文层面则是方法的明确设定。

**训练不是只有单步 teacher forcing。** 论文 §3、表 1 先定义 one-step embedding prediction；context window 的消融取 `W=1...14`，实用配置为 simulated navigation `W=3`、real manipulation `W=5`。论文 §4 又加入 autoregressive multi-step rollout：每个 iteration 保留逐帧 teacher-forcing loss，并把前一步预测 embedding 送回 predictor，再与真实未来 embedding 比较。实验比较 1-step、2-step、3-step、6-step；推荐 simulated navigation 用 2-step、real manipulation 用 6-step。

源码 [`app/vjepa_wm/video_wm.py`](https://raw.githubusercontent.com/facebookresearch/jepa-wms/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/video_wm.py) 约 372–453 行的 `compute_loss` 是 embedding 对 embedding 的 shift=1 损失；约 455–701 行的 `rollout` 才实现序列 unroll。训练入口 [`train.py`](https://raw.githubusercontent.com/facebookresearch/jepa-wms/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/train.py) 约 865–890 行先计算单步项，约 994–1027 行在 `rollout_steps > 1` 时添加 rollout 项。因此“几步”应报告为配置值，不能从 JEPA-WM 一概写成固定 6 步。源码默认还允许配置 rollout prefixes 和 stop-gradient；论文描述的实际多步训练使用 truncated BPTT，约 619–628 行可见递推 context 在 `rollout_stop_gradient` 打开时 detach，约 659–684 行再对未来 encoder features 计算 rollout loss。

**decoder 不在核心 planning/dynamics loss 中。** 论文 §3 明确 JEPA-WM 只做 embedding prediction，不含 reconstruction、reward、value 或 policy head。规划的 cost 是 unrolled predicted embedding 与 goal encoding 的 embedding distance（§3 “Planning”），不是把未来图像或 occupancy decode 后再规划。

源码 [`app/plan_common/models/wm_heads.py`](https://raw.githubusercontent.com/facebookresearch/jepa-wms/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/models/wm_heads.py) 约 30–143 行和 146–179 行分别定义 image decoder 与 state readout；它们有自己的 `compute_loss`，由训练入口单独优化。论文附录 C 把 visual decoder 和 state decoder 描述为分别训练的 probe；附录 G 约 1096–1104 行说明 decoder 用来测视觉/状态可读性，而不是 temporal consistency 或 predictor 的任务路径。尤其 [`app/plan_common/models/state_decoder.py`](https://raw.githubusercontent.com/facebookresearch/jepa-wms/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/models/state_decoder.py) 约 131 行先执行 `video_features.detach()`，所以该 probe 的回归误差不会更新 encoder/predictor。

有一个需保留的源码边界：`train.py` 的可选 `train_heads_on_predictor` 分支中，image head 的 predictor 输入在约 964–969 行显式 detach；state-head 分支在约 980–989 行看起来仍把 `video_features`（真实 encoder features）传给 `compute_loss`，而不是 `pred_video_features`。这使该分支不能被当作“state decoder 训练了 dynamics”的证据；若要引用该可选分支，应先按实际 config 复核。

## 为什么 current reconstruction 不能证明 dynamics-sufficient state

这里有三个独立的逻辑边界。

1. **可重建不等于可预测。** visual decoder 是在冻结 encoder 表征上单独训练的逐帧 probe。它回答“当前画面信息是否仍可被 decoder 读出”，不回答在 action 条件下，表征是否保留速度、接触、遮挡后的隐状态等决定未来的变量。
2. **训练目标没有 current reconstruction 约束。** predictor 被训练成匹配未来 encoder embedding；decoder 只读出 predictor 的结果或真实 encoder 的 probe，不参与核心预测梯度。故当前重建好只能说明观测信息可读，不能推出 latent 对动力学是 Markov-sufficient。反过来，某些对预测有用的抽象也未必能高保真重建像素。
3. **真正的结论标准是闭环任务。** 论文用多步 embedding error、proprioceptive state probe、visual/LPIPS probe 作诊断，最终仍以 physical planning 成功率为任务证据；论文还明确指出较小的 world-model embedding error 不保证 planner 成功，OOD action 和 cost landscape 会造成额外失败。因此不能把 reconstruction 或 probe 优势直接升级成 occupancy/运动泛化结论。

## 对当前 occupancy 方向的可迁移边界

可直接借鉴的是实验结构：冻结观测 encoder 后锁定 one-step teacher forcing 与显式 autoregressive rollout；分别记录 latent rollout 误差、任务原生 readout 误差和最终规划/预测指标；用同输入、同初始化的 decoder/readout 对照来区分“状态没有预测好”和“状态已预测但读出不兼容”。这对当前 T/J/D 的 readout 交叉组合和 future-state 诊断有方法学价值。

不能直接迁移的是任务含义：JEPA-WMS 面向机器人 action-conditioned latent planning，输入含 proprioception，目标是图像/状态 embedding；论文没有证明 dense occupancy、相机/雷达融合、世界坐标位移、物理 EPE 或当前项目的历史边界。其控制结果也不能充当当前 occupancy 泛化证据。当前项目若用类似机制，必须仍以原始 occupancy target、坐标一致的运动指标和 O 的固定评测协议作主证据。

## 结论

官方证据支持的最强表述是：冻结视觉表征配合短上下文、单步 teacher forcing 加有限 autoregressive rollout，足以在其机器人 planning 任务中形成有用的 predictive embedding；decoder 主要是独立 probe。它支持“测试动力学可用性不能只看 current reconstruction”，但不支持“同样的 decoder、rollout 步数或控制收益会在当前 occupancy 任务成立”。


主 agent 已独立读取 v4 正文 §3/§4/附录 G.3，并下载上述固定提交的四个源码文件核对 teacher forcing、rollout detach 和 state decoder。文件哈希记录在本地 `jepa_wms_repo_audit_20260917/source_manifest.json`；该源代码镜像不随研究结果提交。论文定义的是一族方法，其研究实例冻结视觉 encoder，不应把冻结写成所有 JEPA-WM 的必要条件。
