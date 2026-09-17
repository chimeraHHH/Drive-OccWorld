# 从附加运动分支转向可被任务直接使用的状态

日期：2026-09-17。状态：实现及 H200 几何检查已完成；v1 已因独立审查发现覆盖密度输入混淆而停止；v2 的真实训练样本工程 fit 结果另记。目标仍是超过 O，当前没有达到。

## 依据与限制

R/R0 的占据改善伴随 moving recall 下降；K/B、V/G 和 Cpl/Fix 的运动机制增量很小。旧实现保留了 native future-head → occupancy 的输出路径，物理分支收到梯度不意味着预测占据依赖它。代码证据支持存在这条旁路，尚不能证明它是所有失败的唯一原因。外部深度/跟踪器扩大到密集历史后依然未胜过 CV+D，故不再追加该路线。

[LPWM（ICLR 2026 正式论文）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf7ba4b2d14e0f6a0e8247af77745094-Abstract-Conference.html) 提供一个有用的结构参照：先预测状态，再从该状态解码未来。检查其 [models.py 固定版本](https://github.com/taldatech/lpwm/blob/4cf53c403433e64c01652ac2adbec66231a46dea/models.py#L966) 的 sample_from_x/sample_from_z 与 decode_all，预测的粒子位置、尺度、内容、存在性及背景进入 decoder。只借鉴这个信息流原则；本原型不是 LPWM 复现，也不移植其随机 latent-action 模型或将其视为道路占据 baseline。

官方代码固定 commit `4cf53c403433e64c01652ac2adbec66231a46dea`；已下载的 models.py/README.md 与官方 Git blob SHA1 一致，另存本地 SHA256。正式会议信息、摘要与上述代码路径已核对，未声称通读本论文全部实验。此前 PreWorld/OccWorld 等已表明直接未来状态与解码不是新颖性本身；当前首先验证我们的任务连接，而不是命名一个新方法。

## 当前工程候选

`dense_task_state_v2.py`（复用 v1 的状态更新）从原始、冻结的合法 current fused BEV 编码出完整 `200×200×16` 三维状态，每 voxel 8 个 latent 通道。递推同时更新 source-indexed 内容与累计三维位移。每个未来时刻将状态按米制位移 forward splat 到固定 t0 LiDAR 网格，自己的 decoder 直接产生二分类占据。

没有 O future features/logits 残差，没有 O 当前 hard-positive 筛选，没有 CRN 检测框作为覆盖门。状态接口只接收 `[B,40000,256]` 的观测 token；GT、未来 ego pose 与 metadata 都不进入该接口。物理 GT 只用于原 object-group SmoothL1。稀疏监督仍是已经认证的 voxel-center 材料点标签，不能称为逐真实表面点 flow 真值。

direct 臂使用完全相同的编码器、递推、内容和位移预测头、decoder、初始化与独立 optimizer，但以 identity readout 读取内容。v2 的两臂 decoder 都只读取内容，不读取 coverage；其 flow 仍有同样物理监督；occupancy 对独立 velocity 输出头没有梯度，而 transport 臂有。该对照用于隔离几何读出，不称为已证实最强的 future-latent baseline。参数数目和更新数相同，transport 的 scatter 运算和梯度路径使 FLOPs/用时不相同。

两个臂先共用仅在训练集上拟合当前形状的 codec 初值，再分别复制并训练。新 decoder 只产生一层输出，loss 使用原 CE[1,5]+Lovasz；不会复制三份 logits 冒充 O 的三层原 head。原 GT 的 legacy occupied-mode 下采样与 ignore=255 语义保持不变，不能换成 nearest 或 max-pool 标签。

## 有边界的第一次实测

固定 seed11，从 `RandomState(11).permutation(512)` 取前 4 个训练样本。32 次 current-codec 更新，每臂再 32 次 rollout 更新，单样本、AdamW lr=3e-4、物理 loss 系数 0.1、clip=10。上限 1,800 s / 24 GiB，服务器独立 runner 托管；O/M0 与其他任务不修改。完整协议与源码哈希在 `dense_task_state_fit_protocol_v2.json`。

观察同一样本的拟合 loss、梯度、零位移/反向位移干预和移除 decoder 内容后的响应。只看训练接口能否学习，以及内容是否依赖该输入；不读开发集，不筛 best checkpoint，不据此声称泛化或超过 O。共享 codec 与两个固定 final checkpoint 留在服务器，不推送 GitHub。

## 必须继续回答的问题

1. 递推内容仍能生成“看起来在移动”的 latent，因此去掉 O 输出旁路不等于消除所有 content bypass。需要干预结果与 direct 对照共同判断。
2. 状态混合发生在原始 source 邻域，不会随未来位置改变交互拓扑，也没有显式实体身份/物体生成状态。
3. 全网格 splat 混合前景和背景；边界流失不重归一化到边缘。coverage 只用于几何归一化，不是物体存在概率或物理质量，也不作为 decoder 附加输入。
4. 输入是已有一槽 BEV，其中时序信息经过原感知网络压缩；本原型没有增加原始视觉历史。若信息本身缺失，接口改造不能创造证据。
5. 即便小样本 fit 成功，仍需学习曲线、同输入更强直接预测对照、原协议完整占据评测、物理测量和独立确认场景。不能再把 512 次短训或单个总体 IoU 上升当作研究完成。

独立审查还指出名义 0.5/1/1.5/2 s 与稀疏标签实际时间可存在偏差。v2 保留原名义 horizon 输入合同，记录所选样本的实际 dt 和偏差，不声称实际时间精确一致。v1 与 v2 都保留在版本历史中；停止 v1 的依据是接口混淆，未根据其训练指标筛选路线。
