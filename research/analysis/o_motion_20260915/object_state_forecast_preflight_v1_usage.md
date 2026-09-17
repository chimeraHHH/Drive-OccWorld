# V/G 原生四次更新预检（待审核，未派发）

入口是 `object_state_forecast_preflight_v1.py`。它只允许既有 seed11 第一轮排列的前 16 个 train512 样本、每臂 4 次 accum4 更新，不保存权重、不读开发集、不产生成绩。原 `future_state_motion_preflight_v1.py`、O、D、conditioner 和 physical readout 源文件保持不变。

`object_state_forecast_preflight_protocol_v1.draft.json` 的状态为 `REVIEW_REQUIRED`。其中三个尚不存在的 train512 预测资产 SHA 保留 null；实际 CPU/GPU 数据准备完成并审核后，由根任务填写真实 complete/manifest/predictions SHA、核验推理源 SHA、确定预算并冻结状态。没有自动 pilot→正式训练。草案资源 600 秒/32 GiB 只是沿用预检上限，尚无本轮实测。

调用参数沿旧 K/B 入口，额外加：

```text
--predictions 实际已完成CRN_full512输出目录
--raw-metadata 原motion_targets_v1目录
--selection 原selection_v1.json
```

其余必需参数为 `--protocol --connected-protocol --training-run --config --checkpoint --o-checkpoint --repo --runtime-contract --train-cache --sparse-labels --out --max-seconds --max-allocated-gib`，`--device` 默认为 `cuda:0`。`--training-run` 仍指向实际完整 J/D 训练目录，用于严格加载 D；不指向任何预检或 K/B 权重。输出目录必须全新。

预测资产必须是 `crn-state-train512-predictions-v1`，512 个训练样本/256 场景，完整原框顺序；拒绝 dev200 或两锚 pilot 文件。完整 export、manifest 和 predictions 均验 SHA。归一化 prediction 的 translation 已是 global 几何中心，消费端不再加半高。按固定八类保留全部框，不新增 score 阈值、top-k 或 GT 关联。中心、完整旋转和 global `[vx,vy,0]` 用当前 G0 变换到 t0 LiDAR 坐标；只取元数据当前姿态，不向模型传 annotations、未来姿态或未来时间。

两臂从同一个实际 O 和完全相同的新 seed11 conditioner/readout 初始值开始，原 future head、conditioner、物理 readout 都训练；D 与 O 非 future-head 参数冻结。V 使用预测速度，G 仅将三个速度输入槽清零，几何/类别/置信度/参数/订单/RNG/目标/预算相同。conditioner 通过既有 `rollout_prior` 严格注入 frame1..4；物理 readout 保留共享未来特征图，输出 `D + delta`。占据输出直接来自原生 head，不在末端合成 logits；训练后 t0 可以随原读出头变化。

新模块共用 AdamW、原 readout LR1e-3/warm25/clip10；原 head LR1e-5/warm50/clip35。占据使用原 O 六项 CE+Lovasz，物理使用原完整稀疏对象/分组 SmoothL1，权重各1。空/缺失监督语义沿原 helper，不按检测成功裁分母。

预检核初始全部5h×3层 O logits、D 位移及原 full-GT confusion；四次更新记录成对 RNG、源状态/完整标签 SHA、同样本订单和真实 AdamW 步数。第13个 microstep（已有3次更新）分别检查两类 loss 对 conditioner/state_mlp 和原 future transformer 的非零 VJP，避免两个零初始化投影造成的首步假断路。随后同一已更新模型、同一 RNG，仅把输入速度置零：V 的未来状态/占据/物理输出须能变化，G 必须逐字节不变；两臂空 token 集均应等于各自未注入条件的原生通路。该检查不计算指标，不表示状态输入能改善预测。

本地 `object_state_forecast_preflight_v1_cpu_checks.json` 记录 Python3.10 语法、CLI、NumPy 坐标/类别/空集合检查、实际 dev200 资产拒绝，以及真实首16训练元数据的身份/姿态/SHA。没有运行本轮 Torch/native 前向、backward 或 optimizer；原 conditioner 的6项实际 Torch CPU测试是独立既有证据，不能代替本次四更新预检。
