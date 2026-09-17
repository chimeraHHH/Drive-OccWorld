# V/G 正式训练入口草案（未冻结、未运行）

`object_state_forecast_train_v1.py` 已接入固定512更新、4轮 train512、每臂 accum4、seed11 连续排列，以及固定末尾 dev200。当前源码已绑定新增 v2 预检及内部含 p0+h*v 的95,872参数 conditioner；独立10项Torch CPU检查已通过，**真实原生四次更新与 train512 预测资产仍未完成，正式资源未冻结，因此不可派发这个草案**。它不是已完成实验。

两臂全量更新原 native future head 与新增 conditioner/physical readout；V/G 只差预测速度槽，D 与 O 其余部分冻结。新模块联合 AdamW 和 clip10，原 head 独立 AdamW/clip35；学习率、两个原损失及系数沿同源真实预检。模型保存 `future_pred_head`、`motion_readout`、`conditioner`、两个优化器状态、完整 sample orders 和来源链，必须搭配原 M0/O observer，不能当作独立完整模型。无恢复、最佳 checkpoint 或中途开发集选择。

empty object 集保留 native 路径。若某个 accum4 内 conditioner 完全不被调用，PyTorch AdamW 原生跳过其 None-gradient 参数；记录实际逐参数 step，核 V/G 完全一致，不伪称所有参数必有512步。物理 readout 保持原完整稀疏标签，不按 CRN 检测成功筛目标。第12/2044 microstep 的两个损失 VJP 只是来源连接记录，不用其数值改训练或选模型。

原 K/B 的 `gt_only`、`verify_common_denominators`、JSONL writer 直接保留，AST 已核相同。正式评价记录原 `O/V/G` 五时域完整 confusion、原8组共同指标与各自 t0；物理记录 `D/V/G` 的原每对象均值 EPE、组/有效标签/源点数。GT 分母相同，t0 预测不作跨模型等式。未来细分统计器需预先新建并绑定 `summarize_object_state_common_v1.py` 与 `summarize_object_state_physical_v1.py`；当前这两个分析源和预算尚未填写，草案不能执行。

除预检共有路径外，正式入口需要 `--preflight-protocol --engineering-preflight --dev-cache --raw-labels --o-development-records --train-predictions --development-predictions --selection`。原 `--training-run` 是用于加载 D 的已完成 J/D 目录。新目录要求 `--out` 不存在，所有预测输出只能在固定512更新后评价。

train 输入使用完整官方 train512 的已归一化预测，dev 输入使用另存的 `crn_state_dev200_centered_v1`，不重新运行检测器。开发读取器 `object_state_forecast_development_inputs_v1.py` 验证独立 centered schema、三文件 SHA、原 raw 三文件 SHA、原点适配/实测证明、完整200/100场景/94892框，**不再次加半高**。本机已对真实 centereddev 完整认证通过，回执 `object_state_forecast_development_inputs_v1_cpu_checks.json`；没有模型前向或分数。

`object_state_forecast_training_protocol_v1.draft.json` 当前为 REVIEW_REQUIRED，真实 centereddev SHA 已填；train512资产、真实预检 complete/protocol、两统计器及根据真实预检成本制定的正式资源仍待完成。暂未编造正式 argv 或 ETA。
