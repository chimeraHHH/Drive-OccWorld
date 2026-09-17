# V/G 原生完整验证资产盘点（条件方案，未执行）

2026-09-16。本次只读本地资产；不接触服务器、不下载、不运行模型、不读取 V/G 中间开发分数。只有正式开发结果支持继续、根任务作出决定后，才实施以下缺口。完整 SHA 和本机逐文件核验见同目录 `object_state_full_validation_asset_inventory_v1.json`；以下 SHA 缩写仅用于索引。

**现成数据足够支持原 5119/150 任务；尚缺 V/G 完整推理接入和 CRN full 状态子集，不能直接把旧 R 或 K/B evaluator 改个模型名运行。**

## 样本与覆盖：已实际核对

- 原 `../m0_improvement_20260915/full_validation_selection_v1.json` 是 **5119 anchors / 150 scenes**，official_index 连续 0…5118；catalog、原 full NPZ 的 token/scene/index 顺序完全一致。没有 5129 的合同或数据依据。
- 旧 CRN 源 manifest 的 **6019 个 expected_tokens 包含全部 5119，缺 0、多 900**。本机解压全部原框链后，七帧 token 并集恰等该 6019 集合，每 scene 比可用 anchor 多 6 个边界帧；这解释两套任务数量不同，不应按行号对齐或额外扩大占据评价集。此处依据是本地源预查回执内的 expected_tokens，加上绑定原 export SHA 的历史完整扫描；**本轮没有重新读取远端 1.31 GB export**。
- 原 dev200 的 200 个 token 全部在 full 中；其 100 个场景在 full 中含 **3412 anchors**，另外 **50 场景 / 1707 anchors**。自然场景大小为 33–35 anchors。额外 50 场景仅在本轮开发子集之外；M0/C/O 旧 full 已评价全部 150 场景，**不是未曝光或盲测集**。

## 可直接复用的实际资产

路径以本目录 N、兄弟目录 `../m0_improvement_20260915` 为 P；完整路径及 64 位 SHA 在 JSON 中。

| 资产 | 实际文件 / SHA 前缀 | 可复用内容与限制 |
|---|---|---|
| 原 full 身份 | P/`full_validation_selection_v1.json` — `60811c8a0847` | 保持全部 5119/150、原顺序与自然 scene 权重 |
| 原 M0/M0_fp32/C/O 参考 | P/`server_results/campaign_objective_joint_full_v2/summary_v1/`；complete `cdf54175b846`、raw NPZ `4f649ede18d4`、ledger `cc5fde3022bd` | 本次重新核全部 complete 文件 SHA；NPZ 为 `[4,5119,5,2,2]`，另含七帧 GT counts。只有统计和身份，不能代替新模型前向或三维预测缓存 |
| 原始七帧框链 | N/`full_motion_targets_local_v1/`；manifest `48ac12f1ce24`、complete `de10ddd9013d` | 原始 annotation/instance、真实 dt、G0/各帧 pose；本次全部 5119 gzip SHA、解压 SHA 和身份通过 |
| 稀疏运动 GT | N/`full_sparse_motion_local_v1/`；manifest `59535fe4f21e`、complete `68fc77c6fe47` | 本次全部 5119 NPZ SHA 与 raw 来源绑定通过；旧 dev200 科学字段 dtype/shape/bytes 交集审计 `25d53489439b` 已存在 |
| 官方 CRN 导出 | 历史 `/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/radar_fusion_baselines_20260909/CRN/attempt01_full/results_nusc.json` — `fec6541461bf` | 原官方 6019 帧 / 2,826,237 框；原 checkpoint `f725aafc7f03`、commit `5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d`。本机只有 dev 子集与完整源扫描回执，未保存整个原 export |
| 原 O 流与真实两锚预检 | N/`native_full_o_stream_v1.py` — `3a8d338de78c`；N/`server_results/training/native_full_o_stream_preflight_v1/complete.json` — `c35bbe651fe0` | `NativeOStream.fetch/verify_frame/finish` 可复用。后者是 job 名，没有另一个同名 preflight.py；不存在已生成的 full dense native feature cache |
| V/G 前向构件 | `object_state_conditioner_v2.py` — `dad88c05105d`；`future_state_motion_v1.py` — `96e796dd5bb6`；`object_state_forecast_train_v1.py` — `2841aeaa533b` | 复用已冻结 pack27、4h conditioner 注入、native terminal capture 和物理 readout；不能用训练进度或静态源码代替最终权重完成回执 |

原 raw 标签远端为 `…/o_motion_20260915/full_motion_targets_v1`，sparse 为 `…/o_motion_20260915/full_sparse_motion_local_v1`。历史镜像/上传回执分别结束于 2026-09-15 16:55:12、17:03:40 UTC；本轮确认本地资产，**没有再次验证远端可读性或 H200 对原 CRN 路径的挂载**。

物理支持是固定 t0 R、米单位、200×200×16 体素中心的**唯一当前框刚体材料点代理**，不是实测 scene flow。全量保留 102,899 个 anchor-object，共四时域 **382,770 个有效 object-horizon**；27 anchors 没有合法源，23,223 个当前框没有唯一格点，均须披露。future-valid 对象数依次为 100059/97244/94293/91174；未来出 ROI 仍保留误差，未检测对象不能删。新生/缺 t0/无唯一源本就不在该 EPE 支持内，但原 fine GT 与共同占据八组评价继续保留它们。原 fine GT 仍来自 native dataset，不以框标签替换。

## 真正缺口与最短接入路径

1. **CRN full5119 centered 输入资产尚未生成。** 从既有 export 仅按上述 token/顺序提取全部原框；不新增 score 阈值。绑定 export/selection，使用冻结 `crn_box_origin_adapter_v1.py` **一次**把原 global bottom center 转为几何中心，不能重复加半高。再用 `current_states_numpy(boxes,G0)` 生成八类状态；只允许当前 G0，不能读取未来框作为输入。dev200 centered 文件存在，但不能冒充全量。现有训练/开发输入类硬锁 512 或 200、开发 raw ordinal=512+i，需要新的 full 身份薄入口。
2. **最终 V/G 三部件 loader 尚缺。** 真实 trainer 的 `final.pth` 包含 `future_pred_head + conditioner + motion_readout` 及训练证据，不是完整独立模型。须从原 M0+O 构造冻结 O 模板，复制两个独立候选，strict 加载三个部件；D 另走既有严格加载。旧 `load_future_state_motion_v1.py` 只接受 K/B、两个部件和两个优化器，不能复用入口。V/G 新模块存在合法空对象跳步，不能套用“所有 conditioner 参数必有 512 optimizer steps”。本轮未读取候选终态或大权重。
3. **full consumer/collector 需换模型连接，数学可复用。** 每 anchor `NativeOStream.fetch` 一次 → 原 O 输出及 t0 BEV → 冻结 D 一次 → 同当前预测状态分别注入 V/G 各自 native future head → 各自占据直出及共享未来特征上的 physical readout。保持观测 matmul TF32 on、future off、cuDNN on；保留 stream 的原 O hist/GT 身份核对。不能套旧 R 的共同 O future features、输出残差、固定 t0、无 EPE 声明。V/G t0 可变，transition 两端都按各自预测；共同项是 GT/ignore/物理支持。原 dataset 会先读 GT，但 GT 与预测输入分离，不能声称从未提前读取。
4. **full D/V/G/CRN-CV 同支持物理记录尚未产生。** 旧 D、CRN-CV 数字只覆盖 dev200，不能外推或复制为 full。复用原 `gather_sparse/epe_records` 和 CRN-CV 场定义，给全部合法点计算；每对象点均值后对象等权，四时域/三组/XY/XYZ mean、median、p90 全保留。占据仍逐时域池化后四 future 平均、t0/FP/FN/八组/transition；以 150 个自然大小 scene 配对 bootstrap 10k seed11。现有 V/G 汇总入口硬锁 200/100/16074，须另立 full 收集入口，仅复用其无状态数学；不能只改样本计数。

## 最小下一步资源测量（仅条件建议）

先 CPU 提取/认证 full CRN 状态，不重跑检测器；随后若决定推进，用已有两锚 official_index **2036、2998** 运行**完整消费者路径**：真实最终加载、O/V/G 三次 future 输出、D/V/G/CRN-CV 同点误差、原 fine/common 指标及序列化。记录冷启动、逐锚观测/各头/状态打包/CPU common 与 EPE/写盘耗时、allocated/reserved/RSS；复用旧 cache 和最终 dev 交集核对，丢弃前一锚 tensors。两锚可验证接线和初步资源，若不足以确定稳定吞吐，再作事前固定的小热身段，不按性能选择样本。

旧 O-only 两锚真实预检为外层 **72.435 s**，内部初始化 **42.048 s**、峰值 allocated **4,448,122,880 bytes**；原四模型 full 外层 **8528.043 s**。它们仅是历史成本参照，不含这次三部件候选与全部物理评分，**不是 V/G full ETA 或预算授权**。当前不新建 full 协议、不派发任务，也不为负候选自动扩规模。
