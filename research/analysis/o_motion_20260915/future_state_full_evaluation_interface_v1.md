# K/B 原生 5119 验证：条件接口与已有资产

2026-09-16。**这是在正式共同结果支持继续时可采用的接口方案；本备忘没有执行 K/B full、读取其中间性能、加载其最终权重或启动任务。** 只读现有源与历史回执，不改冻结文件。下面的既有耗时不是 K/B ETA，也没有形成新的资源授权。

## 最短可复用源码路径

```text
NativeOStream.fetch(official ordinal)
  原始 dataset → 冻结 O 的原生观测前端（matmul TF32 on）
  原全量 GT / measured BEV / native radar 身份与 SHA
  冻结 O future replay（matmul TF32 off）→ 原 full 五时域×三层 logits/hist exact
  frame = {sample(inputs, targets, record), O_prediction, t0_bev, audit}
       ├─ K 自己的 native model → capture_native_terminal → K occupancy + K features
       ├─ B 自己的 native model → capture_native_terminal → B occupancy + B features
       └─ 冻结 D(t0_bev) 一次 → 两臂各自 motion_readout(features, D)
  预测完成后：同原 fine GT / raw box labels / sparse physical labels
  原 native evaluator + common metric + gather_sparse/epe_records
  stream.verify_frame(frame) → 释放该 anchor → 下一 anchor → stream.finish()
```

- 原流实现在 `native_full_o_stream_v1.py` 的 `NativeOStream.fetch/verify_frame/finish`，SHA `3a8d338de78caae5d73e11012cd9cec38a47621913ddbb0bbda9cfae6e510bef`。没有独立的 `native_full_o_stream_preflight_v1.py`；后者是该源 `main()` 的历史 job 名。这个流可保持不变：每 anchor 观测前端只跑一次，没有写出新的 dense full cache。
- `stream.model` 必须始终保留冻结 O，供 fresh observation 和原 full 对照使用。不能在它上面加载 K/B 头再沿用 O 的 SHA/unchanged 声明；另外构建两个独立 native 实例，非 future-head 状态保持原 O。
- 在旧 `full_future_feature_residual_evaluation_v1.py:245` 附近替换 R/R0 的消费者路径：去掉共同 O terminal-feature capture、hard source mask、splat 和 `compose_prediction`；改为 K/B 分别对同一 `frame.sample` 调用 `future_state_motion_v1.capture_native_terminal(..., training=False)`。两臂的 occupancy 都直接来自各自原生预测，不做末端残差修正。
- 原观测 TF32-on、future replay TF32-off、cuDNN TF32-on 的数值边界保留。`fetch` 返回后已经设为 future 精度；新消费者不应无意改掉它。所有模型 eval/no-grad；B 的 detach 在推理中不改变数值。

## 最终加载与评价中必须换掉的旧假设

正式 `future_state_motion_train_v1.py:350` 的真实保存结构是 `future_pred_head`、`motion_readout`、两组 `optimizers`、`update=512`、`examples=2048`、样本顺序及初始/最终 state SHA。diagnostic_runner 已落盘待验的薄加载器 `load_future_state_motion_v1.py`，拟采用接口如下：

```python
authenticated_training(training_run, protocol_path,
                       selection_path=None, o_reference_path=None)
load_completed_arm(training_run, arm, protocol_path, helper, native_O_model,
                   device='cuda:0', selection_path=None, o_reference_path=None)
# 第二个接口返回 (deepcopied native candidate, motion_readout, receipt)
```

两个可选路径在源码中是 keyword-only 参数。它的意图是先认证完整训练终态，再从冻结 O 模板复制独立模型、strict 加载该臂 future head/readout，并返回实际 state/文件来源回执；推理不创建或恢复优化器，只读核对训练终态证据。D 继续用旧 strict loader 单独加载。此处仅核对已落盘的函数接口，**没有实际加载 K/B final.pth，也不以接口检查代替未来的真实加载验证**。

保留冻结 O 历史全 logits/hist 精确核对；在 full 与 dev200 交集上，K/B 各自最终 hist 也应复现正式 dev 记录。旧 R evaluator 的 `value['t0_boundary'] == O` 与 `all_t0_models_exact` 不适用：K/B 的原生 t0 decoder 已训练，须保留各臂自己的 t0 hash/混淆矩阵，仅核共同 GT、ignore 域、八组支持及 transition GT 行和。transition 两个预测端点都可变化，不再把差异解释成固定 t0 下纯未来效果。

K/B 的 head 身份核对各自最终 SHA；原 `O_HEAD_SHA` 只用于冻结参考 O。旧 R 的 module-only loader、共同 O features、固定 t0 和“没有 physical EPE”声明都不能直接继承。新 K/B EPE 是框刚体材料点代理误差，O 仍没有 native flow。

## full 标签已经存在

| 资产 | 本地目录 | manifest SHA | complete SHA |
|---|---|---|---|
| 原始七帧框链，5119/150 | `full_motion_targets_local_v1` | `48ac12f1ce24ed1639c17aed0fcdc8ad2eef2cceff10c181b2a0f82f81fc90bb` | `de10ddd9013d7efd68249bced50448fb083c6d059546f8cfa0560be02b702252` |
| 同原规则的稀疏材料点标签，5119/150 | `full_sparse_motion_local_v1` | `59535fe4f21e73182da5090e6b12ec60a0dfb40382564f0bb16409565699fc4e` | `68fc77c6fe474fe5ed722e07be0341731a15a3adbb215dc5ecba453e06fec545` |

两者绑定原 full selection `60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391`。原生 fine GT 仍由原 dataset 提供，框标签不替代原 GT。观测 loader 本来在预测前读取 GT，但它与 `sample.inputs` 分离；不能改写成“预测前没有读 GT”的不实声明。

稀疏标签的 dev200 交集已逐 dtype/shape/bytes 相同，仅 `split/ordinal/label_source_sha256` 三个绑定字段变化。对应审计 SHA `25d53489439b35a526810d7bc42a1bd9a582e8951254249b38451f63d19cef06`。原 `helper.load_sparse/gather_sparse/epe_records` 可直接处理 `validation` 身份和同 XYZ 顺序；`labels_manifest` 则硬锁 712，不能复用其入口。D/K/B 用同一合法源、future-valid mask、object index 和实际 dt；未来出 ROI 的目标仍保留，不使用 O hard foreground 筛掉评价对象。

稀疏标签已上传至 `/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915/full_sparse_motion_local_v1`。历史上传回执的时间为 **2026-09-15 17:00:11—17:03:40 UTC**，声明已核全部 5119 NPZ SHA；回执 SHA `2ea1ad3c52a8b704de01776cb55f720caacc8079857f2cae6609f98aa82381f4`。本轮只读这份回执，没有重新 SSH 验证远端状态，也没有重扫全量标签。

我新写的 dev 汇总 `collect/check_manifest` 固定 200/100 和旧开发标签 manifest，不能作为 full 入口。其 `aggregate_counts/aggregate_physical` 纯数学已经按传入 scene 列表计算；full 只需新收集器绑定原 5119/150 身份、全量 manifest 和 shard 完整性，再以 150 个自然大小 scene 做配对抽样。没有必要重写 count、EPE 或 bootstrap 公式。

## 已有资源证据与真正的不确定性

历史 O 流两锚 job 开始于 **2026-09-15 17:15:03 UTC**，结束于 **17:16:15 UTC**，外层驻留 **72.435 s**；内部摘要为 **56.199 s**，其中模型初始化 **42.048 s**。热的第二锚 `fetch` 为 **2.223 s**：观测 **1.832 s**、O replay/evaluation **0.136 s**；旧 cache parity 等检查后该锚共 **3.259 s**。峰值 allocated **4,448,122,880 bytes**。这些只证明原 O 数据流的两锚工程路径；不包含 K/B 两个最终头、辅助读出和完整 common/physical 评价。

该历史 summary SHA `3b2a7c87cf51e0547a6fccfac055a7434e11d4ce2209c193e2b5825e934b5a62`，complete SHA `c35bbe651fe0f28f7d8bbd2c7cf02f50dae893b2bba356cf4931beb26dcebe43`。更早四模型原任务 full 的外层 **8528.043 s** 见 `../m0_improvement_20260915/full_objective_independent_audit.md`；它也不是 K/B 成本预测。

K/B 相比旧 R 路径省掉一次共同 O future capture，但增加两次各自 future replay；前端仍一次。全分辨率三模型 CPU transition/速度归因、D/K/B 的材料点 EPE、三套模型常驻与逐对象 JSON 量都会增加成本。冷启动和两个锚不足以给完整 ETA；后续若结果支持 full，应实际测这条完整消费路径后再定预算。当前不为 full 创建新协议、任务或资源上限。
