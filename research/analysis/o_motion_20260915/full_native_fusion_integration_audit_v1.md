# A/Z 全量原生验证接入审查

结论：可以复用原生数据加载与评价，在**一个冻结 O 模型**的同一观察边界上生成 O/A/Z；无需先制造全量状态缓存。接入代码与两锚点验证尚未执行，本报告不构成启动授权或性能通过。开发集 A 相比 O 的 +0.0865 pp 仍只是开发证据；是否运行 full 等共同指标结果再决策。

## 已有资产与缺口

- 固定 selection：`P/full_validation_selection_v1.json`，SHA `60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391`；5119 anchors、150 官方 validation scenes，按原 ordinal 顺序、自然每场景样本数量。原 ann SHA `83c1637453777f2ff02d5ae0daa719d008f1c7f450662b2ebe77074c35ee7eb9`。
- 已完成 full 位于服务器 `R_P/campaign_objective_joint_full_v2/`，本地镜像 `P/server_results/campaign_objective_joint_full_v2/`。2026-09-15 16:03 UTC 的有界只读目录查询确认：其 evaluation/samples 有 5119 个目录，抽查首尾均仅 records/complete。原 producer 本身也只写小记录，不写状态数组。
- 已知 `/home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915/` 只有 campaign_cache_v1/train512、development200 与 native_preflight_v1/train2。没有发现可用于 full5119 的原生状态缓存；这不是全服务器搜索。证据见同目录 `full_native_fusion_inventory_v1.json`。旧每 anchor 约192.7 MB的完整缓存若线性复制到5119约986.4 GB，没必要为一次 full 预先创建。
- 原始 nuScenes metadata、native radar readonly sweep cache、fine occupancy GT 均已有。**只做原任务 GMO 无需新框标签**；要同步运行共同 change/速度分组指标，需新增 full5119 原始框轨迹包，见下。

其中 P=`analysis/m0_improvement_20260915`，N=`analysis/o_motion_20260915`；服务器 R_P=`/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915`，R_N 为同根下 `o_motion_20260915`。

## 最小执行路径

1. 保留 `P/joint_native_evaluation.py:278–340` 的原始 Config/test dataset/`_selected_rows(selection,'validation',dataset)`/`_prepare_data_info(...,rand_interval=None)`/collate/identity 路径；5119覆盖、nsweeps=5、radar readonly、future_metadata_only 和原配置不变。无需调用其绑定旧模型集合的整个 run/CLI，也无需复制 objective v2 的九处 AST 协议框架。
2. 使用 `native_state_cache.build_native_model` 构造 M0 完整模型，按当前 trainer 的严格方式加载 O payload 的 `future_pred_head`。所有参数冻结、eval；一个 O 模型同时用于原生观测和后续评价。checkpoint/head digest 必须区分历史 tuple-shape 与 motion-helper list-shape 两个哈希域。
3. **精度边界必须保留。** 原 full 的图像/雷达观测是 matmul TF32=True；实际 O future replay 是 TF32=False、cudnn TF32=True。最少新增代码的路径：在现成 `_capture_native` 内做一次原生全前向，捕获进入 future_pred 前的克隆输入与原始 GT；随后对同一个 O 模型 `replay(...,training=False)` 一次，得到计分用 O logits。首次 TF32=True 的 O rollout 仅用于完成原生 capture，不作为 O 计分结果。没有第二个视觉 backbone 或第二个 O 模型。直接整模型 TF32=False 会改变观察特征；直接计分首次 rollout 会改变 O 的精度合同。
4. 复用 trainer 的 shared 数学：捕获的 `prev_bev_input[:,-1]` → 冻结 motion head；O coarse t0 的 softmax/class1>class0 → 预测支撑；A 使用预测位移、Z 使用 zeros_like。各自已训练97参数 gate 后，调用冻结 `compose_prediction`。所有 t0/intermediate/class0/未覆盖 class1 字节门原样保留；输入中不引入框、future label、fine GT 或实例信息，也不改变 O 内部 ego/SE(3) 路由。
5. 用同一个模型的原 `evaluate_occ_records` 对 O/A/Z 的原形状 logits 评价，同一完整 GT（原输入七帧，评价 t0+四未来）、原512×512×40、原0/1/255语义。共同 CPU 指标可复用 `fine_binary` 与 `evaluate_common_occupancy_change`，但其现有 final_dev200 CLI、cache/label总数绑定不能原样冒充 full。原生 fresh loader 本来会先加载 GT；准确声明应是 GT 不进入未来模型输入/融合计算，框 sidecar 只用于预测后的评价，不能照搬缓存版“所有 GT 在预测后首次读取”的描述。
6. O 每条五时域 confusion 应对齐历史 full O；两锚点工程检查还需对已有 dev cache 的有效输入/GT/O logits、A/Z own-dev hist 精确性。唯一已证明无效的 planning `sample_traj` 比较例外不能扩展到其他叶；保留原始输入，计划分支关闭。若换“只在未来函数内部切精度”的新 observer 以省一次 rollout，需独立两锚验证，当前不能视作现成已验证入口。

原满量聚合用全5119 confusion按h求总体比率，主指标为未来四h均值；150个自然大小场景 paired bootstrap 10000次 seed11。仍披露 t0、四h、FP/FN、历史 validation 暴露。旧 M0/C 结果可作为来源绑定的既有参照，不必仅为增加表格行重新运行其完整 backbone；A−O、A−Z是本轮直接同输入比较。共同 recall/change 改善仍不等同于真实位移 EPE 改善。

## 实际加载资产与依赖

- M0：`/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/assets/m0_epoch24.pth`，SHA `0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc`。
- O head：`R_P/campaign_objective_v1/runs/O/latest.pth`，SHA `ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70`。
- Motion：`R_N/source_motion_train_v1/final.pth`，SHA `449e37a18136f51fe3716f9cd0333a3a3888a5d4d6747c01b982c2dd0c9d2d84`。
- A/Z：`R_N/supported_fusion_train_v2/runs/{A,Z}/final.pth`；A SHA `d86457ea5a85a831071f93f37082e7957c011534c34f5e54e8ddc99744ce9dd0`，Z SHA `653b5a9635896a8765918957d7ba90b7fcd1a7356b2906558a20068b41103028`。它们是 gate payload，不能当完整模型。已镜像完整训练回执SHA `9aab1e899ab26b0ffa2803bdc6a5281ce81fe70ccaccbbf1befea8b793fefb4d`。
- 配置/环境/仓库沿历史实际 job state：`.../RadarFlowOcc_sota_p2_20260911/runtime/configs/S0.py`（SHA `c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303`）、`.../RadarFlowOcc_m3_h200/envs/hym_driveocc_m3/bin/python`、`/home/wangning/Workspace/RadarFlowOcc-sota-p2`。
- 数学与加载最小依赖：native_state_cache、train_source_motion_v1/state digest、learned_transport_probe_v1/load_motion、motion_prediction_head、transport_ops、supported_motion_fusion、train_supported_fusion_v2/compose_prediction、oracle_transport_probe 的纯 logits轴转换；验证最终 gate 可调用 common_change_evaluation_v2 的 validate_completed_fusion/load_gates。仅共同指标另需 common_occupancy_change_metrics_v1 与 motion_geometry。评测不调用 trainer.run，不恢复 optimizer，不安装训练 loss adapter。

## 若需要 full 框标签

旧 `build_motion_targets_v1.py:139–143` 硬编码 selection/train512/dev200；不能直接传 full selection。新薄入口只改变 selection/split/总数验证，复用其原始 sample链、LIDAR→global、annotation/instance抽取逻辑，输出新的 validation5119 manifest与逐token JSON.gz；保留过去2+t0+未来4、真实timestamp、原始 GMO instance_token、每帧mask/boxpose/visibility/pointcount。原 metadata 路径 `/home/wangning/data_cache/RadarFlowOcc_m3_h200/datasets/nuscenes_driveocc/v1.0-trainval`，官方 splits.py 已有。metadata只需一次流式读取，并验证150场景属于官方val。旧dev200数据只能按token作为一致性对照，不能把 development 身份字节未经改版直接宣称新 validation 身份。此阶段不需要 full sparse-motion NPZ，不读取预测选择样本，也不需要新的 GT refine。

## 成本与边界

历史同一 full 数据路径真实 outer耗时8528.0427s（约2h22m），两shard分别8394.85/8459.39s，各2560/2559条，实际完成后提前退出；它包含 M0/M0_fp32/C/O 四个模型结果。此前预算21600s/32GiB每卡是上限，不是预计耗时。新 O/A/Z 少了若干未来头重放，却增加 motion/splat/fine共同统计，不能据旧数字承诺更快；两锚点 fresh全链测出GPU/CPU/峰值内存再定预算。旧 raw712标签完整提取真实19.33s CPU可作资产准备量级参考，不能直接线性推定full时长。

本次只读源码、小回执和有界服务器目录元数据；没有读取大权重、重新推理、启动任务或修改任何冻结文件。

