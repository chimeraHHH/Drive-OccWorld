# 固定 D 在 CRN 覆盖内外的互补性：可执行性核对

2026-09-16；**仅源码与已完成回执审查，未执行推理，未改变 Cpl/Fix。**

可做，而且无需重跑 O、视觉主干、CRN 或几何构建。D 是独立的当前 BEV→四时域稠密位移头，现有原生 `inputs.pt` 与完整几何缓存足够。现有逐对象 D 均值无法识别覆盖内外误差，必须重放这个固定头取得逐点预测。当前没有这项实测结果。

## 最小调用链及真实资产

以下远端路径来自已镜像的原 `connected_motion_train_v2/launch.json` 与 `shared_rigid_geometry_cache_v1/launch.json`，本次没有 SSH 确认其当前在线状态。记 `RN=/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915`。

1. 使用原运行环境 `/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/envs/hym_driveocc_m3/bin/python` 与已冻结依赖。调用 `train_connected_motion_v2.load_completed_arm(RN/connected_motion_train_v2/runs/D, connected_motion_protocol_v2.json, helper, device)`，返回 `(D, unused_gate, receipt)`；删除未使用 gate，D 保持 `eval()`、全部参数冻结、`no_grad()`。**该严格加载器还会校验 J/D 双方全部文件 SHA，因此 J 的原 `final.pth` 也须仍在原目录；只对 D 载入张量，不恢复优化器。** 它不是早期 source-motion 预训练头。
2. `helper.cache_index(dev_cache, 'development', protocol_dict)` 读取原 200 条序列（`protocol_dict` 是上述已认证 JSON 的内容）；`helper.load_tokens(dev_cache, record, native, device)` 只读并认证 `inputs.pt`，检查 `prev_bev_input` 原始 SHA，再取最后一帧 `[1,40000,256]`。`dev_cache=/home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915/campaign_cache_v1/development`。`motion_prediction_head.MotionPredictionHead.forward` 输出 `[1,4,3,200,200,16]`，米单位、XYZ C-order，名义时间 `.5/1/1.5/2 s`。**不需要原始 O/M0 权重、O forward 或占据 gate。** `inputs.pt` 含原生类对象，需沿用能导入这些类的原环境，而非假定纯 Torch 环境足够。
3. `GeometryCache(RN/shared_rigid_geometry_cache_v1, complete_sha, selection_path).load('development', ordinal, identity)` 读取原 `owner`、`owner_original_indices`、`cv_velocity_R`。它认证完整 712 缓存、分集顺序、每个 NPZ SHA/数组字节；`owner>=0` 就是预先固定覆盖切分，禁止换成 O 前景、GT 框匹配或事后误差选择。owner 是当前预测框定义，不随未来姿态更新。CV 位移沿用 float64 原场乘名义 h，不从 float32 packed states 重建。
4. **完整 D/CV 场及 owner 先产生，之后才读 sparse 标签。** 用 `helper.load_sparse(...)`、`helper.gather_sparse(...)` 对同一原 `source_flat_indices` 采样 D；相同索引采样 owner/CV。先用 `helper.epe_records` 重现旧 D 的全部对象误差，再额外保存逐点误差或足够重算的子集和/计数。GT 只用于 valid、对象归组和误差，不参与预测或选择分支。

关键 SHA（本次读取本地源及历史回执核对，未重读远端权重）：

| 资产 | SHA-256 |
|---|---|
| `RN/connected_motion_train_v2/runs/D/final.pth` | `7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3` |
| D 实际 motion state（已有严格张量加载回执） | `4abd5bf8aab3fff045d4ea9f4c5d7bc3579f7265f795b4aef87eb5774afadb61` |
| 原训练 `complete.json` | `f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56` |
| `connected_motion_protocol_v2.json` | `e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259` |
| `train_connected_motion_v2.py` | `1df18b52c9e00f168c749384546c5818e6e5abbcc093c84ca23a63086d29e84a` |
| `train_source_motion_v1.py` | `e34efdfd6dca7b60a8a4ddb3359417bbe5ba9e7b04a4b150e8b6428b81dd582e` |
| `motion_prediction_head.py` | `f5e50b7486b5285092ca6523f305a5d643a73355c73382c8eaa24ab928a847b9` |
| development 原生缓存 `index.json` / `complete.json` | `fd42d2e511d754755fa5e2c06527077351869f06e868ecb8acb4404eadaa26c3` / `0128d88e9ed04b85c5bd86a00fc5e273199afe5cedb6fea6586080e6d48b8a84` |
| 完整几何缓存 `complete.json` | `a2504a0389cb2756032531961e305e22287ec890c96085e5ea29b2e61e4f1c99` |
| sparse `manifest.json` / `complete.json` | `cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe` / `3d14a03cd61761a1ebc26313463dbff06854907c19c49057667a0ffd257e05bd` |
| 原 D `development_objects.jsonl`（还含 P/J，须按 arm=D 取原条目） | `b73499b8e4ce1bd0de6ffc3ff6001d3a58ab9f33483c8bfb8f7ed68e6f88d0ea` |
| 原 CRN-CV v2 `summary.json` | `ba67afc75b02ef481b0846914207fc01c6d04a4d0c675d3a7c00d1129b81141e` |

原 `selection_v1.json` 在 `analysis/m0_improvement_20260915/`，SHA `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d`。D 的既有实载证明是 `server_results/training/common_connected_motion_dev200_v2/loaded_models.json`，SHA `c8b1a01f061e2866ba7232392e0bcc26fff3199efe0b047df482a51e40fb0a96`。

## 预先固定的统计，不以检测成功缩小主支持

保留 200 anchors／100 scenes、全部 16,074 个原对象×时域条目，原 valid、dt、stationary/ambiguous/moving 分组不变；四时域均报告 XY/XYZ。原始目标是虚拟刚体材料点位移，不是实测 scene flow。

每对象 i、子集 s∈{covered,uncovered}，记录有效点数 `n_is`、D/CV/zero 的逐点欧氏误差和与条件均值。主要分解使用 `contribution_is=sum(error_is)/n_i`，再对**原全部对象**等权平均；这样两个贡献严格加回原对象平均 EPE。空子集条件均值为 null，贡献为 0，不能将两种量混写。另列条件子集均值/中位数/p90时，明确仅对该子集非空对象等权及其对象/点分母；不能拿它与完整支持结果直接比较。

可同时描述唯一预先指定的固定组合：**covered 用 CV，uncovered 用 D**；分支只接当前 owner，对任意源位置有定义，不按 GT、未来速度组或误差选优。其完整 EPE 等于“CV covered 贡献 + D uncovered 贡献”。这是固定已有读出的互补性诊断，不是学习方法、不是对 Cpl 的修改、不是新占据结果。没有必要加逐点 min(D,CV) 的 GT 选支 oracle。

必须复现：①旧 D 每个对象的 key/scene/dt/group/点数/zero 和 XY/XYZ EPE；②全四时域 4,189/4,076/3,960/3,849 个对象；③几何缓存 `owner_original_indices[source_indices]` 与原 CRN-CV `per_anchor_coverage` 的 owner digest/覆盖数；④旧 CV 每对象 EPE与 coverage 条目；⑤D/CV 分区加回全支持均值。保留输入、标签、模型前后 digest及有限值记录。不把潜在数值差直接当互补性；先在原精度下解释复现差异，不先改容差。

## 可改变的判断与未确认项

若 D 在未覆盖处比零位移低误差，且没有用少量大物体/小子集掩盖其他组损失，就说明**这个固定观测 readout 在预测框未覆盖的既有合法源点上有可利用的信号**。固定组合的全支持净收益量化其价值；不说明 Cpl 已学到该信息、不保证可训练融合有效，也不能证明新生/完全不可观测物体可恢复。若不优于零，只否定当前固定 D 的这项互补性，不否定原传感器或其他表征含有运动信息。

最小后续资源测量是既定 dev 前两条的严格加载、tokens→D 和 cache I/O 计时/峰值内存，再决定一次固定 dev200 的诊断预算；不得把旧原生 O+多模型评价时长当作本诊断 ETA。沿原 FP32、matmul TF32=False、cuDNN TF32=True、benchmark=False 执行才适合复现历史 D；本次未测试 CPU/GPU 重放、未确认远端大权重在线、未取得逐点 D 记录，未部署或启动任何任务。
