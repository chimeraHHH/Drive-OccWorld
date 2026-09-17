# 当前与历史相机输入：已有资产和最低成本读取路径

本轮只读本地源码、冻结配置和既有完成回执，不连接服务器、不解包重型 tensor、不运行图像/模型/训练。**最低成本入口是原 nuScenes 五张 sensor metadata 表 + 当前/前一 keyframe 的 raw JPG，CPU 读取即可；无需重新计算 native BEV，也无需构建完整 dataset。** 源码可读不等于本轮已确认远端文件可用；实时文件存在/解码和路径映射由 root 的另一次只读检查负责。

## 1. 资产现状与准确位置

| 入口 | 已有证据 | 能复用什么、不能声称什么 |
|---|---|---|
| H200 raw 数据根 `/home/wangning/data_cache/RadarFlowOcc_m3_h200/datasets/nuscenes_driveocc` | [motion_targets manifest](../motion_targets_v1/manifest.json)及[complete](../motion_targets_v1/complete.json)；本轮重算 manifest SHA 与 complete 一致。五张表完整字节 SHA 和记录数见末表。 | 路径是既有回执记录。JSON 位于 `v1.0-trainval/{sample,sample_data,ego_pose,calibrated_sensor,sensor}.json`，图像路径为数据根拼接 `sample_data.filename`。本地只有派生回执/标签，不是这五表或 JPG 的完整镜像；本轮不验证远端现存字节。 |
| native train512/dev200 cache | [train index](../../m0_improvement_20260915/server_results/cache/campaign_cache_v1/train/index.json)、[development index](../../m0_improvement_20260915/server_results/cache/campaign_cache_v1/development/index.json)有每份 `inputs.pt` SHA，原 ann pkl 路径/SHA。 | `inputs.pt` 是 future_pred 入口，含当前 fused BEV、radar8ch、img_metas/prev_img_metas，没有 raw 图或逐历史帧 BEV。metadata 可用于身份/文件顺序交叉核验，不能从 BEV 反演图像。 |
| CRN train512 输入视图（L40S） | [manifest](../crn_train512_assets_v2_receipt/manifest.json)、[history plan](../crn_train512_assets_v2_receipt/history_plan.json)，本轮两 SHA 均与 complete 相符；12288 个 image filenames，2048 keyframes。 | 原数据根 `/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/radar_fusion_baselines_20260909/data_views/CRN`；新视图 `/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/crn_train512_state_v2/assets/data`。`samples/sweeps/v1.0-trainval/maps` 通过 symlink 复用，未复制原图。prepare 当时只做 `is_file`，**image_files 无逐 JPG SHA 或解码回执**；不能说图像字节已全部认证。 |

native ann 路径分别为上述 H200 根下 `nuscenes_infos_temporal_train_new.pkl`（SHA `45bff6a897f4835084eb287a7cf6c4cf2b6cf8330196e494987f512055080584`）和 `nuscenes_infos_temporal_val_new.pkl`（`83c1637453777f2ff02d5ae0daa719d008f1c7f450662b2ebe77074c35ee7eb9`）。CRN 是另一份作者 full-train metadata：新视图下 `nuscenes_infos_train.pkl`，801937487 bytes、SHA `c718c9f1e5520d5e083190c683c3995668912c1f9b29ba48bafccce1c473e9d1`。它不是 native ann，也不必为本次相机诊断加载这份约 802 MB pkl。

## 2. 最小安全读取接口：keyframe 与实际时间分开

复用 [build_motion_targets_v1.py:51](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/build_motion_targets_v1.py:51) 的纯 Python `table(path, ledger)`：1 MiB 分块读取 JSON array，完整消费后记录全文件 SHA、size/mtime 与 records_read。只调用此函数，**不要调用包含 annotation/未来链的 main**；它不需要 Torch/SDK。停止迭代会丢失完整 SHA 证明，必须遍历到 EOF，同时只保留需要的 token。

建议映射顺序：

1. 由固定 selection 的 anchor sample_token 在 raw `sample.json` 中定位当前 sample，取 `sample.prev` 为前一 keyframe。验证相同 scene、prev.next==current.token、时间递增；这对应 native 相邻 keyframe，仍应和实际 cache/ann 身份核对。raw `sample.json` 本身通常没有 `data` 字典；`nusc.get('sample', token)['data']` 是 SDK reverse-index 装配结果，不能直接用于裸 JSON。
2. 扫 `sample_data.json`，仅选择上述两个 sample_token 且 `is_key_frame==True`。通过 `calibrated_sensor.sensor_token → sensor.channel` 建立六相机和当前 LIDAR_TOP 索引。不要假设 raw sample_data 已有 SDK 添加的 channel/sensor_modality。每 `(sample_token, camera_channel)` 要唯一。
3. 保存各相机自己的 `sample_data.token/filename/timestamp/ego_pose_token/calibrated_sensor_token/width/height/prev`。按需求收集 pose 和 calibration，再读 raw JPG。**sample.prev 是前一 keyframe；sample_data.prev 是该传感器上一采集，常为非 keyframe，不能互换，也不能假定后一种 high-rate sweep 的图像已实际部署。** 本轮推荐 keyframe 路线。
4. 每张图的时间用其 sample_data timestamp（微秒→秒），不要套 sample 时间或固定 0.5 s。当前六相机也非严格同刻；记录相对参考 LIDAR_TOP 的 dt。输入“已观察”应明确以原 native 当前 keyframe 六图为边界，不能额外取 next/future；若严格以 LiDAR 时间为截止，则需单独检查六图是否都不晚于该时刻，不能事先假定。

如果已有 NuScenes 实例，等价接口是 `get('sample', token)`→SDK `data[channel]`→`get('sample_data', token)`→各自 `ego_pose/calibrated_sensor`；但新建 SDK 会加载更多表，CPU streaming 更省内存且更容易保持 annotation 隔离。`SensorOnlyNuScenes`（[prepare v1:105](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/prepare_crn_train512_inputs_v1.py:105)）可作调用白名单参考，不必运行 prepare。

## 3. 六相机顺序与 native 已观察历史

native [get_data_info:180](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:180)严格沿 `info['cams'].items()` 的持久化顺序生成 filename/K/extrinsic 列表。现有 [converter:251](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/tools/data_converter/nuscenes_converter.py:251)写入顺序是：

`FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT`（均有 `CAM_` 前缀）。

这是源码顺序，**最终逐项顺序应以已认证 ann 的 cams 或缓存 filename 列表为准**，不能仅按字母排序。CRN `gen_info.py:24` 的字典顺序为 `FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT`；而其实际 [4key config:109](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/exps/det/CRN_r50_256x704_128x128_4key.py:109)输入顺序又是 `FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT`。三者不可凭索引混用；新 CPU 图像证据按 camera channel 键关联最直接。

native 真实配置 `queue_length=2, memory_queue_len=1`，不是“只有一帧图像”。[模板:600](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py:600)按默认 `rand_frame_interval=(1,)` 取 `[index−2,index−1,index]`；[union2one:213](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_v1.py:213)只 stack previous_queue 的三个六图张量。[原 detector:998](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:998)先用两帧历史获得 BEV，再处理当前帧。名义时刻为 −1、−0.5、0 s，真实时间需 metadata。future_length=4 属于未来 occupancy/ego metadata，不是四幅未来图像；`future_metadata_only=True`。

CRN 的 key offsets `[0,−2,−4,−6]` 约为当前及 −1/−2/−3 s，含同场景边界回退，其 plan 已记录真实 token/time；不是 native 的历史队列。另 `gen_info.py:70–90` 的 cam_sweeps 虽记录 sweep filename/time，ego_pose/calibration 却取外层 keyframe `cam_data`：**不要复用它来做 actual sample_data.prev 的精确时刻姿态**。本轮不更改作者资产；raw keyframe 每条 sample_data 的自身 pose 可直接避开这个语义问题。

## 4. raw 图、native 网络图、CRN 网络图是三种入口

本轮实读 [S0.py](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py:564)，SHA `c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303` 与 cache config 完全一致。[native_state_cache.extract:491](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:491)对 train/dev 都复制 `cfg.data.test`，train 仅换 ann_file，故：

- **实际 native cache 图像流水线无 resize/crop/flip/photometric distortion**。加载 float32，mean `[103.53,116.28,123.675]`、std 全 1、`to_rgb=False`，再 pad 到 32 倍数。原 900×1600 图通常得到 928×1600；padding 不是有效图像区域，也不改变原 K。别把 S0 的 train CropResizeFlip 配置套在这些缓存上。
- raw JPG 可直接用 PIL/OpenCV CPU 解码。PIL 默认 RGB 与 native BGR/mean 顺序需区分；做新的 photometric 诊断可固定一种颜色/灰度约定，但不能宣称字节等同原网络 tensor。若另行 downsample，要对像素坐标/K 应用完全相同 resize 变换。
- CRN 的 eval `sample_ida_augmentation()`（[dataset:237](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/datasets/nusc_det_dataset.py:237)）是 raw 900×1600 → resize 0.44 → 396×704 → bottom crop `(0,140,704,396)` → 256×704，eval flip=False/rotate=0；其 `ida_mat` 才与变换后图像配套。不要拿 raw K 投影进 CRN crop 图，也不要拿 native padded 坐标当 CRN 坐标。

cache `CustomCollect3D`（[transform_3d.py:243](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py:243)）保留 filename、lidar2img/lidar2cam/cam2img 和 sample/scene 信息，但默认 meta_keys 没有每相机实际 timestamp，因此 temporal dt 仍应来自 raw sensor metadata。`native.load_sample` 会同时读 targets，不适合纯输入诊断；若仅需核 cache metadata，可参考 [load_tokens:215](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:215)的文件 SHA 后 CPU torch.load/validate_inputs 路径，但它仍需兼容 Python 类，不如 raw JSON 省依赖。本轮未解包实际 inputs。

## 5. 与 D 固定 R 坐标对齐

设 R 为当前 LiDAR，`G0 = T(global←ego_L0) T(ego_L0←L0)`。相机时刻 τ 的投影为：

`p_Cτ = inverse(T(ego_Cτ←camera)) inverse(T(global←ego_Cτ)) G0 p_R`，再以该 calibrated_sensor 的 raw K 作正深度投影。

每个相机使用其自身 ego_pose；不要共用 anchor pose，也不要把列向量公式和 native info 中的转置/行向量代码混搭。[obtain_sensor2top:342](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/tools/data_converter/nuscenes_converter.py:342)保存原 sensor timestamp 和各自 pose，并构造 sensor→当前 LiDAR 变换，可作为源码对照。所有旋转保留完整 quaternion；图像宽高按实际解码核验。过去相机的匹配使用过去 pose，**不应读取 future2ref transform、annotations、未来 boxes 或未来图像**。只做静态坐标投影并不完成运动补偿或遮挡判断。

## 6. 本轮证据与仍需现场确认

已核本地配置 SHA、raw manifest→complete、CRN manifest/history_plan→complete 的链，并只读了列出的实现。没有验证五张 raw JSON/JPG 当前是否存在于任一服务器，没有 decode 图像，也没有逐点投影/时间差的实际数值结果。H200 当前/前一 keyframe 全六图文件可用性、raw token/channel 唯一性、实际相机顺序与 native ann 的一致性，仍由 root 当前的实际检查完成；尤其 CRN 12288 文件存在回执不能替代 native 所有前一 keyframe 或高频 sweeps 的可用性。

五张 sensor 表的既有完整文件回执如下；这是历史认证值，本轮不冒称重新读取原表：

| JSON | bytes | records | SHA-256 |
|---|---:|---:|---|
| sample.json | 7424233 | 34149 | `6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d` |
| sensor.json | 1185 | 12 | `4d5c96570e2d8b09b88ce4c605e40ee43c4909f97bea5741185f18f92eb491ae` |
| calibrated_sensor.json | 3268133 | 10200 | `67781a5dd7b2504b046ef89d6dcb267b12d1cc91af21f1ec5758624588e99865` |
| sample_data.json | 1345678379 | 2631083 | `6dcad49f0b9bd7b1cef04e2a0ff2ac2879b46b938e8d48153f00693cebb4de21` |
| ego_pose.json | 645746265 | 2631083 | `be12bd501f694b344628ba0680a37d6e1ec83e62b37f310c205c1dbe41cd09ff` |

本轮关键本地来源指纹（不声明未纳入旧 runtime contract 的文件是历史实际运行版本）：

- `analysis/sota_p2_20260911/configs/S0.py`: `c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303`
- `analysis/m0_improvement_20260915/native_state_cache.py`: `41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0`
- `analysis/o_motion_20260915/build_motion_targets_v1.py`: `0064b3bb3dd3d8a124e49524ae848e0af1c556e12850947577280e988a7f2685`
- `code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`: `1e641ad8e1108ab01d3225df6dfe696e918674cf72c81036720e62f9cdcf95ef`
- `code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py`: `9de0dfbba0b77567cf011fb0adb163492e8d5e6f510040b95626b3bcded822cc`
- `code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_v1.py`: `c54dd6823a736ef283fe213771d563833975df6076756d6831f96718ff143923`
- `code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`: `b211570e173600874edcc8bd0ed11286a76a8d5bb1f2ca56a1227c90eb4420b1`
- `code/Drive-OccWorld-sota-p2/tools/data_converter/nuscenes_converter.py`: `0a88ebf1a6f25bd7c1072f7689a82b992ed14a73d50b50b0397cc6a3a5df2b82`
- `code/radar-fusion-baselines/CRN/scripts/gen_info.py`: `c65bd94133a987855d1f63e82d9f03c3a66084135f7e67aebab07b4b76ef1ba8`
- `code/radar-fusion-baselines/CRN/datasets/nusc_det_dataset.py`: `45651e062f2258092f18720890ff7eff77c3f71664d751e04e865e9f55927ed3`
- `analysis/o_motion_20260915/motion_targets_v1/manifest.json`: `4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71`
- `analysis/o_motion_20260915/crn_train512_assets_v2_receipt/manifest.json`: `0fcb6ef052155731ed4cf27119f4d951afc89d2b1bf269c5e8c88bc03b55e042`
