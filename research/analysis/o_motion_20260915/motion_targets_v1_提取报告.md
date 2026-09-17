# 原始七帧运动标签交付

已完成固定 **512 train / 200 development anchors** 的原始 nuScenes box-chain 提取并全部回传。服务器 CPU 提取实际 **19.32 秒、峰值 RSS 506 MiB**；712 个 gzip JSON 共 **10,363,273 字节**。未加载模型、权重、原生特征缓存、预测、图像或点云，未启动 GPU/训练，也未改旧配置、标签或缓存。

本地目录：[motion_targets_v1](motion_targets_v1/)。服务器目录：`/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915/motion_targets_v1/`。逐文件 SHA 和原序见 manifest；生成 complete 之前已完成全部712文件写入、解压回读验证及metadata未变化检查。

| 认证文件 | SHA256 |
|---|---|
| `build_motion_targets_v1.py` | `0064b3bb3dd3d8a124e49524ae848e0af1c556e12850947577280e988a7f2685` |
| 原 `selection_v1.json` | `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d` |
| `motion_targets_v1/manifest.json` | `4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71` |
| `motion_targets_v1/complete.json` | `e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69` |
| `motion_targets_v1/statistics.json` | `e99825503dfe8f025ce4ef5ca19f1b986d047afff778123a378cf532cd516f61` |
| `motion_targets_v1_local_audit.json` | `6fd520e1131b4122f4eccaa5c2da33243be94a22fdaefa4fc69cd8a3d386ce7a` |

来源是既有nuScenes目录中的11个原始JSON表；流式读取sample_data、ego_pose、sample_annotation并只保留所选sample关联项，**完整文件全部字节均计算SHA**，没有反复完整反序列化。manifest记录表路径、字节数、mtime、记录数、SHA。官方train/val划分从已安装SDK的split源码字面量解析，不导入SDK或sklearn。metadata匹配此前两样本审计的所有小表完整SHA；大型表完整SHA由这次绑定源码的服务器提取给出，本地未再次读取大型原表。

## 文件与读取约定

每anchor路径为 `<split>/<sample_token>.json.gz`，schema=`raw-nuscenes-motion-target-v1`。可用标准 `gzip.open(path, 'rt')` + `json.load` 读取；没有pickle或可执行对象。

- `identity`：原selection条目；`ordinal`：原712条全局顺序，train先512、development后200。
- `frames[7]`：sequence_index=0…6、relative_frame_index=−2…4，sample/scene token、原 timestamp_us、相对t0的 dt_seconds、LiDAR sample_data token/时间、sensor与ego pose token、原外参/ego平移四元数、`lidar_to_global_column_matrix[4][4]`。**t0是第2项**。
- `tracks`：七帧中出现的全部GMO instance_token并集，按token排序。每项有category与原fine class；`valid_mask[7]`；`annotation_tokens`、`global_centers_m`、`sizes_wlh_m`、`global_rotations_wxyz`、visibility/point-count/attributes、原annotation prev/next等各长7的数组。缺失帧对应元素全部null；不复制旧框、不平滑、不插值。
- 不按ROI、当前/历史可见性、传感器点数、速度或预测筛实例。因此未来新出现实例、t0缺失实例、域外框也存在，使用者必须显式定义监督覆盖；不得把null填成零速度或有效框。

category严格采用官方16类映射后原fine GMO `[2,3,4,5,6,7,9,10]`，包含construction；pedestrian只保留其映射至原类7的四个子类，排除原ignore子类、barrier和cone。完整12项映射在manifest中。[官方类别对应](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md)

`global_rotations_wxyz` 为原box→global姿态；原 `sizes_wlh_m` **是w,l,h**，做box局部xyz测试时半尺寸应重排为 `[l,w,h]/2`。构造原global box位姿 A_t 与 `G0=frames[2].lidar_to_global_column_matrix` 后，固定t0 LiDAR中的刚体对应为：

`p_h^R = inv(G0) @ A_h @ inv(A_0) @ G0 @ p_0^R`。

此式只对相同instance的有效t0与h框成立；是原框派生的刚体对应，不是实测点级scene flow。原始sample相邻间隔实测 **0.394377–0.601561秒**，不能固定除以0.5；这批 LiDAR keyframe与sample时间戳差全0。当前交付保留全部rawglobal量，不预先选择动态速度阈值，也不修改O的任何内部坐标。

## 覆盖与缺失

以下“实例对”指anchor×instance，有重叠/重复观测，不能当独立样本数量；“缺失/消失”只表示标注槽缺失，不证明物理对象不存在。

| 计数 | train | development |
|---|---:|---:|
| anchors / scenes | 512 / 256 | 200 / 100 |
| anchor×instance对 | 16,458 | 6,259 |
| 有效box-frame | 96,799 | 36,660 |
| 缺失box-frame槽 | 18,407 | 7,153 |
| 历史至t0均未出现、仅future有框 | 1,752 | 672 |
| t0无框（包含仅future出现） | 2,666 | 1,001 |
| 历史有框、最后一帧无框 | 2,537 | 987 |
| 七帧均有框 | 11,339 | 4,247 |
| 有框但LiDAR+radar计数为0 | 15,269 | 6,285 |

全部712样本官方scene归属通过，train/dev的scene、instance token和annotation token集合均无交叉。历史val已暴露，此交付不产生新的盲测集；只生成标签，没有训练，单seed11约束不变。

## 实际验收范围

本地逐项重验所有712个压缩文件与解压JSON SHA、选择顺序/身份、帧时间、7槽null掩码、有限box值、四元数、位姿矩阵，并独立复算上表。全部矩阵旋转正交/行列式误差≤1.78e−15。与先前**独立mmap查询**的两样本77条原始GMO annotation逐字段精确相同，14个位姿最大差3.33e−15。learner对实际提取脚本独立只读审核未发现当前范围阻断。

执行命令、源上传核验与真实stdout在 `motion_targets_v1_execution.log`；本地窄测试在 `motion_builder_local_check.json`。初次两样本小查询曾在SDK导入时触及110秒上限，证据独立保留；后续成功的小查询与本次712提取均未执行该导入，也没有隐去失败记录。本次全部正式标签已完成并验收；其运动学习收益尚未测试。
