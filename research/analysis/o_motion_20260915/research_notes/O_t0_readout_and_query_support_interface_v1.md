# O 的 t0 读出与无 GT 查询支持接口

2026-09-16。只读现有源码所得接口备忘录；未运行模型、未修改实验源码。
**可以绕过四步未来 rollout 读取 O 的当前 GMO 概率，但仍需 O 的已训练末层读出权重。**
以下窄调用由源码支持，尚未执行与原完整前向的数值或逐字节等价验证。

## 1. 当前读出不依赖未来 rollout

原输入 `inputs['prev_bev_input']` 为 float32 `[1,1,40000,256]`。
`state = inputs['prev_bev_input'][:, -1]` 为 `[1,40000,256]`。
原 `future_pred` 将它原样复制给三个读出分支，作为 t0 feature；
未来 transformer 另外生成四个 future feature，最后才与 t0 合批读出。
见 [drive_occworld.py:477](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:477) 与 [合批读出:589](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:589)。

原 `forward_head_layers` 是逐 token MLP、reshape、permute，不混合不同时域。
因此当前末层可由 `model.future_pred_head.bev_pred_head[-1](state)` 单独得到。
本配置输出 `[1,40000,32]`，其中 `32 = 16 heights × 2 classes`。
见 [world_head_v1.py:162](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:162)。

轴转换为 `reshape(1,200,200,16,2)`，此时两平面轴为 **Y、X**；
再 `permute(0,4,2,1,3)` 得 `[B=1,C=2,X=200,Y=200,Z=16]`。
按 `softmax(dim=1)[:,1:2]` 得当前 GMO 概率 `[1,1,200,200,16]`。
原全输出形状为 `[5,3,1,1,40000,16,2]`，同轴约定见
[predictions_to_xyz:43](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/oracle_transport_probe.py:43)。

必须装载 O 的读出参数；缓存 `native_preds.npy` 是 M0，不能替代 O。
O head 装载见 [native_full_o_stream_v1.py:204](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/native_full_o_stream_v1.py:204)。
缩小 GEMM 批量可能改变浮点实现细节，不能仅凭逐 token 数学结构声明 bitwise 相同。

## 2. 不读 GT 的输入、规则坐标与 D

现有 [input_only:86](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/common_change_evaluation_v2.py:86) 认证并只读 `inputs.pt`，不打开 targets。
D 使用相同当前 BEV，输出 `[1,4,3,200,200,16]`；原标签索引采样只是后续操作，见
[load_tokens:215](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:215) 与 [gather_sparse:236](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:236)。

规则查询为当前 LiDAR 系 R 的 640,000 个体素中心，XYZ C-order、Z 最快：
`p_R = (-50.944 + .512*x, -50.944 + .512*y, -4.75 + .5*z)` 米。
范围 `[-51.2,-51.2,-5,51.2,51.2,3]`；接口见
[voxel_centers_xyz:123](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/motion_geometry.py:123)。
可以先在全网格定义 O 概率与 D 位移，再用独立标签评分；无需由 GT 定义前向查询。
但当前诊断的 `source_flat_indices` 实际经过当前 GT 框唯一覆盖筛选，见
[build_sparse_motion_supervision_v1.py:141](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/build_sparse_motion_supervision_v1.py:141)。

## 3. 概率支持不等于可见表面

class0 同时包含背景静态语义与空体素，class1 是 GMO。
`p(GMO)` 是车辆、行人等可动物体语义组概率；停车车辆仍可属 GMO，并非当前正在运动的概率。
标签映射见 [loading_occupancy.py:106](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/pipelines/loading_occupancy.py:106)。
因此 `p(GMO)` 可作预测软支持权重，**不能直接作透明度、遮挡概率或可信深度**。
全网格或 O 预测支持仍是虚拟体积点；消除 GT 选点依赖不等于取得可见表面点。
当前 256 维 BEV feature 没有独立高度轴；16 高度由分类读出展开，不是测得深度特征。
原 encoder 的 `bev_mask` 仅检查正深度和图像边界，未检查遮挡，见
[encoder.py:136](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/modules/encoder.py:136)。

## 4. 历史相机接口及最小接口边界

既有 metadata 已保存各相机自身 timestamp、`camera_to_global`、`K`、图像 SHA，
以及当前 LiDAR 的 `G0/t0_lidar_us` 和 `input_availability_us`，见
[extract_history_camera_inputs_train_v1.py:100](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/extract_history_camera_inputs_train_v1.py:100)。
[project:75](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/history_camera_evidence_v1.py:75) 已按真实相机时间及 `inv(GC)@G0` 投影；当前相机稍晚于 LiDAR 时须沿已认证 availability 边界解释。
需要新增导出的只是 O t0 概率及明确的无 GT 查询地址；D 全场与相机变换接口已有。
这说明接口可实现，不证明历史对应有用，也不证明软支持可以解决遮挡、表面或运动预测问题。
