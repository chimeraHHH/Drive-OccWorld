# M0 当前坐标合同独立审计

结论：**已确认局部源码中的显式几何采样约定与监督网格不一致；尚未证明训练后 checkpoint 的输出实际错位，也未证明它导致现有成绩。** 标签被转换到观测时刻的 LiDAR 参考系；未来查询与生成记忆的显式 SE(3) 路由则采用各自未来时刻坐标。末端没有看到将未来局部输出变回观测参考系的确定性 warp。可学习的采样偏移、动作条件和特征可能补偿这种差异，故这属于需要辨别的几何先验与任务连接问题，不是“该网络不可能拟合标签”的证明。

本次仅本地只读源码审计与 CPU/NumPy 几何实验；不连接服务器，不修改冻结代码，不改变或停止训练。工程回放一致只能证明复现了既有链路，不能证明链路的物理坐标合同正确。

## 1. 当前生效的数据与模型分支

审计配置 [S0.py](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py) 的 SHA 为 `c92faf7c…374303`，与本轮冻结协议一致。`data.train/test.pipeline` 均使用 `LoadOccupancy(use_fine_occ=True,time_history_field=2,time_future_field=4)`；类别为 GMO/non-GMO，规划、flow、motion-residual、Doppler 分支均关闭；future head `use_plan_traj=False`、`sem_norm=False`、`ego_motion_ln=True`。本轮缓存拟合也复用该原生 test pipeline，仅切换 train annotation。

因此这里不存在启用规划后重写位姿、GT-conditioned semantic normalization 或外部分支输出 warp 对结论的抵消。

## 2. 标签明确落在观测参考系

设列向量形式的 LiDAR 到世界位姿为 (G_t=E_tL_t)。当前观测帧是 (t=0)，其 LiDAR 坐标系记为 (R)。

- [dataset template:327](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py:327) 的 pose helper 返回 **未旋转的负平移**与**逆四元数**。这一步明确了 loader 中加减号含义，并非依靠变量名猜测。
- [template:500](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py:500) 记录 `index-2…index+4` 的七帧 pose；第 2 项就是当前输入 anchor。
- [LoadOccupancy:65](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/pipelines/loading_occupancy.py:65) 先将源 `[z,y,x]` 体素中心还原成米制 `[x,y,z]`，L75–89 的实际运算组成 (p_R=G_0^{-1}G_t p_t)，随后按同一 `pc_range` 体素化。所有未来标签都是未来世界状态在**固定观测 LiDAR 系**的快照。
- [union2one:218](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_v1.py:218) 将当前帧持有的整段 `gt_occ` 原样交给 `segmentation`，未重新转到未来帧。`future_metadata_only` 只省去未来图像加载，不重写该标签。

GMO 是可移动语义类别，包含静止停放车辆；以下静止点反例不把 GMO 当作真实运动掩码。

## 3. 查询/记忆使用另一套显式空间约定

当前观测 BEV 的几何来源是当前 LiDAR 网格： [encoder:112](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/modules/encoder.py:112) 将归一化点转成 LiDAR 米制坐标后，经 `lidar2img` 投影取图像特征。历史对齐得到当前观测，不会改变未来标签的参考系。

[dataset V1:75](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_v1.py:75) 实际生成行向量矩阵 `future2ref=(G_0⁻¹G_t)ᵀ` 及其逆，而 [detector:319](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:319) 对规则查询坐标 (q) 使用：

\[
q_{\mathrm{history},k}=q_t\,T_{t\to R}\,T_{R\to k}.
\]

初次历史槽就是 (R)，故零偏移对应关系是 (q_tT_{t\to R})，即**把未来局部查询投回当前观测**。 [decoder:244](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/modules/world_decoder.py:244) 将此坐标传给 cross-attention；L639–644 再加可学习 `sampling_offsets`。输出 token 顺序保留查询顺序。

[detector:576](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:576) 把生成状态与 `ref2future` 成对入队，下一步继续按“该状态属于上一未来局部帧”寻址。双槽 adapter 保留了同一约定，未修正这项合同。

逐层检查未发现漏掉的确定性补偿：

- `PredictionTransformer` 仅转发网格与 query；文档称 “reference frame” 不足以推翻上述实际矩阵运算。
- [ConditionalNorm:228](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/modules/conditionalnorm.py:228) 将相对位姿编码成 `gamma/beta` 并逐位置调制通道；没有按位姿重采样空间索引。
- [head V1:162](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:162) 是逐 token MLP、reshape、permute；不是 SE(3) warp。
- [compute_occ_loss:596](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:596) 和 [evaluate_occ_records:718](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:718) 只做 token→`[x,y,z]` 轴排列、时域切片、分辨率插值和同位置评分。固定 x/y 转置不能补偿随样本/时间变化的自车位移与旋转。

## 4. 可复现物理反例

[coordinate_contract_phantom.py](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/coordinate_contract_phantom.py) 精确提取并执行 loader 的 L68–89 AST、dataset 的两方向矩阵表达式，以及 native query 的两个矩阵乘法表达式。仅以 NumPy 代替矩阵乘法、以显式正交旋转矩阵代替四元数对象；未执行神经网络、learned offset、体素插值或真实 `.npy` 数据。选择平面位姿排除高度/roll/pitch 的旁支问题。

固定世界点为 `(10.1,4.3,0.1)`，观测位姿为恒等变换：

| 自车未来运动 | loader 的 GT 位置（参考系 xy） | 几何路由正确读到静止点的未来 query xy | 在 GT 同数值 query 上实际读取的旧状态 xy |
|---|---|---|---|
| 不动 | (10.1,4.3) | (10.1,4.3) | (10.1,4.3) |
| +x 2 米 | (10.1,4.3) | (8.1,4.3) | (12.1,4.3) |
| +90° yaw | (10.1,4.3) | (4.3,−10.1) | (−4.3,10.1) |
| +x 2 米、+90° yaw | (10.1,4.3) | (4.3,−8.1) | (−2.3,10.1) |

四例全部通过；另有非恒等 LiDAR 外参例验证 loader 确实组成 (G_0^{-1}G_t)。结构化来源哈希、执行片段与全部矩阵见 [coordinate_contract_proof.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/coordinate_contract_proof.json)。这证明**显式对应关系的反例**，不模拟真实类别 logits。体素量化可带来亚体素偏差，但不能普遍抵消米级、随 ego 改变的差异。

## 5. 识别边界与下一项核验

1. 学习后的 token 语义最终由监督塑造。offset/action conditioning 可学到反向 ego 补偿；所以不能直接给旧 checkpoint 的输出再加 warp，也不能由此宣称所有旧实验无效或数值低的主因已确定。
2. 需要根任务只读核验服务器的 `nuscenes_world_dataset_template.py`、`e2e_predictor_utils.py`、`encoder.py` 与本次证据 SHA 一致：这些辅助文件不全在既有 13 文件 runtime 合同中。已冻结 loader/detector/config 的本地 SHA 已逐项匹配。
3. 后续低成本真实核验应固定一条含非零 ego 位移/旋转的记录，核源 occupancy 点、pose 列表、第 2 帧 anchor、最终 GT 索引与 future reference-point 网格；不依据预测好坏挑样。源 `.npy` 是各时刻 LiDAR 局部占据的实际存储合同也应以该例核实。
4. 若要接 future-feature teacher，先明确一致目标：全程参考系表示，或未来局部内部表示并在监督前明确转换到参考系。未经坐标合同区分，单纯增加 teacher/持久记忆不能被称为修复此问题。本审计不提出当前立即部署的修补。

## 6. 根任务补充核验（2026-09-15）

服务器三个辅助源码与本地逐项同 SHA，结果见 [coordinate_auxiliary_source_match.json](receipts/coordinate_auxiliary_source_match.json)。

另核查 [OpenOccupancy 作者数据说明](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md)及其[原始 loader](https://github.com/JeffWang987/OpenOccupancy/blob/main/projects/occ_plugin/datasets/pipelines/loading.py)：按各 scene/lidar token 加载稀疏 `[z,y,x,cls]`，在各自规则体素中心解码为米制坐标；只施加 BEV augmentation 后用 LiDAR-to-camera 投影。这支持“源占据按各帧 LiDAR 局部系解释”的代码推断，不构成对已下载标注全部生成过程的独立真值验证。未来各帧在本项目统一到 anchor 的数值执行仍由 real_coordinate_audit.py 验证；无 checkpoint 输出语义/性能结论。
