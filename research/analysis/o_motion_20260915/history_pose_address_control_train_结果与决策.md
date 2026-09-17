# 历史姿态投影对照：结果与研究决策

**历史姿态修正能改善固定 patch 匹配，效果在较高预测速度区间更明显；但当前图像分数仍不足以判断运动预测是否可靠。** 下一步将图像中的过去对应估计与 D 的未来轨迹预测分开，先评价冻结的学习对应模型。继续保留 O/M0，不训练当前弱 ZNCC 门控，不恢复 S3。

这是一项使用历史 GT 的定位实验，不是可部署方法，也没有产生超过 O 的新模型结果。

## 实际执行

服务器 CPU 完成原训练集 512 个样本、256 个场景，重新认证并解码 6,144 张原始图像；提取耗时 137.757 s。全部 512 个输出及元数据已回传并核验 SHA。作业正常退出，runner/child 均已不存在。

保持原 source 材料点、相机、7×7 灰度 ZNCC、D/zero 和历史图像横移对照。唯一新增参考是：用相同 GT instance 的历史框，在相机实际时间上插值中心和旋转，再投影同一材料点。只允许三个历史时刻；缺框、断链、时间越界和投影越界均保留为缺测，不借未来框补齐。第一样本原 D/zero × true/broken 共 96 个相关分数重新计算后与缓存完全一致。

参考同时改变 current 和 past 两端地址，故结果是**两端投影位置的联合效应**，不能单独归因于过去端。它也没有认证体积点是可见表面，框插值不是真实稠密 scene flow。

## 同一支持上的主结果

以下为 2 s 未来有效、CRN 未覆盖、无同格雷达回波、原图像及新参考均有效的同一人口：253,253 点，原对象权重总量 2,085.593；完整对象分母仍为 10,332。各条件均值使用原对象内点权重，不对选中点重新分配对象权重。

定义 `ΔD = corr(GT reference) − corr(D)`，`Δ0 = corr(GT reference) − corr(zero)`。正值表示参考位置具有更小匹配残差。broken 使用同一过去图像横移对照；adjusted 是逐点 `Δ − Δbroken` 后聚合。**单位为无量纲相关系数差，不是 EPE、mIoU 或百分点。**

|对照|真实图像差值|打乱图像差值|扣除打乱对照后|
|---|---:|---:|---:|
|GT reference 相对 D|0.032516|−0.001967|0.034483|
|GT reference 相对 zero|0.014093|−0.004444|0.018537|

GT reference、D、zero 的平均真实相关系数分别为 0.288718、0.256202、0.274625；GT reference 的 broken 均值为 −0.003867。扣除 broken 后，相对 D 的差值仅在 53.135% 原权重上为正，因此平均改善不等于逐点可靠。

按预定 D 预测速度区间分解：

|D 速度区间（m/s）|点数|adjusted GT−D|adjusted GT−zero|
|---|---:|---:|---:|
|[0, 0.1)|175,492|0.022302|0.011624|
|[0.1, 0.5)|47,906|0.027810|−0.000121|
|[0.5, 1)|7,695|0.010174|−0.030475|
|[1, 2)|6,433|0.160752|0.095831|
|[2, 5)|8,410|0.170857|0.159243|
|[5, 10)|6,303|0.206140|0.254292|
|[10, ∞)|1,014|0.192031|0.248416|

较高预测速度区间的效果支持继续研究运动轨迹与图像投影的连接；它不支持把全部结果解释成“patch 完全没有响应”。这些区间按 D 速度划分，并非 GT 速度。这里没有置信区间，也没有宣称区间之间的差异显著。

按原未来 GT 组划分，stationary / ambiguous / moving 的 adjusted GT−D 为 0.033957 / 0.028116 / 0.037861。moving 组真实过去位移均值为 1.381772 m，而 D 恒速向过去延拓的误差为 1.014465 m；GT 与 D 历史投影位置平均相差 27.873 px。整体对应值为 0.481235 m、0.380980 m、11.284 px。未来组标签不是过去运动类别，不能把 stationary 组的真实过去位移强设为零。

## 原图像判别仍弱

用修正到实际过去相机时刻的历史 GT 参考重新定义过去收益：

`Bpast_cam = ||Hxy|| − ||Hxy − (tpast_camera − t0_lidar) · vD_xy||`。

相同参考有效人口上，保持原图像 true/broken 分数和原速度分数，固定方向评价 `Bpast_cam > 0` 对 `< 0` 的 AUC：

|原未来 GT 组|图像 true|图像 broken|D 预测速度|
|---|---:|---:|---:|
|all|0.517367|0.497906|0.618435|
|stationary|0.529333|0.499325|0.365280|
|ambiguous|0.513630|0.516984|0.443369|
|moving|0.511725|0.487589|0.694497|

新参考位置的图像均值改善，与原分数能否排序运动收益是不同问题。不能用前者代替后者，更不能据此报告未来 EPE 改善。这些 AUC 的目标和支持与上一轮不同，不能把 0.517367 与旧总体 AUC 当成模型提升。

## 完整支持与缺测

原无雷达、CRN 未覆盖、2 s 未来有效人口共 337,615 点。六类互斥分区继续使用完整 10,332 对象分母：

|参考状态|点数|原未来负代价贡献（m）|
|---|---:|---:|
|参考有效|253,253|0.022713053|
|原图像无效|63,195|0.004216588|
|当前姿态不可用|13,460|0.001246303|
|过去姿态不可用|6,364|0.001167391|
|当前参考投影无效|3|0.000000118|
|过去参考投影无效|1,340|0.000073758|
|全部|337,615|0.029417211|

参考有效部分承载原人口负代价的 77.210%。剩余 22.790% 没有删除或填零。这里的代价是原 D 相对 zero 的未来误差贡献，用于说明支持覆盖，并非使用 GT 后实际消除的误差。

## 下一实验与研究含义

1. **停止训练当前 ZNCC gate。** 历史标签连接存在、姿态位置有效果，但可用的图像判别分数尚弱；继续堆叠 loss 或训练时长不能替代观测验证。
2. **独立估计过去图像对应。** 使用预先指定的官方 RAFT Things 权重，冻结模型；原图像先产生 current→past 对应，D/zero 只用于随后检验运动假设，不参与光流初始化。固定图像对的无 GT 接口／资源检查现已完成，见下；全 512 批量质量诊断尚未实施或启动。
3. **保留两种失败解释。** 学习对应若改善同人口判别，支持固定 patch 表征不足；若仍弱，则优先检查可见表面与体积点的关联、及将未来恒速延拓当作过去轨迹的假设。二者可能共存，不作唯一归因。
4. **最终检验仍在原任务。** 可部署证据需要在固定验证集上降低 moving 物理误差并控制 stationary 代价，随后证明原 O 占据指标改善。当前结果不满足该目标。

本轮仅 train512 内部描述性诊断；重复材料点、图像 patch、四个未来时域并非独立观测。未训练模型、未调阈值、未增 seed、未使用开发集选择方案。

## 已实际完成的后续资源检查

官方 RAFT 固定 commit `2888e15a51fa41140771d3f498ed8023cff098d1`，Things 权重 SHA256 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`。源码逐文件匹配官方 Git blob；权重取自该仓库下载脚本指向的发布包，完整传输与 ZIP CRC 通过。本地 SHA 是获取指纹，没有发现官方独立权重校验和。

H200 GPU0 上固定原 train512 第一 anchor 的 CAM_FRONT 当前／过去图像；原始 RGB 1600×900，仅官方 pad 到 1600×904 后 unpad。full RAFT、20 iterations、FP32、TF32 关闭、eval/no_grad，独立预测两个方向。严格加载官方权重，未使用 O/D、GT、训练或初始化光流。

- 两个输出均为有限 float32 `[1,2,900,1600]`，压缩保存后回读精确一致。
- 第一个方向前向 1.739 s，随后反向前向 0.198 s；没有预热或重复计时，不能将两数当作稳定吞吐或完整 512 的运行时间。
- 峰值已分配显存 4,290,969,600 bytes（约 4.00 GiB），峰值保留显存 5,230,297,088 bytes（约 4.87 GiB）。含加载、认证、保存的进程内总耗时 26.673 s。
- 作业正常结束，两进程均已退出。这只验证接口和资源可行性，尚未验证光流准确率、运动判别或占据改善。

[实际资源报告](server_results/diagnostics/raft_official_pair_probe_v1/report.json)与[结束观察回执](receipts/raft_official_pair_probe_v1_watch.json)保留全部设置、图像身份、来源和测量值。

## 工件

- [实际统计](history_pose_address_control_train_analysis_v1.json)，SHA256 `3f3067c189bd5121dd2e39993faae469f155c3d28ba31995388c2e0c7076f2f4`。
- [提取协议](history_pose_address_control_train_protocol_v1.json)、[预先固定的统计规则](history_pose_address_control_train_analysis_rules_v1.json)。
- [统计代码](analyze_history_pose_address_control_train_v1.py)、[合成 QA](receipts/history_pose_address_control_train_analysis_v1_analytic_qa.json)。
- [诊断图 PDF](figures/history_pose_address_control_train_v1.pdf)、[PNG](figures/history_pose_address_control_train_v1.png)、[图源数据](figures/history_pose_address_control_train_v1.data.json)。
- [独立实际 2 s 重算](receipts/history_pose_address_control_train_v1_independent_2s_audit.json)及[主线程比对](receipts/history_pose_address_control_train_v1_root_independent_comparison.json)：独立读取 2,048 个 NPZ，重算 no-radar/all 和 moving 的六项 contrast、三项 AUC 与完整缺测收益账；没有导入原分析 helper。限定这些统计的交叉核验，不声称全部时域、速度区间或图像几何已独立重算。
- [服务器结束回执](receipts/history_pose_address_control_train_v1_watch_final.json)、[全部输出回传核验](receipts/history_pose_address_control_train_v1_transfer.json)。
- [上一轮历史标签连接](history_motion_alignment_train_结果与决策.md)、[官方光流接口与研究边界](research_notes/history_camera_frozen_flow_feasibility_v1.md)。
