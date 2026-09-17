# 运动监督与评价合同审查

2026-09-15。结论：**原始 nuScenes 实例时序框可以提供可审计的物体中心位移与刚体对应监督；现有 fine occupancy、O checkpoint 和已完成混淆矩阵不能直接给出真实点级 scene-flow 精度。** 下一步应保留 O 的原始 GMO 任务，同时新增独立的运动监督和分层评价；不能把误报减少或 GMO IoU 增长直接称为运动变准。

## 已核实的真实资产

本次固定取旧 selection_v1 的首个 train、首个 development anchor；每个取原 sample_annotation 文件顺序前 8 条实例及其原始前后链，没有按运动速度或预测选例。服务器只读查询用时 43.61 秒，无模型、checkpoint、缓存张量或传感器图像/点云读取。大 JSON 只做只读 mmap token 搜索并反序列化匹配对象，记录字节区间 SHA、文件大小/mtime；**未声称对大 JSON 完整 SHA 认证**。证据 [motion_metadata_evidence_v2.json](motion_metadata_evidence_v2.json)，SHA `e72922db8d49fd89696209173ddf7c69bd13129ccbe4cb2cd35fa553d7984021`。前次查询在导入 SDK 引发 sklearn 时触及 110 秒 CPU 上限，失败记录独立保留；成功版只解析官方 split 源码字面量。

共同数据根是 `/home/wangning/data_cache/RadarFlowOcc_m3_h200/datasets/nuscenes_driveocc/`，官方表在 `v1.0-trainval/`。可用原字段：sample 的 token/scene_token/timestamp/prev/next；sample_annotation 的 instance_token、annotation token、sample token、global translation、wlh、wxyz quaternion、visibility_token、attribute_tokens、num_lidar_pts/num_radar_pts、prev/next；sample_data 的 is_key_frame、LIDAR_TOP 外参/ego pose token；calibrated_sensor 和 ego_pose 足以组成每帧 LiDAR→global。`sample_annotation` 本身没有 velocity 字段，category 通过 instance→category 关联，不能把每帧 annotation token 当实例 ID。[官方 schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)

| 固定例证 | 真实时域与运动 | 可见性边界 |
|---|---|---|
| train `464fe0be05a74ec9852573cb9e3afd89`，scene-0706 | dt=[−1.000307,−0.499884,0,0.499866,0.999223,1.499062,1.998386] 秒。car instance `5b2de5ea016b46a3a661e1d3522227e5`，t0 LiDAR 中心 (−15.5179,−22.8469,−2.7689)→最后 (−15.2846,−49.5432,−4.3562)m；相邻未来平面速度约13.38m/s | t0 visibility4、13个LiDAR/2个radar点，未来 visibility降至1且有0点帧，原始框仍存在。缺传感器点≠无GT/静止 |
| development `297c52902a384b84ba179f3160ac54e9`，scene-0770 | dt最后1.997862秒。parked car `35bd679e02594cbaa2b3f913fa63a858` 中心xy (23.8471,5.4023)→(23.8534,5.4025)m；stopped car `70c3ebae645640f8834cd9ab9a465b93` 约0–0.0045m/s | 停车框有毫米级抖动；“非零差分”不能定义真实 moving。两例官方 train/val scene 成员核对通过 |

两例 t0 原 occupancy 分别是 int32 `[172259,4]` 和 `[18459,4]`，列为 `[z,y,x,class]`；前者 SHA `c30d684853547fa373f5f05f0ce2b6628bec830061b18f5f76d79ca4baf20bfc`，后者 `83a5d1694a4d7fe161e4b7a84e20dee6b684168a7896cdbbe1bcf5b4bd9e0ca5`。路径见 JSON。既有真实七帧审计也均为4列。只证明已检样本格式，非全数据每文件穷举。

## 现有源码支持什么，不能直接复用什么

本地/服务器以下8文件逐项同 SHA：dataset template/V1、loading_instance/occupancy、drive_occworld、world_head_v1、semkitti_loss、nuscenes_converter；完整 hash 见证据。

- `loading_occupancy.py:60–65` 注释兼容7列，但实际只取前三坐标与末列类别，丢弃中间字段；当前 S0 的 `use_fine_occ=True` 输出二值 GMO，原类 `[2,3,4,5,6,7,9,10]→1`。0也包含静态语义类，**不是纯物理 free-space**；255保持 ignore。
- `native_state_cache.py:166–200` 明确拒绝 instance/flow/gt_future_boxes 等进入模型输入；target只保留原 `[1,7,512,512,40]` occupancy。新 labels 必须是独立旁路文件，不能混入这个输入树。
- 旧 `loading_instance.py:183–244` 以 ratio4 生成 128×128×10 标签：首帧是体素→本帧实例中心，后续是本帧体素→前帧实例中心的反向 offset，单位**0.8m粗格**，不是 m/s，也不只是物体平移。`fill_occupancy` 填的是旋转框外包的轴对齐长方体，并要求整框落ROI；因此不能将它当原 fine occupancy 实例真值。
- `nuscenes_world_dataset_template.py:397–409,440–461` 会过滤未在历史可见的未来实例；随后缺帧复制旧框、每轴xy位移≤1m则固定旧位置/旋转。这可能抹掉真实低速运动；新监督禁止用 refine 后字段。原 annotation 的断链/漏帧必须 mask，不得补为静止。
- `world_head_v1.py:302–328` 的旧 flow loss 是三个层各0.5×masked SmoothL1；`drive_occworld.py:1056–1062` 将 flow 转实例后测VPQ，不是物理EPE。fine-occ 的 `union2one` 没有创建 flow_list；直接开启旧 turn_on_flow 既无适配标签，还有未赋值路径。默认 `data/cam4docc` 路径不存在，本次未声称服务器所有位置都没有其他派生flow。
- SDK `box_velocity`（已安装源码377–422）是全球中心差分，优先前后中心差、边界单侧差，无足够链或过大间隔为NaN；默认中心差允许总间隔3s、单侧1.5s。它会用未来 annotation，故可作监督/评价，不能作为只见历史的模型输入或历史CV基线速度。[原 SDK](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/nuscenes.py)

## 坐标、时域与信息边界

统一列向量记 `G_t=T_global←ego(t) T_ego←lidar(t)`。所有 GT 框的 global center `c_t` 变为固定 anchor 坐标：`c_t^R=G_0^{-1}[c_t;1]`；真实物体位移是 `c_h^R−c_0^R`，除以真实 `timestamp` 差才是m/s。自车走了10m不应让静物获得10m物理位移。box orientation 也需先按 global→R 旋转，四元数保留原wxyz并做单位范数检查。

原 fine GT loader 已将未来 occupancy 统一至t0 LiDAR，原 evaluator/GT/mask必须保留。原生 latent **显式采样几何**随未来LiDAR格并使用行向量矩阵、BEV参考点z=1；但这不能证明已学输出必然处于未来局部语义。先前全reference A 已失败，不能据此再给 O logits 盲目加 warp。任何新增 transport 的点/向量变换、逆向grid_sample、米↔格↔[0,1]/[−1,1]换算须各自验证，尤其区分位移向量（无平移）和点。

S0 已启用 future_can_bus/use_vel/use_command；V1以未来自车轨迹、位姿构造这些条件。这是**已知未来ego条件的环境预测**，并非完全无条件未来预测。新比较应保持这份既有条件。未来物体框、future image/occupancy或由中心差得到的物体速度只能进入 train loss/stop-gradient teacher 或评价，不进入测试输入。历史CV基线只能用≤t0的有效观测；不能用 SDK 中心差偷偷读t+0.5s。

物体框 SE(3) 可构造 `p_h^R=c_h^R+R_h^R(R_0^R)^T(p_0^R−c_0^R)`，forward displacement=`p_h^R−p_0^R`（m）；反向采样须另推导，不能直接把符号取反当全空间inverse。只对唯一原始实例覆盖的有效fine前景赋值，交叠/未关联/缺帧/超ROI另mask并报覆盖。它是**box rigid correspondence派生监督**，行人关节等不是刚体，也不是实测点级scene flow。UniOcc已有这种实例位姿构造先例，并明确修过 annotation_token/instance_token 混淆，不能将构造法本身包装成新贡献。[作者实现](https://raw.githubusercontent.com/tasl-lab/UniOcc/main/uniocc_flow_gen.py)、[项目记录](https://github.com/tasl-lab/UniOcc)

## 可计算的评价与不允许的推断

1. 保留全GT、原argmax和四未来 macro GMO，以及t0/逐h混淆矩阵。再以原框连续位移分 moving/stationary/ambiguous（阈值从训练集噪声与预登记决定，勿按dev效果调；类别或moving attribute不能独代物理速度）。在同一固定GT对象集合/空间支持报告 IoU、召回、精度、TP/FP/FN、目标质量/漏检率、每类每速度/距离/可见性覆盖。移动前景召回与目的地恢复可揭示“缩小预测只减少FP”。不要只报告移动框内部precision，因为内部几乎全是正类；背景/邻域支持也须固定且不得替换全GT主指标。
2. 目前 O 没有显式velocity/flow输出，不能从已存confusion复原轨迹。若新方法输出位移，直接报告按原instance匹配的四时域中心ADE/FDE、向量EPE（m）、速度误差（m/s），包含静止/低速/快速和遮挡分层；预测漏检必须保留在覆盖/失败统计，不能只平均匹配成功目标。O的占据场可用**同一冻结的非学习解码规则**提取轨迹代理，但需要单独工程验证与缺失惩罚；不能把GT给定局部框内的质心误差当无oracle的端到端预测精度。若O无共同motion输出，则只把EPE作为新分支机制验证，不能据此单独声称相对O运动更准。
3. 原GT的新增/消失前景集合与旧位置残留率可作低成本补充，但包含采样密度/遮挡/标注变化，不全是运动；应结合实例轨迹限定解释。box填充结果不可替换 fine GT。输出只有显著性、没有运动量及漏检覆盖，仍不足以支持新目标。

现有目录认证的原生可用训练为23,930 anchors/700scenes，完整val为5,119/150scenes，train/val scene无交集；当前固定512训练/256scenes和200开发/100scenes来自其子集。原来标为locked的50scenes已进入完成的full验证，**当前不能继续声称盲测**。新motion标签本身不会创造新独立测试集。训练与开发分别提取/存储，运动阈值、类别映射、关联/歧义规则和调参只能由train确定；保持单训练seed11，最终不把场景bootstrap当多seed稳健性。

根任务已授权下一动作：按同一 selection_v1 原序，CPU提取512/200的原始七帧box-chain，先交付独立压缩标签、完整metadata来源SHA与覆盖统计。它不修改模型/缓存、不筛预测、不启动S3；暂不把新motion labels当已有运动增益。
