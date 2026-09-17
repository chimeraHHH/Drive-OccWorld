# SelfOccFlow：任务与连接边界（只读核对）

核对日期：2026-09-16。已读 arXiv v1 全文方法、实验及失败案例；仅采用论文和共作者主页两个一手来源。以下论文简述为 **175 English words**，未引用长段原文。

Inference uses current camera images plus ego-aligned previous-frame features; it predicts current occupancy and t→t±1 scene flow. It does not demonstrate four-horizon 0.5–2 s occupancy forecasting. Training queries t−1,t,t+1; future images/fields are training signals, not stated inference inputs. Exact deployed frame intervals remain unverified.

Motion changes the coordinates at which neighbouring dynamic SDFs are sampled; geometry losses therefore train flow through warping. Static and dynamic fields are separate. Detached dynamic BEV features yield local cosine-argmax displacements, scaled by cell size and broadcast vertically. L1 flow supervision is weighted by dynamic occupancy and forward/backward consistency. Supervision also uses photometric/LiDAR rays and Grounded-SAM masks; no annotated occupancy/flow or pretrained optical-flow teacher is used for training. The reported ablation cannot learn meaningful flow without similarity supervision.

KITTI-MOT EPE is rendered optical-flow pixel error against teacher pseudo-labels, not metric material-point EPE. nuScenes mAVE uses dynamic-class ray endpoints that pass 2 m RayIoU, in m/s; norm/averaging details are not expanded. Printed outlier inequalities/labels are inconsistent and need code verification. RayIoU, including rays from future poses, evaluates geometry, not future rollout.

依据：[论文全文](https://arxiv.org/html/2602.23894v1)，§III/Fig.2、式(2)–(8)、§IV Data/Experiments/Ablation；[同版 PDF](https://arxiv.org/pdf/2602.23894v1) pp.3–7。以上是论文表述，**没有作者代码实证**。

共作者 Daniel Göhring 的[官方出版列表](https://page.mi.fu-berlin.de/drgoehring/publications.html)收录本工作及 RA-L 接收信息，但该条目未附代码或权重入口。论文的 PDF/HTML 和 arXiv 条目也未给出可核实的作者仓库；限定题名/作者检索未找到可认证入口。因此官方 code/weights、license、commit、实现文件/行号均记为 **未核实**，不是“不存在/未发布”的结论；没有用猜测仓库的404作证据。SelfOcc 与 Self-Flow 的代码不能当成本论文实现。

对当前 SharedRigid 的判别性启示（本项目推理，不是论文替我们作出的结论）：

- 当前 Cpl/Fix 的有效问题是：在同一强 CV 起点、同一残差 value 与物理输出下，让残差姿态改变 occupancy 的空间权重是否带来可复现收益。源码已将区别限定在 `query_centers/query_rotations`，参见[本地连接](../joint_rigid_object_features_v1.py:190)。这值得检验，但不能仅凭一条非零梯度就认定语义绑定成功，更不能据本次文献核对宣称新颖性。
- 本轮保持已定的物理监督与两臂设计。若未来另设 cosine 伪标签实验，必须单独冻结其特征来源、坐标、搜索范围和训练/推理访问边界；不得把未来观测特征混入当前候选的前向输入。这个建议不授权新增本轮实验。
- 保留我们的共同 occupancy/change 风险和完整预定材料点支持；不能换成只在预测成功位置评分的运动指标来制造提升。Cpl 改善一个分数并不能自动证明位移、漏失与误报同时改善；需继续报告这些各自结果。

尚未闭合：实际数据loader时差、flow→velocity换算、mAVE范数/加权方式、KITTI outlier具体不等式、官方代码/权重和复现许可。没有安装、训练、GPU、远端执行或改动候选源码。

可机读来源与检索范围见 [sources.json](selfoccflow_task_and_coupling_boundary_v1/sources.json)。
