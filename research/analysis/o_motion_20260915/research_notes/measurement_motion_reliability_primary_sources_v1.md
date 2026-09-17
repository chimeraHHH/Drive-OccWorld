# 从运动自洽转向真实测量对应：两项原始工作的边界

本轮仅阅读两篇原文、作者固定 commit 源码及 nuScenes 官方 devkit；没有执行外部代码、提取数据或训练。已有笔记未检出 RaFlow/CMFlow 的源码审计，未重复 DeGO/UniAD 等综述。精确 URL、访问时间、commit、源码 SHA 见 [sources.json](measurement_motion_reliability_primary_sources_v1/sources.json)。

**可迁移的重点不是再加一项一致性 loss，而是让运动假设接受实际观测位置、时间和测量方向的约束。** 多时域 D 输出彼此一致，只是预测内部一致；观测残差必须来自实际输入。下述论文的相邻已观测帧 scene flow 并不等于我们的 0.5–2 s 未来材料点位移预测，数据集和支持也不能直接比较。

## 两项直接先例

| 工作与原文 | 实际源码机制 | 对本项目的边界 |
|---|---|---|
| RaFlow，*Self-Supervised Scene Flow Estimation with 4D Automotive Radar*，RA-L/IROS 2022。[原文 §III](https://arxiv.org/html/2203.01137v2) | [models/raflow.py L57–131](https://github.com/Toytiny/RaFlow/blob/c01897faeb5d1e766ba899a677b38b1282b142e4/models/raflow.py#L57)：两次真实扫描，经 patch correlation 后估计 flow；刚体变换与逐点径向测量残差产生静态内点，再作刚体修正。[losses/loss.py L78–103](https://github.com/Toytiny/RaFlow/blob/c01897faeb5d1e766ba899a677b38b1282b142e4/losses/loss.py#L78)直接比较位移沿雷达视线投影和 RRV×实际 interval。 | “测量残差识别静态、用刚体流替代”已有先例，不能将一个 gate 包装成新颖性。双帧相关承认回波并非稳定一一对应，不能把最近邻当真实身份。它的输出支持仍是雷达点，不是任意无回波 voxel。 |
| CMFlow，*Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision*，CVPR 2023。[原文 §3](https://arxiv.org/html/2303.00462v1) | [模型 L171–194](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/models/cmflow.py#L171)：训练用 pseudo static mask，推理换预测 mask；加权 Kabsch 后修正静态 flow。[radar_loss.py L207–240](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/losses/radar_loss.py#L207)：真实两图光流给出终点像素，将 warped 3D point 与对应相机射线比较。[预处理 L91–148](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/preprocess/utils/get_flow_samples.py#L91)使用 RAFT 和 LiDAR 跟踪等训练监督。 | 多模态监督仅训练使用，推理仍需要两帧已观测 radar。其 LiDAR track ID/pseudo mask 不是我们当前检测框 index 自动具有的身份。图像对应是额外测量关系，不能把它简化为已有 D 的局部平滑；也不能将其训练 pseudo GT 直接当我们推理输入。 |

这里的“原始测量约束”并不保证可靠：稀疏回波、多径、关联失败和切向运动均有不可辨识性。CMFlow 代码还把径向约束 interval 固定为 0.1 s（[L100–120](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/losses/radar_loss.py#L100)）；移植时不能沿用此常数代替 nuScenes 的真实时间差。

## 8 通道到底丢失了什么

当前 native 的 presence/count、mean z/RCS/vx/vy/speed/lag 是多个 radar、多次 sweep 的格内汇总。[已有本地源码审计](inference_motion_evidence_available_assets_v1.md)已核其编码和 R 坐标。它没有保留 **哪个传感器在什么时间、沿哪个方向，在什么实际位置给出了哪一条速度与质量记录**，也没有保留两个真实观测之间的对应关系。均值不能恢复这些联合关系或多峰分布；mean speed 不等于完整方差。

D 已间接消费融合 radar8ch 与相机历史的当前 BEV。重新使用逐回波、逐时刻关系可增加汇总所丢失的可检验结构，但不是一个与 D 统计独立的新传感器。CRN 现有 7 列点文件已经按速度×时间做过位置补偿，缺精确时间和 sensor ID；不能用这种位置重合“证明”同一速度正确。须重新只读当前/过去 raw PCD、图像、calibration/ego_pose/sample_data 元数据；本轮未生成或认证这类新资产。

## nuScenes 的必要语义边界

- PCD 给的是 `vx,vy` 和 ego-compensated `vx_comp,vy_comp`，不是本文两套 4D radar 数据可直接互换的独立 RRV 标量。把补偿速度投影为径向量，只是该速度记录的派生量，不得称恢复了原始 Doppler 测量。[官方字段 L312–331](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/data_classes.py#L312)
- 视线应从**该次雷达传感器原点**指向回波；把五个传感器合到 R 后，用 `p_R/||p_R||` 代替每传感器 LOS 会改变测量算子。速度只乘旋转，不乘平移；官方 `PointCloud.rotate/transform` 只改变 xyz，不能假设它已同步旋转速度字段。pitch/roll 也不能随意丢掉。[实现 L160–187](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/data_classes.py#L160)
- 每次 scan 有自己的 ego_pose 与 timestamp；多扫聚合只是坐标对齐，不是动态物体位置对齐。官方 multisweep 返回逐扫 lag 并沿 `prev` 读取，不能把 mean lag 当独立时间序列。[L58–132](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/data_classes.py#L58)
- 相机与雷达不能共用一个“当前 ego pose”投影：官方先经过点云时刻 ego→global，再到相机时刻 ego→camera。即使这样也未补偿时间间隔内物体运动；相机 z 是深度方向，不能沿用 radar 的 x-forward 语义。[官方投影 L884–907](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/nuscenes.py#L884)
- `dyn_prop/ambig_state/invalid_state/pdh0` 是传感器状态或概率档位，不是 GT 运动标签，也不能把 RMS 编码当已校准连续方差。默认过滤已筛过这些状态；比较必须记录实际 loader 的过滤策略，不能静默改成仅 moving returns。[默认过滤与字段](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/data_classes.py#L261)

## 一个优先下一实验：无同格 radar 处的历史图像对应检验

主线程的新真实诊断提示同格雷达仅覆盖少数待补区域；这不等于全部 radar 信息的上限。因此优先检验**另一个实际观测通道是否支持无回波区域中的运动假设**，不把 radar 格子随意膨胀来制造覆盖。此处仅为候选，未实施、未冻结参数，也没有新方法结果。

固定 D、原 grid 与全部 source 支持，在读取 GT 前，对每个固定源位置 `p0` 用短时假设 `v_D=D_0.5(p0)/0.5`，以及 `v_zero=0`，构造 `p(τ)=p0+(τ−t0)v`。只使用决策时刻之前已获得的两次同相机观测，分别按其真实 τ、标定和 ego pose 投影，比较图像 patch 的对应残差；不需要 GT 深度、GT flow 或 future image。运动地址由假设和已知成像几何确定，不能用未来标签寻找匹配。这个诊断是在检验“未来预测速度向过去短时外推”假设，**不声称 D 已经预测过过去轨迹**。

关键控制是正确时间/图像对应与破坏该对应的匹配控制，同时保留原 D-speed、可见性/纹理信息和 same-cell radar 分组。D/zero 共同可投影的范围内作配对比较；低纹理、遮挡、出视野或无有效图像的点必须单独报告，原 full-object 分母及无测量部分仍保留，不能只挑有利纹理点。patch 尺度、度量与控制在读效果标签前确定，不通过扩大搜索半径直到命中来选规则。全场分数建立后才按原 sparse 标签评估：它是否在既定速度分层中，区分 D 相比 zero 真正有益的补覆盖与静止误运动，超过速度/可见性对照。train512 仍为 D-seen 内部诊断；之后冻结规则在既定 dev200 复核，也不能称全新未接触测试集。

如果真实对应残差不能优于速度/可见性及破坏对应控制，或有效测量依旧不触达主要误运动支持，就否定这一测量联系的当前实现，不用更大邻域或更多 loss 挽救叙事。正结果也只支持这条观测联系有判别信息，不等于已改善 occupancy/未来 motion。遮挡、重复纹理、光照、固定 grid 的高度歧义、短时加速/转向，以及径向观测的切向盲区都需保留为限制。

这个实验受 CMFlow 的真实图像对应启发，但不等于复现它的 RAFT/LiDAR 训练；本轮没有加载那些模型。共享测量约束、静态修正或 gating 本身均已有先例，未来贡献必须来自被控制实验验证的具体观测—运动连接及完整支持上的净收益。
