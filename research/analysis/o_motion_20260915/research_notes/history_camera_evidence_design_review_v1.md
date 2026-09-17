# 历史图像能否鉴别 D 补运动：最短固定诊断草案

状态：只读设计审查，未实现、未执行，以下常数是执行前一次性固定的建议，不是已认证的结果。依据当前 `next_diagnostic` 和[已有原始文献审计](measurement_motion_reliability_primary_sources_v1.md)。本轮不拟合 gate、不训练、不读取开发集，也不改变占据或物理评价。已有雷达诊断中，无同格回波部分承载约 95.395% 的负收益代价；这支持检查另一种测量关系，不能证明图像一定能解决它。

## 1. 时间、坐标与两种假设

保留 train512/256 场景、原 source 点及四时域支持。对任意当前 LiDAR 固定参考系 R 中的 `p0`，沿用上一诊断的四时域名义最小二乘速度：`vD = Σ h·D_h(p0) / Σ h²`，`h=(0.5,1,1.5,2) s`；另一假设 `v0=0`。不改用效果较好的某个 horizon，不使用未来真实 dt 来生成信号。公式对全空间定义，计算可只查询原固定 source 索引；GT 对象身份、位姿和运动分组不能参与场的定义。

图像只取**原 native 已消费的 current 相机记录**和**立即前一个 keyframe sample 的同一相机记录**，不搜索最匹配时间、相机或光流。使用各自实际 camera timestamp `τc, τp`，缺少前帧或非同场景即记缺测。令 `G0=T(global←LiDAR(t0))`，`Cτ=T(global←ego(τ))·T(ego←camera(τ))`，使用列向量：

`q_a(τ) = inv(Cτ) · G0 · [p0+(τ−t0)·v_a; 1]`，`u_a(τ)=project(Kτ,q_a(τ))`。

current 和 past 两端都作该运动时间补偿，不能只移动过去点，也不能用同一个 ego pose 投两帧。`vD` 是 R 中向量；上述先在 R 内更新位置再经过 G0，已包含正确全三维旋转，不能另加 ego velocity。这里检验的是“未来预测的平均速度向历史短时延伸”的假设，不声称 D 原生预测历史轨迹。

**availability** 定义为本样本原 native 实际消费的 current LiDAR、六相机及 radar 扫描中最大 timestamp，记为 `T_available`；记录它相对 t0 的延迟及两图时间偏移。current 相机比 LiDAR 稍晚、但已属于原输入且不晚于该 availability，不自动构成未来泄漏；结论必须写成“在原输入 availability 下”，不能冒称严格 t0 在线预测。禁止 sample.next、超过 availability 的额外图像，以及未来标签、future ego pose 辅助对应。若原输入的时间边界尚未认证，先完成元数据检查，不先赋予在线因果解释。

## 2. 一套固定的视图、patch 和错误对应对照

- **视图只选择一次**：按 zero 在 current 时刻的投影，筛选正深度且完整 patch 位于图内的相机，选归一化光轴距离 `||(qx/qz,qy/qz)||` 最小者。平分遵循已认证 native 顺序 FRONT、FRONT_RIGHT、FRONT_LEFT、BACK、BACK_LEFT、BACK_RIGHT。不读取图像相似度、不依据 D 或 GT 选择。其后 D/zero 在该相机 current/past 四个投影都须正深度、完整 patch 可采样；失败记缺测，不换相机补救。
- **固定度量**：原始分辨率、7×7 像素、双线性亚像素采样，灰度 `(.299R+.587G+.114B)/255`。两 patch 各减自身均值后为 a、b；`corr=Σab/sqrt((Σa²+49ε²)(Σb²+49ε²))`，固定 `ε=1/255`，残差 `1−corr`。`s_photo = residual_zero − residual_D`，方向预定正。无窗口搜索、无局部最优匹配、无 RAFT、无学习特征或多尺度择优。这是在实际数据分析前按根任务要求固定的 regularized ZNCC，替代本页未执行的 Census 初稿。
- **低纹理规则**：仅以所选 current-zero patch 的标准差 `<2/255` 标低纹理缺测；该常数执行前固定，不按收益改。其他实际或 broken patch 不再按相似度或纹理筛除，另记标准差；固定正则项令平坦 patch 的相关为零而非除零，但相等残差不是静止证据。实际与 broken 完全共用该 mask，禁止对 broken 另立纹理筛选。图像缺失、几何越界和低纹理分别记账，不填零分数。
- **唯一 broken-correspondence 控制**：过去图像水平循环移位 `W//2` 像素，current 图像不变；投影、时间、相机、patch 大小、纹理筛选与 D/zero 均不变，控制 patch 采用循环索引。实际/控制使用完全同一有效点集合，比较同方向的两种 `residual_zero−residual_D`。这一固定操作保留该图的亮度分布及垂直天空/道路布局，破坏正确位置对应；不尝试多个移位择优。循环接缝及重复纹理使它不是完美的 nuisance-matched 因果对照。

## 3. 最少输出与如何否定

提取器先产生输入定义的信号和缺测原因，再加载标签评分。保留 sample/scene/source index、camera/sample_data token、两时刻及 availability、intrinsic/extrinsic/ego pose 与图像 SHA、四个投影、chosen-view/current-zero 纹理、D-zero 像素分离量和 zero 投影深度、真实及 broken 残差。无需保存所有 patch。相机/雷达信息已进入 D，这不是统计独立的新传感器，而是对原观测关系的显式检验。

评分沿用固定方向、原七个 LSQ-speed bins、四个 future horizon 和原 `y=E_zero−E_D`。主分析是 current CRN owner<0 的源点，其中无同格 radar 为预定重点，其他点照常披露；point 权重仍为 `1/n_original_valid_object`。有/无有效图像、有/无回波及低纹理部分的收益/损害均报对**完整原对象分母**的贡献，不能只报告可投影对象。AUC 只排除零收益且另计数；报告逐场景分布与 null，不声称训练泛化。纹理、深度、D-zero 像素分离量是几何混杂记录；同有效集合报告 D-speed 对照并在原 speed bins 内比较真实/错误对应，避免把大投影位移或偏好的相机区域当运动证据。

预先可证伪的问题是：**正确历史对应相较固定错误对应，是否提供超出速度幅值的收益排序信息，且是否触达无回波误运动代价所在支持？** 若真实/错误对应没有可重复的差别，或差别只随投影分离与纹理缺测出现，不进入 gate 拟合。若可用覆盖很小，继续完整披露其覆盖外代价，不扩大邻域直到出现有利结果。正结果仅支持一种有判别信息的观测联系，不能直接认定运动正确、可部署泛化或占据改善。

最重要的限制：这些 source 是 GT box 内的虚拟刚体材料点，**不是已测到的表面点**；正深度和入画只说明可投影，不说明无遮挡或同一表面。不得用 GT depth、GT mask、框 ID 或成功对应筛选来绕开这一缺口。遮挡、物体转向/加速、重复纹理、曝光变化、透视尺度变化和体素高度歧义均可使真实速度得分差。因此负结果否定这套短时恒速 patch 检验的当前实现，不否定相机中的全部运动信息；正结果也不能单凭 photometric advantage 排除相机投影 shortcut。

来源边界：继承文献审计中 CMFlow §3 的真实两图对应动机及 nuScenes 官方 devkit `b40adc467b919192899405d9b77871afee8efa07` 的逐相机时间投影合同；本草案没有复现 CMFlow 的 RAFT/LiDAR 训练监督，也未再次联网、调用外部模型或执行实验。
