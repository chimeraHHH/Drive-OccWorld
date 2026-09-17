# DELTA官方两帧回溯：优于匹配RAFT，仍未胜过已有强对照

2026-09-17。上一轮是有效进展：独立Metric3D＋RAFT在完整原人口失败，从而转向联合时序跟踪。本轮完成了DELTA官方权重接入、H200实际推理、匹配控制及独立评分复算；没有外部模型训练，没有O占据评测。

## 主要结果

**DELTA的两帧联合回溯优于同网格、同深度处理的RAFT，但还不能作为超过O的候选。** 在原CRN覆盖区域保留CV，只在未覆盖区域补入图像速度，2秒水平位移EPE如下（米；对象等权，越低越好）：

| 方法 | 运动对象137个 | 静止对象184个 |
|---|---:|---:|
| 原CRN-CV | 3.022346 | 0.092182 |
| 原CV＋D固定0.5 m/s规则 | 2.737708 | 0.101027 |
| RAFT原深度，stride32＋FB | 3.083256 | 0.405189 |
| RAFT匹配深度处理，stride32＋FB | 3.294486 | 0.390325 |
| DELTA，stride32＋官方vis | 2.997021 | 0.266921 |
| RAFT匹配深度处理，不加FB | 3.597032 | 1.284455 |
| DELTA，不加vis筛选 | 3.019913 | 0.399132 |

原CV＋D仍同时更好。DELTA相对单独CRN-CV的moving改善只有0.025325米，stationary却增加0.174739米。还不能把更高可见率、更平滑轨迹或相对弱图像基准的改善写成实用运动收益。

不使用quality筛选时，DELTA仍优于匹配RAFT，说明结果不只是不同quality阈值造成；但这是两个完整估计器的比较，不能唯一归因为时序深度修正。DELTA的vis>0.9与RAFT的FB<=1像素不是同一种量，也未在本数据上校准。

## 完整人口与支持率

固定原16个train锚点、8个scene、六相机；原41,122个材料点。四个horizon的有效对象数373/366/358/359、有效点数40622/39975/38639/39460，与原基准一致。没有丢掉无表面支持、低置信度或远处对象；缺失点按固定零运动回退，混合臂保留已有CRN覆盖点的CV。

stride32/phase4每相机1400个原图查询，总计134,400个。全部原材料点支持数：DELTA未筛选11508、官方quality9533；匹配RAFT未筛选11138、quality6734。2秒moving对象等权quality支持比例DELTA30.95%、RAFT18.03%；stationary为26.57%/21.68%。这些是不同统计分母，不能混用。

单独图像场、缺失置零时，DELTA的2秒moving EPE为6.655016米，而全零预测为6.664724米，净改善很小。这既可能含深度/对应误差，也含把预测表面插值到虚拟体积材料点时的归属误差和短历史外推误差。目前不能归因单一环节。

评分位置是GT当前框内虚拟材料点，非认证可见表面。它们只在预测完成后用于查询固定三维场与评分，不参与选图像网格、轨迹推理或挑选邻居。没有新future GMO IoU，不能将这些EPE称为O的运动读出。

## 匹配与实现核验

- 使用官方固定commit `3367cda1c74d19e73296165f9826b213211678dd`，核验64个源码文件的Git blob和SHA256。权重来自官方README Google Drive链接，236,969,860字节，SHA256 `7da306765904ec0b02e9cc8a33406250818680a1ad0b63384ae73cd58d4ef6cc`；官方独立校验值未找到，本地SHA是获取指纹。59,174,014参数严格加载，无missing或unexpected键。
- 输入[past,current]，queries=[1,x,y]，官方backward_tracking=True，B=1，window_len=16。官方内部复制填充到16帧，实际只有2帧观测；保留默认36个support queries，输出移除这些点。
- depth仍为同一个冻结Metric3D。DELTA最近邻缩放深度至384x512，再在query处双线性采样；RAFT对照做相同深度变换。96对的当前uv、当前深度及当前参考点逐元素完全一致；重复Metric3D与上一轮输出在所用网格也完全一致。
- RAFT flow地址与FB来自上一轮已认证官方冻结输出，按相同stride32/phase4子集复用；另保留原深度stride32 CPU对照，避免把降低网格密度或深度缩放收益混成DELTA收益。
- 使用每帧真实K和位姿lift到共同t0 LiDAR坐标，再除以真实相机时间差。帧时间戳到t0用估计恒速配准；这不是曝光积分或rolling-shutter建模。
- 输入截止时间沿用原同步包input_availability_us，可能略晚于t0 LiDAR。没有目标未来帧或GT深度输入，亦不声称严格LiDAR时间戳零延迟预测。
- 环境仅在任务私有目录加入PyPI wheel校验后的jaxtyping0.2.36，没有改共享环境；官方源码未修改。官方train()缺少return self导致首次probe的eval链式调用失败，已改为分步调用，失败记录保留。v2单对probe与正式96对均正常完成。

正式推理267.17秒，峰值3.359GiB，无优化更新；runner635069和child635070均已消失。主评分之外，三个控制/候选各独立复算46,592个XY/XYZ误差标量，总139,776个，最大差低于3.2e-13米。源查询/深度/基准人口相同另有逐元素核查。以上验证算术与接入，不等于验证集泛化或因果机制证明。

## 下一步怎样改变研究，而不继续罗列模型

**暂不把DELTA速度接入O，也不训练置信门控来掩盖当前净收益不足。** 保留O/M0、固定seed11和已有CV＋D对照。

下一项有区分力的实验是同一0.5秒历史窗口内的观测密度。首个锚点每个摄像头两个关键帧之间都有5张sweeps图像，文件已确认存在；目前只检查了文件名时间和文件大小，尚未完成sample_data/ego_pose认证，不能立即视为可评分输入。

准备一次明确的2帧vs含中间帧对照：固定当前query、depth模型、标定、原评分人口与回退，保持历史起止时刻不变；官方DELTA分别回溯两端点和完整历史。轨迹终点差分与基于真实时间戳的多点速度拟合分别报告，避免把采样密度与拟合规则混为同一因素。增加了观测数量，故这是信息量诊断，不是等输入超过O的证据；若带入正式模型，必须增加同样历史输入的O控制。

若增加真实历史仍不能带来有意义moving收益并控制stationary代价，就停止扩展外部三维速度路线，投入任务可用未来状态原型，让占据读出直接消费预测状态，并用同输入、同预算的简单增容/未来latent对照检验。若历史信息有效，则将其作为可关联观测状态接入原型，最终仍需原协议future GMO、moving recall与物理运动误差共同支持。目标仍是超过O，当前尚未达到。

产物：[DELTA完整评分](delta_history_train16_evaluation_v1/complete.json)、[匹配RAFT评分](raft_matched_history_train16_evaluation_v1/complete.json)、[原深度降密度控制](raft_original_depth_stride32_evaluation_v1/complete.json)、[源查询一致性核查](delta_vs_raft_source_matching_audit_v1.json)、[Luna接口核查](research_notes/delta_interface_review_20260917.md)。

官方依据：[DELTA仓库与ICLR2025论文入口](https://github.com/snap-research/DELTA_densetrack3d)，[固定Predictor接口](https://github.com/snap-research/DELTA_densetrack3d/blob/3367cda1c74d19e73296165f9826b213211678dd/densetrack3d/models/predictor/predictor.py)。
