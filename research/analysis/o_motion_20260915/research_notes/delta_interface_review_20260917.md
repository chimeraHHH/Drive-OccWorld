# DELTA 输入接口核查

2026-09-17，GPT-5.6-Luna独立只读核查，主agent对照官方固定源码实施。无skill，无新外部训练。

- 官方Predictor3D:101–155将RGB缩放至384x512（bilinear/align_corners=True）、深度nearest，再在查询点bilinear采样深度。原图phase4/stride32共1400点，与旧RAFT网格的子集一致。
- 输入[past,current]，queries=[time=1,x,y]，backward_tracking=True，grid_query_frame=1。官方反向计算仅替换早于query的输出，再把query时刻的坐标/深度改回输入。当前query深度因此是输入定义，不能把它作为新模型预测收益。
- 官方默认追加36个t=0 support queries，输出时移除；保留默认行为。B=1。window_len=16，两帧输入由官方重复填充至16，不引入14张新观测，输出T仍是2。
- 返回vis是sigmoid>0.9；query时刻强制True。它不同于RAFT FB<=1pixel，二者均不是经本任务校准的概率。报告不筛选全部finite点及官方quality筛选两种结果。
- 返回3D字典只是各时刻相机坐标，且颜色路径不完全匹配query_frame；不用该字典作评分。逐帧用各自K及inv(G0)@camera_to_global变换，以实际相机时间差估计三维速度。
- 评分人口仍为原GT当前框内材料点。两帧图像回溯速度估计不是未来动力学模型，也不是占据结果。
- 输入截止时间沿用已认证原协议input_availability_us，可能略晚于t0 LiDAR。因此表述为截至当前同步包可用的历史输入，不能称严格LiDAR时间戳零延迟因果预测。
- 静止平面+深度偏差仍可导致三维假运动，即使高vis；联合跟踪是否缓解必须实测，不能预先断言。
