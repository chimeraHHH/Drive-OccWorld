# 本次历史姿态插值与官方 nuScenes 接口的边界

2026-09-16，实际读取官方固定 commit `b40adc467b919192899405d9b77871afee8efa07` 的 [`NuScenes.get_boxes`](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/nuscenes.py#L292)。

该实现对 keyframe 返回关联 sample 的原框；对于中间 sensor frame，已存在的同 instance 使用中心线性插值与 quaternion SLERP。它也包含时间钳制和缺前框时返回当前框的逻辑。

本实验的相机都是 keyframe，但我们额外在实际相机时间构造历史插值姿态，并拒绝时间越界和缺框，不作上述回退。因此这是独立定义的刚体插值诊断参考，不能称为 SDK keyframe 输出的等价复现，也不是额外采集的 camera-time 真值。所采用的插值形式有官方实现先例；本次只检验观测地址假设，不声称插值方法创新。

实际几何、有效支持和残差仍由冻结协议定义；本源码查阅未修改既定实验、插值区间、缺失策略或统计口径。
