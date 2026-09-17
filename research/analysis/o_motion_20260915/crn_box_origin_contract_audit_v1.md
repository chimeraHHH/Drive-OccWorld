# CRN 导出框原点与材料点覆盖诊断

已确认一项代码链坐标错误：CRN 训练使用中心 z，继承的解码器将它改为底面中心，导出器却把底面中心直接作为 nuScenes 框中心。原 v1 分数、框、尺寸、分数排序与评价分母均未修改。本报告只解释原始覆盖，不包含修正后的预测或 EPE。

## 可核查源码链

官方仓库为 [youngskkim/CRN，固定 commit 5e9d2fa](https://github.com/youngskkim/CRN/tree/5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d)。本地相关文件与该 commit 逐字一致；实际 L40S 原解释器与 PYTHONPATH 下重新只导入所得到的文件路径和 SHA 也一致。mmdet3d 为 1.0.0rc4，`BEVDepthHead.get_bboxes is CenterHead.get_bboxes` 为真。

1. `datasets/nusc_det_dataset.py:678–690`：由 nuScenes annotation 构造 `Box`，变到 CRN ego frame，直接把 `box.center` 写入前三维，没有减半高。原材料点 GT 则使用全局 annotation center 和 local `[length,width,height]/2`。
2. `layers/heads/bev_depth_head_det.py:228–272`：该自定义 target 函数直接取输入框 z，写入 height 回归目标，没有调用底面框的 `gravity_center` 再加半高。
3. `exps/det/CRN_r50_256x704_128x128_4key.py:234–242` 指定 `CenterPointBBoxCoder`；其 `centerpoint_bbox_coders.py:167–191` 将回归 `hei` 原样拼入解码框。
4. `models/camera_radar_net_det.py:15–31` 继承 BaseBEVDepth，使用 BEVDepthHead；`models/base_bev_depth.py:138–151` 将 `get_bboxes` 直接委托给该 head。继承的 `centerpoint_head.py:718–723` 执行 `z -= h*0.5`，再构造 `LiDARInstance3DBoxes`。该类默认 origin `(0.5,0.5,0)`，`.tensor` 保留底面坐标；`.gravity_center` 才会加回半高。
5. `exps/base_exp.py:307–330` 的真实 `eval_step` 取 `.tensor`；保留的官方评测入口 `evaluate_official.py` 调用此函数，并把其结果交给原 evaluator。
6. `evaluators/det_evaluators.py:257–267,289–295` 直接用 `center=box[:3]` 构造 nuScenes `Box`，随后执行 ego→global 并导出 `nusc_box.center`。此处没有底面→中心转换。

源文件的绝对路径、完整函数文本、SHA 和原官方任务收据保存在 [runtime 回执](receipts/crn_z_origin_runtime_v1.json)；额外 dataset/config/coder/模型来源核验见 [source parity 回执](receipts/crn_z_origin_source_parity_v1.json)。这证明当前保留的原环境和源码导入链，不是假称能读取已结束历史进程的 `sys.modules`。导入未构造模型，CUDA 未初始化，没有重新推理。

## 语义逆变换

令导出中心字段为 \(t_e\)、导出旋转为 \(R_e=R_{ego}R_z(\psi)\)、框高为 \(h\)。应恢复的中心为

\[
t_c=t_e+R_e(0,0,h/2)^\top.
\]

因为 \(R_z(\psi)e_z=e_z\)，这正好抵消解码器在 ego 坐标中的减半高，再随 ego→global 旋转后的位移。该式完全由数据表示转换推出，不使用 GT 匹配或拟合。不能只加 global z：ego 的 pitch/roll 已进入导出 quaternion。原 float32 减法有舍入，故这是语义逆变换，不保证逐 bit 恢复尚未导出的回归中心。

## 原始框的真实覆盖分解

独立 NumPy 运算耗时 5.52 秒。全部 200 anchors 的原框 owner 索引 SHA 与 v1 评分一致；16,074 个对象时域的分母和原 XYZ 覆盖计数逐项相同。4288 个去重 anchor-instance 的全部 661,476 个材料点均在其原 t0 中心定义 GT 框内，不存在这里把 GT 高度当半高的错误。

这里 XY 指**仅删除同一全四元数框的 local-z 不等式**，即沿框局部 z 无限延伸的棱柱；不是修改预测器，也不是重新定义一个全局水平投影模型。

| 固定 2 s moving 支持 | 点数 | 比例 | 对象 any / full / none |
|---|---:|---:|---:|
| 全部原支持 | 132148 | 100% | 1226 |
| XY 覆盖 | 110057 | 83.2831% | 1099 / 484 / 127 |
| 原 XYZ 覆盖 | 55186 | 41.7608% | 1061 / 7 / 165 |
| XY 有覆盖但被 Z 排除 | 54871 | 全点 41.5224% | — |

Z 排除占原未覆盖点的 **71.2962%**，占 XY 覆盖点的 49.8569%。其中 54,775 点在所有横向覆盖框的上方，仅 96 点在下方，没有上下混合情形。这与底面中心被当中心造成框整体偏低的代码证据一致。全部去重材料点对应的原 XY/XYZ 覆盖为 78.5471%/38.2232%，完整分组和每对象记录见 [几何结果](crn_coverage_geometry_dev200_v1/summary.json)。

不能据此认定全部运动误差都由原点问题造成。XY 本身仍缺失一部分源点，框尺寸、平面位置、速度、重叠归属及刚体近似误差仍存在。本报告没有计算任何修正后覆盖、CV 误差或新置信区间，也没有把 GT 匹配变成预测输入。

脚本：[crn_coverage_geometry_diagnostic_v1.py](crn_coverage_geometry_diagnostic_v1.py)，SHA `f9efe47f1195eac9083e1f8a0151ba6d5b15243a58d58c3836606be914a0efc5`。结构化源码与结果绑定见 [audit JSON](crn_box_origin_contract_audit_v1.json)。
