# 密集历史 DELTA 接口审查

2026-09-17，GPT-5.6-Luna 只读审查，主 agent 对照原始产物核验并补强断言。没有使用 skill。

LSQ 的当前锚点公式、每帧真实 K/pose 到同一 t0 LiDAR 的变换、1400 点查询网格及官方 depth resize 路径未发现数学错误。CPU 检查覆盖不等时间间隔恒速、两帧退化、坐标刚体等变，以及独立 `numpy.linalg.lstsq`。

运行接口的关键约束：`--twoframe` 必须指向原输出的 `delta/` 子目录，评分也必须分别指向 `delta_dense_endpoint/` 和 `delta_dense_lsq/`，顶层完成状态不被旧 evaluator 接受。本轮调度器使用正确的子目录。

补强项已经纳入冻结推理代码：固定密集输入完成文件 SHA；原两帧 protocol SHA；16 个 anchor 数量；首尾 sample_data/K/pose/image 与原记录逐项相等；prev/next 相机链连续；严格时间顺序；图像尺寸；完整 96 条输出计数。CPU 输入复核同样通过，序列帧数为 5 帧×3、6 帧×12、7 帧×81。

endpoint 使用端点可见性；LSQ 使用所有帧可见性及有效性。因此它们是完整估计器比较，不能把收益唯一归为 LSQ 拟合。两者都保留原材料点人口和既定回退，分别报告筛选前后支持率。额外历史输入只用于原端点时间区间，不含未来目标；仍遵循原 sensor-packet availability，不能声称 t0 LiDAR 的严格零延迟。

推理完成后需用保存的原始 UV/depth、真实时间和标定独立重算三维轨迹、速度、t0 点和 masks，并核验全部当前 query/depth 与两帧输出相同。相应检查脚本为 `audit_delta_dense_history_geometry_v1.py`。单序列资源预检已成功：9.60 s、峰值 3.575 GiB、零优化更新；不从该预检报告科学效果。
