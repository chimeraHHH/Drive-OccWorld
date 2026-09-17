# Cpl/Fix 当前预测几何适配

```python
geometry = geometry_from_centered_boxes(
    centered_boxes, G0,
    grid_shape_xyz=(200, 200, 16),
    extent_xyz=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
)
```

模块只接收当前已归一到 geometric centre 的预测框、当前 G0 和规则网格。它不读取数据文件、GT mask、稀疏材料点、未来姿态或权重，不再次做原点适配，无 Torch 导入。数据文件/identity/source provenance 由现有 `PredictionInputs` / `DevelopmentPredictionInputs` caller 认证。

- `points_R`、`cv_velocity_R`：float64 `[Q,3]`，XYZ C-order、Z 最快，全格无筛选。后者与旧 CV 全点矩阵路径一致。
- `owner`：int64 `[Q]`，-1 或 retained GMO 顺序中的索引；`owner_original_indices` 保留原 export 索引。
- `retained_original_box_indices`：int64 `[M]`，原顺序全部8类 GMO。无 score/top-k/覆盖数/ROI 筛选。
- `c0_R/R0_R/wlh/velocity_R/score/classes`：直接对应原 `current_states_numpy` 的 `center/rotation/size/velocity/score/classes`，不经 float32 重建；`states_numpy` 同时保留原六字段。二者为数组别名，caller 不应单独修改别名。
- `receipt`：依赖/本源 SHA、网格、当前 G0 SHA、原索引映射、覆盖数、完整格生成耗时和总耗时。

刚体模块调用方将 `owner` 对接 `[M]` 对象，使用 float64 原几何；只有原 packed state 路径转 float32。不能直接拿 `owner_original_indices` 去索引 retained 对象，也不能用 BEV 的 `y*X+x` 顺序解释 XYZ 全格。

测试：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B \
  analysis/o_motion_20260915/test_shared_rigid_prediction_geometry_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B \
  analysis/o_motion_20260915/test_shared_rigid_prediction_geometry_v1.py --real-first-dev-anchor
```

默认三个解析测试覆盖非连续原框索引、非 GMO、score tie、空集、3D 旋转，以及 owner 往返映射。真实入口固定 dev[0]，核 centered 资产与原 carrier 字节，只将 frame2 当前 G0 送入 geometry；逐点比较全部640000个原 owner 与 CV field bytes，并逐数组比较全部原状态 bytes。单锚耗时只代表本机该锚 CPU 几何，不代表512锚或GPU训练吞吐。此模块不创建全train缓存。
