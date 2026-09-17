# 原 CRN-CV 规则的全网格几何

`crn_full_grid_geometry_v1.build_full_grid_geometry(centered_boxes, G0, *, grid_shape_xyz, extent_xyz, box_origin='global_geometric_center')` 只接收当前预测框、当前 LiDAR→global 的 G0 和显式规则网格。必须事先完成一次原点适配；函数不猜测原点、不再次适配、不接收 GT 或未来信息。

返回 `velocity_R: float64[X,Y,Z,3]`（米/秒）和 `owner: int64[X,Y,Z]`。C-order 中 Z 最快；owner 保留原 export 列表 index，无覆盖为 -1。`boxes` 为所有原框的同序数组：`export_index/eligible_GMO/c0_R/R0_R/wlh/score` 及 `detection_name`。非 GMO 框也保留元数据，但不参与所有权。学习方如需 eligible 子集，只能在内部建立映射，不改变该 owner 的定义。

所有权使用原函数的 `(global_points-centre) @ rotation`、wlh→lwh、1e−9 米 closed-face 和严格 score `>`；平分保持先出现的框。无 top-k、置信阈值、2D 高度投影或 GT support。R 中的 box metadata 不代替原 global→local 运算。字段返回新数组，输入不修改。首版全网格逐框计算，与原 reference 的矩阵形状、运算顺序相同，没有 AABB 筛点；真实 200×200×16 成本未知，不能据 synthetic 测试估计批量吞吐。

本地默认测试：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B \
  analysis/o_motion_20260915/test_crn_full_grid_geometry_v1.py
```

六项解析测试包括非方形 XYZ、空框、8 类/非 GMO、score tie/负 score、平移 yaw、pitch/roll、wlh 和所有轴正负闭合面/容差内外点；逐点 owner 与 float64 velocity bytes 均与已冻结 `predicted_object_state_cv_diagnostic_v2.py::predicted_velocity_field` 对比，另有手算几何断言。

仅供主代理审核后执行的真实单锚入口（本次未执行）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B \
  analysis/o_motion_20260915/test_crn_full_grid_geometry_v1.py --real-first-dev-anchor
```

只使用固定 dev[0]，认证 centered prediction complete/ledger 与 raw manifest/单锚 SHA。raw 文件仅提供当前 frame 的 G0；tracks 和未来信息不进入 build。比较完整 640000 点，分别报告新 build、原 reference 以及总比较耗时，不存密集场、不建 train512 cache、不推理或训练模型。NumPy/BLAS 平台差异未作跨平台宣称；真实单锚通过后才有该平台实际等价与成本证据。
