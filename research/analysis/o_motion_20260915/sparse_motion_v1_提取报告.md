# 稀疏 box 刚体运动监督 v1：真实提取记录

已在本机 CPU 完成固定 train512／development200，共 712 个压缩 NPZ，耗时 **13.0338 秒**，NPZ 总计 **35,207,607 bytes**。未读取模型、预测、原生输入缓存或 fine occupancy GT，未执行 GPU 或优化器。原始 `motion_targets_v1` 未修改。

这些标签是 **t0 原始 GMO box 内的虚拟网格材料点的刚体对应代理**，不是实际观测点，也不是实测 scene flow。它们可以用于单独的运动监督或评价；当前／未来 box、由未来计算的速度分组均不得进入推理输入。

## 文件和完整性链

输出路径为 `sparse_motion_v1/{train,development}/<sample_token>.npz`，与原始选择顺序完全一致。`manifest.json` 的 `records` 按原 ordinal 0–711 排列，每条含 `identity`、`file`、`sha256`、原始标签文件与 SHA、数组 shape/dtype、对象与点统计。必须先核 `complete.json`，再核其绑定的 manifest，最后核每条 NPZ；不能以目录存在代替完成。

| 文件 | SHA256 |
|---|---|
| build_sparse_motion_supervision_v1.py | a63f5083dad4f5092bdc9d4d87b888dd53f54b95986b2f8b3a13769e9e78f023 |
| motion_geometry.py | e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31 |
| sparse_motion_v1/manifest.json | cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe |
| sparse_motion_v1/complete.json | 3d14a03cd61761a1ebc26313463dbff06854907c19c49057667a0ffd257e05bd |
| sparse_motion_v1/statistics.json | 7bff30b77ae1ce1523b6a880cae5a0efc7ffa51c61ded7cf80ecc017268615e2 |
| sparse_motion_v1/material_point_audit.json | 2cd4937cece2ac75d5c70814c4bb1d2ac5df1df276d141d3ec2bdbb0c3b94737 |

原始标签 manifest SHA 为 `4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71`，complete SHA 为 `e79b12f752fc76df4036025728a131c75c55f2994e9ef9058560c3084839ce69`。原始选择 SHA 为 `5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d`。训练 256 个 scene 与开发 100 个 scene 不相交；官方 split 身份继承该已冻结原始标签 manifest 的核验。开发集已有历史暴露，不是新的盲测。

## 数组接口

网格 physical XYZ 为 `(200,200,16)`，边界米单位 `[-51.2,-51.2,-5,51.2,51.2,3]`。使用体素中心，C 顺序为 `flat=((x*200)+y)*16+z`。坐标 R 始终是 t0 LiDAR，不使用未来 ego pose 制造目标运动。

| key | dtype / shape | 含义 |
|---|---|---|
| source_flat_indices | int64 `[N]` | 升序、唯一的 t0 网格点编号 |
| object_index | int64 `[N]` | 点对应保留对象的紧凑索引 0…K−1 |
| instance_tokens | Unicode U32 `[K]` | 依原 instance_token 字典顺序保留 |
| target_displacement_m | float32 `[4,N,3]` | 固定 R 中的材料点位移，x/y/z 米 |
| valid | bool `[4,N]` | 原始对象在相应未来帧有标注 |
| center_delta_R_m | float32 `[4,K,3]` | 同对象原始中心的 R 位移 |
| object_future_valid | bool `[4,K]` | 对象级未来有效性 |
| object_speed_group | int8 `[4,K]` | 0：≤0.1；1：(0.1,0.5]；2：>0.5 m/s；−1：未来缺失 |
| dt_future_seconds | float64 `[4]` | 相对 t0 的实际 sample timestamp 差 |

另含 `schema`、`sample_token`、`scene_token`、`split`、`ordinal`、`label_source_sha256` 标量，以及 `grid_shape_xyz` 和 `extent_xyz_m`。使用 `np.load(..., allow_pickle=False)` 即可读取；没有 object dtype 或 pickle。

速度分组严格按每个 horizon 的 `||c_h(global xy)−c_0(global xy)||/dt_h`，是端点平均速度幅值，**不是累计行程速度**，也不能将第一组直接解释为物理绝对静止。分组在 float64 原始中心上计算，未按存储后的 float32 位移再分组。

刚体标签使用 `inv(G0) @ Ah @ inv(A0) @ G0 @ p0 − p0`，其中 `G0` 是 t0 LiDAR→global，`A0/Ah` 是同 instance 的 box→global。原 wlh 对应 box-local xyz 长宽高 `[l,w,h]`；只做刚体旋转和平移，不用前后 box 尺寸改变缩放材料点。

只有 t0 唯一 box 覆盖点参与：调用既有 `unique_box_assignment`，重叠点全部排除，不做先到先得。为减少 CPU 工作先取带保守容差的旋转 box 外包围盒候选并集；精确内外判断仍由原 helper 执行。t0 box 在网格内没有唯一点时不进入 K，并报告数量；未来新生对象没有 t0 源点。未来缺失时位移与中心差填 0、valid=false、分组 −1，**必须掩码掉填充值**。未来出 ROI 的有效位移保留，统计仅描述未来点中心是否落在半开 extent 外；实际 splat 的边缘权重丢失规则由传输算子决定。

## 实际覆盖

所有数量按 anchor 重复计数，不是互相独立的全局对象或观测点。

| 数量 | train512 | development200 |
|---|---:|---:|
| t0 存在的原始 box | 13,792 | 5,258 |
| 保留对象 K 合计 | 11,540 | 4,288 |
| 无唯一网格点而未保留的 t0 box | 2,252 | 970 |
| 源点 N 合计 | 1,740,053 | 661,476 |
| 排除的重叠网格点 | 14,810 | 4,456 |
| 0.5 / 1 / 1.5 / 2 秒有效点 | 1,709,467 / 1,675,202 / 1,639,983 / 1,608,197 | 652,710 / 642,680 / 632,865 / 621,434 |
| 2 秒有效但移出 extent 的点 | 30,071 | 12,462 |

训练样本 `fb0b52fdbcef4ae7aa0f55547be86360` 的 `N=K=0`；这是唯一没有有效运动监督的样本，仍完整保留其身份和顺序。训练器应对该样本返回与计算图相连的零运动损失，不删除样本、不除以零，也不把空样本当运动正确。其余 711 个样本至少存在一个有效未来点。

## 已执行验证及解释范围

1. 712 个输入 gzip 与其解压 JSON 均核原始 SHA；每个输出 NPZ 保存后按所有数组的 dtype、shape、原始字节回读核验。完成前再次核全部原始标签、NPZ 与源文件 SHA。
2. 前两个真实 manifest 样本 ordinal 0/1 的完整 640,000 点 assignment 与候选加速结果逐点精确一致。这两个 anchor 属同一 scene，未冒称跨 scene 抽样。
3. 对这两个 anchor 全部保留点及有效未来帧，独立使用 Rodrigues 四元数矩阵、显式 global→box0 local→future global→t0 R 路径复算，未复用 `rigid_displacement` 或 `box_to_global`。共 6,984 个有效点×时域对应通过，最大 float32 存储绝对误差为 **9.5308×10⁻⁷ 米**。实际样例、annotation_token 和数值保存在 `material_point_audit.json`。
4. 提取结束后另一次只读检查全 712 个 NPZ 的 SHA、身份、紧凑 object 映射、索引递增、shape、有效掩码映射、缺失填零与速度分组范围，全部通过。

上述证据证明提取与所定义的 box 刚体代理一致，不证明 box 内每个体素都是真实占据，不证明非刚体物体点遵守刚体运动，也不证明任何模型运动精度已提高。覆盖点数受 box 尺寸影响；对象级与点级汇总应分别说明权重。标签的有效性不依赖 fine GT 的 0/255、实际可见性、观测点数或预测置信度。

实际执行命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B analysis/o_motion_20260915/build_sparse_motion_supervision_v1.py --targets analysis/o_motion_20260915/motion_targets_v1 --out analysis/o_motion_20260915/sparse_motion_v1
```

构建器拒绝覆盖已有输出。当前真实输出已完成，不应原路径重跑。
