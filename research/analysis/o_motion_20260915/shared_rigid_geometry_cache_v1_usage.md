# SharedRigid 无损几何缓存 v1

实现：`shared_rigid_geometry_cache_v1.py`，当前源码 SHA256 `d47431242dff6d9dba10fbd791e6516739b410282e60bf7f73470186c90fcaaa`。

只缓存当前 centered CRN 全部原框和当前 LiDAR→global 位姿经冻结 adapter 产生的几何。没有 GT/support 裁剪、重复 origin 适配、阈值、学习或模型调用。压缩不改 dtype、形状或任何数组字节。原 dense CV 的计算路径和 `current_states_numpy` 的对象矩阵路径各自保留。

## 加载 API

```python
from shared_rigid_geometry_cache_v1 import GeometryCache
cache = GeometryCache(cache_root, actual_complete_sha256, selection_path)
g = cache.load('train', 0, train_identity)
g = cache.load('development', 0, development_identity)
```

`identity` 至少含 `sample_token, scene_token, official_index, split`。训练 ordinal 为 0–511；开发 ordinal 为 0–199，其 raw carrier global ordinal 明确是 `512+i`。默认必须有完整 712 样本 terminal complete；QA 单样本仅允许显式 `allow_dev0_roundtrip=True`，不能用于正式训练。

`load` 返回与 `geometry_from_centered_boxes` 相同的几何键：`points_R, owner, owner_original_indices, cv_velocity_R, c0_R, R0_R, wlh, velocity_R, score, classes, retained_original_box_indices, states_numpy, receipt`。`owner` 是保留的 8 类 GMO 顺序索引，`owner_original_indices` 是原 export 框索引。未覆盖 owner=-1，CV=0；保留全部原 GMO 框，未按网格覆盖数筛选。

`states_numpy` 的 `center/rotation/size/velocity/score/classes` 是对应六数组的同对象别名。所有返回数组 readonly；caller 自己负责 device 转换且不得就地改缓存。共享 `points_R` 构造时读取/文件 SHA/数组 SHA 验证一次，以后每次 load 核共享文件 stat 未变；单样本每次 load 都核 NPZ 文件 SHA、dtype、shape、各数组完整字节 SHA、有限值和 owner 映射。

`cache.receipt` 有 `complete_sha256, manifest_sha256, selection_sha256, source_sha256, dependencies_sha256, points_file_sha256, samples, split_counts, train_predictions, development_predictions, raw_manifest_sha256, raw_complete_sha256`。两 prediction receipt 直接保留既有输入认证器字段，含训练 4 项/开发 3 项固定合同 SHA。常量 `SOURCES` 包含全部 5 个直接/传递依赖，连同缓存源本身应加入正式 runtime source 合同。

## CLI（本轮未执行全量构建）

默认 `--mode check` 只认证两 prediction 资产、selection、raw 元数据和源码；不建立输出目录。

```sh
python3 shared_rigid_geometry_cache_v1.py --mode check \
  --selection /ABS/selection_v1.json \
  --train-predictions /ABS/crn_train512_full_official_env_v1 \
  --development-predictions /ABS/crn_state_dev200_centered_v1 \
  --raw-metadata /ABS/motion_targets_v1
```

全量必须显式 `--mode build --out NEW_DIR --max-seconds ROOT_APPROVED_SECONDS --workers 4`。这三个大写值须由根任务实际填写，本文没有授权或承诺全量预算。并行需在启动 Python 前设 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`，最多 4 个 spawn worker。workers=1 为默认。内限由 monotonic 检查执行；单个已进入的 NumPy/压缩调用不是可抢占的，外层 runner 必须给独立 hard timeout/cleanup。失败无自动重试，已有输出目录一律拒绝。workers 只返回小 receipt，不回传/累积主进程完整几何列表。

目录：共享 `points_R.npy`；每样本 `samples/{split}/{ordinal:04d}_{token}.npz` 与 `.complete.json`；终点 `manifest.json`、`complete.json`。NPZ 含上述除 points/states/receipt 外的 10 个数组；shared points 为 float64[640000,3]、XYZ C-order/Z 最快。每个 NPZ 写后重新解压逐数组 bytes 比较，再发布单样本 completion。终点 manifest 绑定所有样本 NPZ/array/completion hashes、原框记录 digest、原 carrier compressed/uncompressed SHA、当前 G0 SHA、原 geometry receipt、预测资产/selection/raw 元数据与全部 source SHA。complete 绑定 manifest 和 points 文件；构造器核全部小 completion，NPZ 懒校验。

## 已执行的本机唯一真实往返

输出 `shared_rigid_geometry_cache_dev0_roundtrip_v1/`。固定 development[0]，token `297c52902a384b84ba179f3160ac54e9`，原 raw ordinal 512。NumPy 2.4.4，1 worker，OMP/OPENBLAS/MKL=1；CLI max_seconds=120 仅是本机这一个 QA 的上限。

实际返回码 0；构建加实际加载器验证 4.606 秒。全部 640000 点及 10 个数组字节精确，六状态别名精确，Torch 未导入。432 原框、361 GMO 框，13159 格点有 owner；NPZ 111282 bytes，共享 points 文件 15360128 bytes。这是单锚点存储工程数据，不代表服务器 712 样本速度。默认 formal 构造器对这份 QA 明确拒绝。尚未执行正式 712 构建、SSH、GPU 或训练。

QA complete SHA：`7f85c43537c174bf34c6071d6650d2723e6d94ab694a3b47aa66870c8b521d1b`。实际 loader 往返收据为该目录 `roundtrip.json`；完整源码/静态核对收据另存 `receipts/shared_rigid_geometry_cache_local_v1.json`。
