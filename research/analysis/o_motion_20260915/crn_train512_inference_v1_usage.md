# CRN train512 官方权重预测资产：待审核入口

这是新资产准备与冻结官方权重推理，未运行、未训练。原始 checkpoint、代码、环境、val 导出均不修改。实际 L40S 检查确认官方权重 SHA 正确，但 CRN 专用完整 train infos 尚不存在；Drive-OccWorld temporal pickle 不可替代它。

先运行 `prepare_crn_train512_inputs_v1.py`。准确准备 argv、目录与已知源 SHA 在 `crn_train512_execution_plan_v1.json`，状态为 REVIEW_REQUIRED。新根拟为 `/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/crn_train512_state_v1`；package 包含两个新脚本、原点适配器、25文件源合同和原 `selection_v1.json`。CUDA_VISIBLE_DEVICES 为空；使用原 CRN Python 3.8 及 mmcv/nuscenes 依赖。建议 CPU inner1800/outer1860 秒，两投影线程；这只是上限，尚无本轮实际吞吐。

该入口原样调用作者 `generate_info(nusc, splits.train)`，保留完整 28130 样本原场景遍历顺序。随后按固定 train512/256 场景定位原索引，按原 `[0,-2,-4,-6]` 同场景回退规则取历史并集，最多2048帧。只对这部分调用原 radar_bev 和 radar_pv worker。投影输入删除 ann_infos，NuScenes 查询限制为 sample/sample_data/ego_pose/calibrated_sensor；保留原8 radar sweeps、过滤、原 compensated velocity×lag 和相机投影。完整 infos 内作者原本的 annotation velocity 仅保留为 schema，绝不进入推理 forward。

数据准备输出 `assets/manifest.json`、`complete.json`、`history_plan.json`、完整 `data/nuscenes_infos_train.pkl` 和所需 radar_bev/radar_pv 文件。原 samples/sweeps/metadata 为指向既有只读数据的链接。每个投影文件的 bytes/SHA 均在 manifest 内；失败停止，不覆盖、不自动恢复。

随后，`crn_train512_inference_v1.py --mode pilot` 在固定 selection train[0] 与首个不同场景 anchor（即 train[2]）上作真实工程检查。需要参数：

```text
--assets NEWROOT/assets
--assets-complete-sha256 ACTUAL_COMPLETED_ASSET_SHA
--selection NEWROOT/package/selection_v1.json
--repo /home/huayiming/Workspace/RadarFlowOcc/code/radar-fusion-baselines/CRN
--checkpoint OLDROOT/official_assets/CRN_r50_256x704_128x128_4key.pth
--source-contract NEWROOT/package/crn_train512_source_contract_v1.json
--source-contract-sha256 e147d9deca1409ee060a1ceb384f89424b55a9c40a866576b638f92f7f414e5a
--gpu-index ACTUAL_IDLE_INDEX --gpu-uuid MATCHING_UUID
--out NEWROOT/pilot --max-seconds 600 --max-allocated-gib 16
```

命令中的未完成资产 SHA 和执行时空闲 GPU 必须在真实准备成功后填写，不能预填假值。脚本先验源码、资产、checkpoint 和 GPU UUID/无 compute PID，再载模型。GPU 建议 pilot inner600/outer660秒、16GiB；实际初始化、冷/暖数据加推理时间及峰值返回后，根任务冻结 full512 预算。脚本 full 上限1800秒，仅为防越界，不代表已授权派发；没有自动 pilot→full 或自动重试。

正式入口加 `--mode full --pilot NEWROOT/pilot`，必须有同源、同资产、同权重的真实 pilot complete。Dataset 内部仍是完整 train infos，`Subset` 只把原索引传给 `__getitem__`，不切坏历史索引。使用原 val/eval pipeline、原4 workers与原官方推理 seed0；这不是新增训练 seed。原 pipeline 在超过1536 radar点时，即使 eval 也会随机抽样，该算法保留，因此不声称两样本 pilot 与 full 中相同样本逐 bit 相等。未引入多 seed 搜索。

模型输入仅 image、mats、t0 metadata、radar；原 GT batch 位置全部替成 None。没有 `backward`、optimizer 或官方 GT metric 调用。只调用原 `_format_bbox`，保留原 NMS 后全部导出框、分数、类别与原排序（包括作者原 top500）。不新增 score 或类别筛选。

输出：

- `raw_export/results_nusc.json`：原作者导出，translation 是本次已核实的 global 变换后的底面中心，原字节另存。
- `predictions.json`：schema `crn-state-train512-predictions-v1`，records 每条含 ordinal/sample_token/scene_token/official_index/split/boxes。只通过冻结 `crn_box_origin_adapter_v1.py` 改 translation 为几何中心，完整保留其它字段和顺序；顶层 `box_origin='global_geometric_center'`。**消费此文件不得再次加半高。**
- `manifest.json`：source/selection/asset/checkpoint/origin-adapter SHA、原 train 索引、固定参数摘要、预测框数、耗时与显存。
- `complete.json`：full 状态 `COMPLETE_CRN_TRAIN512_INFERENCE`，pilot 状态 `PASS_CRN_TRAIN512_PILOT`；绑定上述三份文件、资源对应来源和样本数。终点实际 hash 尚不存在。

截至交付，仅完成 Python3.8 AST/CLI 检查及25份L40S源码字节核验；没有生成真实 train infos/radar投影，没有构造或运行本轮模型，也没有任何预测质量结论。

执行路径修订：推理入口在任何 author imports 前显式将已校验的 repo 与其 vendored mmdetection3d 放入 sys.path，不再仅依赖 cwd。兼容依赖仍使用原 CRN Python/PYTHONPATH；父级 hard timeout 必须保留。唯一源码差异为这两行导入定位。

当前 inference SHA `b77f18b1ba9af4a29213ad3a17048f55ebd0259abb5a788147e748661ff8e290`；prepare SHA `75b5bba81c59348f9c0f190a34065d84ed465814e2cbd98f84578a823dff6e38` 未变。
