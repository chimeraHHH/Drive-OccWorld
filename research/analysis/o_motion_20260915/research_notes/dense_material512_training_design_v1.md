# DenseMaterialState train512 训练协议设计 v1

本文件对应 `analysis/o_motion_20260915/train_dense_material512_v1.py`，只描述一个可审计的训练工件，不给出 O 对比结论。当前 `shared_current_codec512_v1` 必须先完成 2048 个 current-only 更新；本训练读取它的完整 `state_dict`，不读取旧的 train4 fit，也不使用 warm start 以外的参数来源。

## 三个固定 arm

三个 arm 的 encoder 和 decoder 都冻结，只有 `dynamics`、`content_increment`、`velocity` 训练。它们从同一份 codec checkpoint 开始，使用同一个 seed 11 和同一组四轮顺序：单个 `np.RandomState(11)` 连续产生四个 `permutation(512)`，共 2048 updates。

| arm | rendered material | readout |
|---|---|---|
| T | fixed，始终使用 current encoded content | transport |
| J | evolving，使用 recurrent state content | transport |
| D | evolving，使用 recurrent state content | direct |

T 保留 recurrent state 更新，因此 fixed material 只干预渲染内容；它不会删除未来 velocity 所需的隐状态递推。D 与 J 参数边界相同，区别只在 readout。

每步目标是原 world head 的 `CE_ssc_loss + Lovasz`，加上 `0.1 * helper.object_group_loss`。输入是认证过的 native `prev_bev_input[:, -1]`；occupancy loss 使用 raw target 的 `[0,2:]` 五个时间片。target 下采样、trilinear logits resize 和原 occupancy loss 在 CPU 上完成，原 `gather_sparse/object_group_loss` 在 CUDA 上完成；`physical.cpu()` 将物理损失与 CPU 占据损失相加，保留 autograd 路径回传到 CUDA head。未来 target 只进入监督，不进入 model input、codec 初始化或状态更新。

## FROZEN protocol 必须绑定的内容

协议 schema 是 `dense-material512-training-v1`，状态必须为 `FROZEN`。至少绑定以下键，并在启动时逐项拒绝不一致值：

- `training`：`seed=11`、`samples=512`、`epochs=4`、`updates=2048`、AdamW、`lr=3e-4`、`weight_decay=.01`、betas `[.9,.999]`、eps `1e-8`、clip 10、physical weight `.1`、LRU 8、五个 logits/四个 future horizons，以及上述 sample order；`target_slice=[0,2,7]` 明确绑定 raw 的 `[0,2:7]` 五帧。
- `model`：shape `[200,200,16]`、extent `[-51.2,-51.2,-5,51.2,51.2,3]`、class weights `[1,5]`，冻结模块 `encoder/decoder`，训练模块 `dynamics/content_increment/velocity`，以及 T/J/D 的 arm 映射。
- `numerical_policy`：FP32、matmul/cudnn TF32 均 false、cudnn benchmark false、deterministic algorithms 和 cudnn deterministic 均 true、`loss_device=cpu`、`physical_loss_device=cuda`、`downsample_device=cpu`、`cublas_workspace_config=:4096:8`。进程启动前必须已经设置该环境变量，脚本不自动降级或静默替换策略。
- `sources_sha256` 和 `runtime_source_sha256`：绑定训练脚本、DenseMaterialState、DenseTaskState v1/v2、transport、helper、native cache 以及 native/mmdet3d 运行时源码；source hash 在 import 前认证。
- `codec_complete_sha256`、`codec_checkpoint_sha256`：分别绑定 current codec 的 `complete.json` 和 `shared_codec.pth`；complete receipt、manifest、protocol、samples、training、evaluation 文件也逐项验 SHA。codec 必须是 `COMPLETE_CURRENT_CODEC512`、2048 updates、current-only supervision。
- `cache_index_sha256.train`、`labels.manifest_sha256`、`labels.complete_sha256`：绑定完整 512 train cache、712 条稀疏标签 manifest 及其 complete receipt。启动时认证 512 个唯一 sample token、256 个 scene token、train/dev scene disjoint 和原始 source-indexed XYZ labels。
- `resources.train.max_seconds`、`resources.train.max_allocated_gib`：本版每 arm 固定为 `10800` 秒和 `24` GiB；另需检测到至少 `32` GiB free。CLI 上限必须逐项等于冻结协议。

CLI 使用 `--arm T|J|D`，因此三个 arm 可以分别运行并分别只产出一个最终 checkpoint，再由独立 evaluator 统一读取。

## 数据边界、日志和输出

输入 target 文件完整验 SHA 后只复制 raw `[0,2:]` 到 CPU；每个 cache item 保存五个 target、CPU downsample target 和 sparse rigid-box labels，LRU 最大 8 个样本，不缓存 predictions。每次 cache 首次命中写入 `samples.jsonl`，每个 update 同时以 fsync 写入 `sample_order.jsonl` 和 `training.jsonl`，包含 epoch、position、ordinal、sample/scene token、loss 分项、梯度裁剪前 norm 与三类可训练模块的 update norm。

训练不读 development split、不做中间 checkpoint、不 resume、不按 dev 或 best 选择。完成时仅保存 `model_final.pth`，其中包含完整 model state（包括冻结 codec 和未更新模块），并认证 `initial_parameters_sha256`、`final_parameters_sha256` 以及 encoder/decoder 的 `frozen_sha256`。`complete.json` 的状态为 `COMPLETE_DENSE_MATERIAL512_TRAIN`，绑定 protocol、manifest、sample order、training 和 `model_final.pth` 的 SHA。每个 arm 的 initial full-model digest 应与 codec final digest 相等，跨三个独立输出目录可据此认证相同起点。

## 解释边界

这是 512 train samples 上的固定预算未来状态对照，不是泛化结果，也不是 O 对比。物理监督是 sparse rigid-box source-point displacement proxy；它不认证 dense flow。D 的 direct readout 可能绕过位移，J/T 的 transport 仍由 feature splat 产生 occupancy 梯度；因此后续比较应由同一 evaluator 使用固定原材料、fine XYZ 预测和原始 target，报告 occupancy 与支持点物理误差，并保留 zero/reverse 等干预。训练脚本自身不做这些评分，避免训练工件读取 dev 或选择 checkpoint。

主 agent 数值策略复核：occupancy resize/CE/Lovasz 在 CPU，原 helper 的 gather_sparse/object_group_loss 在 CUDA；physical.cpu() 保留梯度后与 CPU occupancy 相加。该完整路径已由 material512_determinism_probe_v2.py 在 H200 实测两次 loss/gradient SHA 完全一致。当前 codec 是合法共同初始化，未加载任何先前 future-fit 权重。
