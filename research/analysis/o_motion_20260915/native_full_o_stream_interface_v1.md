# 原生 full O 流式接入接口（尚未 GPU 预检）

`native_full_o_stream_v1.py` 只建立可复用输入组件和固定两锚工程入口。它不训练、不加载 J/D、不派发任务，也不把两锚通过当成 full5119 完成。

## 公共接口

```python
stream = NativeOStream(
    config=S0, checkpoint=M0, o_checkpoint=O_final, repo=native_repo,
    runtime_contract=runtime_json,
    selection=full_validation_selection_v1_json,
    reference_dir=original_full_summary_directory,
    device='cuda:0', ordinals=ordered_official_ordinals,
    max_allocated_gib=32, check=caller_deadline_function)
frame = stream.fetch(ordered_official_ordinals[0])
# frame['O_prediction']: native float32 [5,3,1,1,40000,16,2]
# frame['t0_bev']: measured float32 [1,40000,256], never a future feature
# frame['sample']: separate inputs / targets / record
# stream.model: frozen O, available for its original evaluate_occ_records
# J/D receive the same frame; targets never enter their prediction inputs.
stream.verify_frame(frame)  # after downstream work, before discarding
# Discard each frame before requesting the next; do not collect tensors in a list.
del frame
# Once every planned ordinal is processed:
receipt = stream.finish()
```

默认 ordinals 是官方 0…5118，150 场景；显式子集或分片必须严格递增、唯一，`fetch` 必须遵从给定顺序，无替换、重试。调用者负责最终分片互斥/全量覆盖和结果汇总。本组件只持有模型、dataset 和小型参考数组，不持久化 inputs、GT、BEV 或 logits。

一个严格加载的冻结 O 模型先按原生 `_prepare_data_info(..., rand_interval=None)` / collate / `_capture_native` 路径，在 matmul TF32=True、cuDNN TF32=True 时捕获观测边界。首次完整 O rollout 仅用于完成 capture，**不能称为 M0 输出，也不用于计分**。随后同模型在 matmul TF32=False、cuDNN TF32=True 下 replay，返回计分 O。没有 loss adapter、memory adapter 或坐标改动。

原生 dataset 会在预测前读取 GT；准确边界是 targets 与 inputs 分开、GT 不作为未来预测或 t0 BEV 的输入。输入内保留原 prescribed ego/action conditioning，不能误称只有传感器张量。

## 固定两锚 CLI

```sh
python native_full_o_stream_v1.py \
  --config /storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/runtime/configs/S0.py \
  --checkpoint /storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/assets/m0_epoch24.pth \
  --o-checkpoint /storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915/campaign_objective_v1/runs/O/latest.pth \
  --repo /home/wangning/Workspace/RadarFlowOcc-sota-p2 \
  --runtime-contract runtime_source_contract.json \
  --selection full_validation_selection_v1.json \
  --reference-dir /storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915/campaign_objective_joint_full_v2/summary_v1 \
  --pilot-selection joint_preflight_selection_v1.json \
  --parity-cache /home/wangning/data_cache/RadarFlowOcc_m0_improvement_20260915/campaign_cache_v1/development \
  --o-development-records /storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/m0_improvement_20260915/campaign_objective_v1/runs/O/development_records.jsonl \
  --out NEW_DIRECTORY --device cuda:0 --max-seconds 600 --max-allocated-gib 32
```

脚本依赖同包五个冻结源（本机也允许兄弟 P 目录解析）：`native_state_cache.py`、`joint_native_evaluation.py`、`objective_joint_evaluation_v2.py`、`observation_memory.py`、`memory_experiment.py`。后两者仅使用已冻结源码/摘要函数，不安装 memory 干预。非 Python 包资产是两份 selection 与 runtime_source_contract；权重、原全量参考目录、旧开发缓存和 O 开发记录使用上列服务器既有路径。

固定两锚是 official_index 2036、2998（两个场景）。每锚核原 full 的 GT 字节、GT counts、观测 BEV/radar 字节、O 全五时域三层 logits 字节和五时域完整 confusion。对旧 dev cache 另核：除原先证实 inactive 的 `plan_dict.sample_traj` 外，全部 typed 边界严格相同；实际输入不替换，双方完整 hash、例外叶 hash/type/shape/dtype/差值完整记录。planner 必须关闭，唯一 consumer 原源码/AST 被锁。完整历史 full 只有完整树 hash，因此一般 full 样本不声称已进行历史 active-tree 逐叶比较。

旧 dev cache 没有 O 的历史全层 logits；预检比较的是**本次**同一 O 从 fresh 边界与旧缓存边界 replay 的全五时域三层输出。历史 O confusion 独立与冻结开发记录精确核对。

输出小文件：manifest.json、loaded_model.json、records.jsonl、summary.json，complete.json 绑定四件 SHA。异常写 failed.json 并退出，无自动重试。资源 summary 的 max_sample_seconds 包含额外旧缓存重放，适用于工程预算估计；elapsed_seconds 还包括前置文件/源码核验。GPU 实际耗时/峰值和两锚数值门仍未执行，不能据本机 CPU 检查估计正式 full 的最终预算。

已完成本机有限检查：Python 3.10 语法兼容、无 Torch 的 --help/五依赖导入、真实 full5119/150 的四件文件 SHA、NPZ GT/confusion 方向与行和、官方身份/旧 data_info_index 映射、原两锚选择。没有生成 mock 成果，也没有运行 Torch/CUDA 前向。
