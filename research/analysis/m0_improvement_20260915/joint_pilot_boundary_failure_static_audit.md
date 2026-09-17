# Joint pilot v1 原生边界失败静态审查

本审查只读源码、缓存索引和失败回执，未连接服务器、未前向、未训练，未改任何冻结源。两个 shard 完成数均为 0；已读取 shard 0 的真实 `Fresh observed boundary differs from frozen pilot cache` 回执。该错误出现在完整树哈希比较，位于 GT、原生 logits 和模型 confusion 的后续 parity 门之前。因此目前不能说预测或指标不同，也不能说只是不影响预测的 metadata 差异。

## 直接代码事实

| 项目 | cache 提取 | joint native stream | 结论 |
|---|---|---|---|
| 数据集 | `cfg.data.test` 深拷贝，`test_mode=True`，去 `samples_per_gpu`；development 不替换 ann_file | 同样处理 `cfg.data.test` | 声明的 recipe 相同 |
| 实际索引 | `_selected_rows` 将 usable index 转为 data_info index；`_prepare_data_info(..., rand_interval=None)` | 复用同 helper 和同调用 | 两个 identity 已固定；不是通过 `__getitem__` 随机替换 |
| 捕获 | 同一 `_capture_native`，原 `future_pred` 调用前深拷贝完整输入；GT 分开存 | 直接复用 | 没有有意丢弃/新增输入字段 |
| 原 M0 | 同一 `build_native_model`、epoch24、strict state_dict、`.eval()`、所有参数冻结 | 同上；另外提前 deep-copy 3 个单槽 replay 模型、加载 final heads | M0 值流未有意改动，但显存分配环境不同 |
| forward mode | `MMDataParallel` + `no_grad` | 同上 | 原 `forward_test` 也先 `self.eval()`；history 路径不会无条件转回 train |
| 观察器精度 | matmul TF32=True、cuDNN TF32=True、benchmark=False | 原生 forward 前相同；仅后续 replay 改 matmul=False | 当前门失败前不存在有意 TF32 差别 |
| 全局随机数初始位置 | `native_state_cache.main` 在 mmcv/mmdet3d 数据集导入之前 seed11 | joint run 在这些导入及 CPU final-source 审计之后 seed11 | 导入期间是否实际消耗 RNG 未证实，不能仅凭同 seed 宣称流一致 |
| 样本流 | dev200 持续抽取并逐条 replay parity | 两 shard 各从 seed11 开始，pilot 各读一条 | RNG 历史与完整 cache 不同 |

真实 devcache index 确认：

| sample token | official index | 原 cache ordinal | joint shard 本地 ordinal |
|---|---:|---:|---:|
| `297c52902a384b84ba179f3160ac54e9` | 2036 | 0 | 0 |
| `01076466d815421c857d2c8cf8034488` | 2998 | 2 | 0 |

**因此，仅“跳过先前样本导致 RNG 不同”可以解释第二条，不能单独解释第一条。**

## 优先检查的具体机制

### 1. 关闭规划时仍生成并捕获随机候选轨迹

`nuscenes_world_dataset_template.py:492` 在当前帧加载时无条件调用 `get_trajectory_sampling`，没有 `turn_on_plan` 条件。该函数调用 `samplers/sampler.py:22` 的 `sample(..., M=1800)`；后者在第 42、46、47、57、123 行使用全局 NumPy RNG 采样加速度、速度、曲率相关参数和选择变量。

`drive_occworld.py:1031` 随后仍把这批值放入 `plan_dict['sample_traj']`。它是完整输入树的一部分，typed `tree_digest` 理应计入。原生 `future_pred` 中该字段的取用仅在 `if self.turn_on_plan and occ_flow == 'occ'` 分支（第 554–556 行）；M0 builder 强制关闭规划。真正使用的 `plan_dict['gt_traj']` 仍是未来 action/几何条件，不能一并忽略或替换。

这是“输入树不等但有效 M0 前向可能相同”的具体候选机制，尚不能认定为本次唯一原因。尤其第一条失败仍需要实际叶差异和 RNG 来源闭环。

对完整冻结 detector 的 AST 静态枚举还确认了以下数据流：

- `nuscenes_world_dataset_v1.py:253–254` 将当前帧 `sample_traj` 原值转交 `ret_queue`；不是从它构造真实 ego trajectory。
- 同文件第 235–249 行从 `sdc_planning` 标签形成真实未来位移，第 259–267 行从真实 ego 位置/角度与 canbus steering 形成 `vel_steering`。两者独立于随机候选轨迹。
- `forward_test:1029` 创建的 action dict 只有 command 和 vel_steering，第 1031 行将 sample_traj 与真实 gt_traj 分别放入 plan dict；不存在二者替换。
- `future_pred` 的全部 `plan_dict` 名称读取仅六处：第 482 行 ref_pose_pred、第 500 行 gt_traj、第 556 行 sample_traj 和 gt_traj、第 562/567 行 sem_occupancy。sample_traj 的唯一读取受 `self.turn_on_plan and occ_flow == 'occ'` 严格支配，没有把整个 plan_dict 传入其他函数。
- 第 500–503 行把真实 `gt_traj` 切片赋给 action dict 的 `plan_traj`。第 537/546 行调用未来 head，只传 action/cond 两个字典及空间引用等，未传随机候选轨迹。第 554 行的关闭分支内才读取 sample_traj 并可能调用 plan_head。
- 原生单槽适配器仅增添输入维度/无 GT 条件守卫；其 `_validate_inputs` 不读 plan_dict。capture/replay 的通用树复制会复制 sample_traj，但不把它交给 occupancy 计算。

因此，在冻结 source、`turn_on_plan=False` 且其他有效输入相同的条件下，静态代码支持“改变 sample_traj 值本身不会改变本次 occupancy 前向”。**这一条件性结论不允许删去数据生成过程**：生成轨迹仍消耗全局 RNG，可能影响之后的随机操作；本审查也未据此放行实际失败 gate。

### 2. eval 路径也可能推进 NumPy RNG

`models/utils/grid_mask.py:86` 为 `if np.random.rand() > self.prob or not self.training`：Python 从左到右求值，即使 eval 直接返回图像，仍先消耗一次 NumPy 随机数。它不会因此启用 eval 图像掩码，但会影响后续样本的轨迹随机数流。`_prepare_data_info(..., rand_interval=None)` 也调用 `np.random.choice(self.rand_frame_interval, 1)`；实际 interval 配置应原样记录，不能直接替换成固定值。

### 3. 数值路径/源文件差异仍待叶差异排除

两个脚本同为 `.eval()`、`no_grad`、TF32=True；`observation_memory.install_observation_memory(..., 'native1')` 保持原参数，并没有 NumPy 随机初始化，原 native model 不安装该适配器。没有发现会把 M0 变成训练模式或把 F/O loss adapter 装到推理模型的代码。

不过，多模型副本及临时投影副本改变 CUDA 分配环境；benchmark=False 本身不是所有 CUDA 算法的逐位确定性证明。如果差异叶包含 `prev_bev_input`，应量化逐元差异并查 observer 参数/缓冲区与实际后端状态，不能预先归因于无用轨迹。

当前 runtime SHA ledger 锁住了 detector、BEVFormer、雷达和 `nuscenes_world_dataset_v1.py` 等文件，**没有单独锁住** `nuscenes_world_dataset_template.py`、`samplers/sampler.py`、`grid_mask.py` 及第三方 dataset/pipeline 依赖。这是溯源覆盖边界，不是已证明源码漂移。真实诊断应追加这些实际导入文件的路径/SHA 和版本，且不能把今日新采集的 SHA 冒充旧 cache 提取时已记录的 SHA。

本地三个补充源码 SHA：

- template：`9de0dfbba0b77567cf011fb0adb163492e8d5e6f510040b95626b3bcded822cc`
- trajectory sampler：`65b7cea1de9e6563674b6faf80ba809b2497083fe3a5141f92d8ea9e9beee95c`
- GridMask：`92589caffcd8102b02cafff78e64614ecf8bc8ad548ab95cbf4d1a898327f150`
- 以上只是本地现有文件，服务器实际导入版本须实测。

## 同两条样本只读诊断应记录什么

Root 提议保持原 `ObjectiveEvaluation` 私有 run、完整模型分配和原 gate，在第二次 `tree_digest`（旧缓存输入）时做 observer diff，并仍返回原哈希让原门失败，这能保留当前运行环境，适合第一轮定位。

1. 对完整树逐叶列出 path、容器/标量类型、tensor/array dtype 和 shape、双方 typed hash、是否逐位相同。浮点叶另外给逐元 unequal count、max/mean abs diff、最大差异坐标与有限性；区分数值相等但 dtype、正负零等字节不同。不要只输出第一条差异，也不要删去 metadata。
2. 单独摘要 `prev_bev_input`、`radar_bev`、`action_condition_dict.*`、`cond_norm_dict.*`、`plan_dict.sample_traj` 与 `plan_dict.gt_traj`、两套 metadata。小标量和 4×4 几何矩阵可完整列出；大张量只写摘要和少量差异坐标。
3. 在不放行原 gate 的前提下，旁路观察当前已经算出的原生 GT 和 `[5,3,...]` logits，与已验证 SHA 的 cache 文件逐位比较；分别记录 7 帧 GT 和 5 时域×3 层 logits，原生 M0 五时域 confusion 也独立精确比较。它们只是诊断结果，不能写“pilot PASS”。
4. 记录 Python/NumPy/Torch CPU/CUDA RNG state 摘要、实际 `rand_frame_interval`、pipeline transform 名称、model/submodule training flags、TF32/benchmark/deterministic 状态、torch/CUDA/cuDNN 和设备身份。RNG 摘要本身不得抽随机数。
5. 记录实际模板、sampler、GridMask、加载 pipeline 源码路径和 SHA；保留原已冻结 source gate。此次新证据只用于解释失败。

若只有 `sample_traj` 不同、全部有效状态/action/GT/原生 logits/hist 完全一致，可支持“未启用规划支路的随机载荷造成过强边界相等要求”的解释，但修订下一版合同仍应由 root 明确决定并记录，不能在本次 observer 中跳字段。若 `prev_bev_input` 或有效 action 不同，则需要进一步定位观察器/随机数/数据来源，不应进入 full。

## 本次核验的冻结文件

- native_state_cache.py：`41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0`
- joint_native_evaluation.py：`280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578`
- objective_joint_evaluation.py：`ab0b80aa14ecd8587077e20cf9fbc06c6c8df724f7e0a3c8c3abece2285ee9c7`
- observation_memory.py：`0f01a29dc79daf2f0abb42bd2a23f8d168606cd3928212229ee5aeb3f94a5733`
