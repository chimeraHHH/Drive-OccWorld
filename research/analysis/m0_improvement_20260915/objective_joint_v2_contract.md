# Objective joint v2：历史缓存中未启用规划的随机叶

这是独立新版本，不修改 v1、旧 cache、训练结果或实际模型输入。v1 两个 pilot shard 完成 0 条；随后同一原始失败门的只读诊断发现第一条样本仅 `inputs.plan_dict.sample_traj` 不同，其他叶、完整 GT、原生 5 时域×3 层 logits 和 M0 confusion 完全相同。诊断 SHA 为 `aecf29305e3d4b086e834df4b391580e0964a330a22f4d91c09566b4036b95a5`。第二条样本尚待 v2 pilot 严格检验，不假定它必然通过。

## 唯一比较例外

原始 native loader 仍正常随机生成并传递 `sample_traj`；fresh 输入不会被缓存值替换、清零、去掉字段或重新播种。所有模型继续复用同一个完整 fresh 输入，前后用原完整 typed hash 检验没有 mutation。

只有在 pilot 对照历史 cache 时，使用两层字典浅拷贝建立比较投影，投影中移除唯一精确路径 `inputs.plan_dict.sample_traj`。其余所有 typed 叶必须精确一致。该叶本身要求双方同一 Python 类型（Tensor 或 ndarray）、float64、形状 `[1,1800,5,3]`、全部有限；分别保留双方完整树哈希、投影哈希、叶哈希、类型、dtype、shape、元素差异计数及绝对误差摘要。

运行时每个实际模型必须 `turn_on_plan=False`、所有模块 eval；原 M0 consumer 的整文件及函数 AST 必须仍为冻结版本，并静态检查唯一 sample_traj 读取被关闭的规划条件支配。GT、M0 全 logits、每个最终模型自己的五时域 dev confusion 原精确门保持。Full 仍使用完整 fresh 输入，根本不执行 cache 投影比较。

## 协议新增字段

协议 schema 为 `objective-joint-evaluation-protocol-v2`，status 保持 `FROZEN_BEFORE_OBJECTIVE_JOINT_PILOT`。父/训练协议、候选选择规则、完整五模型开发结果、资源天花板和 selection 约定与 v1 相同。新旧 producer/merger 与 `native_boundary_diff.py` 全部继续列入 source SHA。

`inactive_boundary_policy` 精确字段：

```json
{
  "path": "inputs.plan_dict.sample_traj",
  "shape": [1, 1800, 5, 3],
  "dtype": "float64",
  "planning_must_be_disabled": true,
  "comparison_only_projection": true,
  "source_function_ast_sha256": "2593f6549742ec99c7c0b9877d63361c57f3126012626f51b7e96db7d494cbba",
  "previous_cache_source_coverage_claim": false,
  "sources": {"<three actual absolute source paths>": "<actual SHA256>"},
  "diagnostic": {
    "directory": "<real difference.json and complete.json directory>",
    "difference_sha256": "aecf29305e3d4b086e834df4b391580e0964a330a22f4d91c09566b4036b95a5",
    "complete_sha256": "<actual completion SHA256>"
  }
}
```

以上是字段说明，不是可部署冻结协议。`sources` 实际含 template、trajectory sampler、GridMask 三个独立路径，取自 `receipts/inactive_planning_additional_sources_v1.json` 的服务器只读 SHA 核验。旧 13 项 runtime map 原值不改，不声称新补记的源在旧 cache 提取时已被锁定。目录允许相对协议，以便真实结果本地镜像复核。

## 输出字段

- common/manifest schema：`objective-joint-evaluation-contract-v2`，新增完整 `inactive_boundary_policy`。
- sample schema：`objective-joint-sample-v2`；原顶层 `input_tree_sha256` 仍是实际完整 fresh 输入的哈希。
- pilot `engineering_parity.full_boundary_hash_equal` 现在是实际布尔值，可能为 false；`active_boundary_hash_equal` 必须 true。
- pilot `engineering_parity.inactive_random_sample_traj_audit` 含完整双方哈希和叶差异，不删去无法逐位复现的事实。字段名以 `compare_cache_boundary` 实际输出为准。
- 原 `exact_gt_equal`、`native_all_5h_3layer_logits_bitwise`、`all_models_5_horizons_full_GT_confusion_equal` 必须 true。
- shard completion status 保持原值，新增 schema `objective-joint-shard-complete-v2` 和 `planning_disabled_all_models`：每模型 `{turn_on_plan:false, all_modules_eval:true}`。Full 同样记录这个实际运行门。
- merged pilot schema：`objective-joint-pilot-merge-v2`；complete schema：`objective-joint-merge-complete-v2`。Pilot summary 顶层保留相同 policy。
- full authorization schema：`objective-joint-full-authorization-v2`。实际资源与 CLI 完全相等、且不超过评价协议天花板，仍须绑定新的真实 pilot PASS。

## 源码与调用

`objective_joint_evaluation_v2.py` 继续从旧 native run 派生九处计数 AST 修改：原七处合同/架构/回执/资源守门，加历史 cache 比较和真实 parity 字段两处。逆变换须精确还原原 native run；全模型 loader、capture、前向、GT 和 evaluator 表达式不变，不安装 loss adapter。

CLI 与 v1 相同，仅入口和协议为 v2：

```bash
python objective_joint_evaluation_v2.py \
  --evaluation-protocol EVAL_V2 --objective-protocol OBJECTIVE --protocol PARENT \
  --config CONFIG --checkpoint M0 --repo REPO --out NEW_ROOT \
  --mode pilot --shard-index 0 --parity-cache DEV200 \
  --device cuda:0 --max-seconds 600 --max-allocated-gib 64
```

另一 shard 使用 1。Full 去掉 parity-cache，加真实 v2 full-authorization。这里不创建任务、不 dispatch、没有自动重试。

本地完成 Python 3.10 语法、CLI、九处 AST 逆变换、独立 namespace、真实诊断回执读取及 consumer 条件支配静态检查；独立审查用 NumPy 叶反例确认无用叶差异可记录，而其他输入、dtype、shape、NaN、规划启用等全部拒绝。未将无 Torch 的 CPU 测试描述为真实 Tensor/CUDA 验证。真实 v2 两分片 pilot 尚未运行。
