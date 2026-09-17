# 同一最终 V 的 N/P/Z 速度分配诊断草案

**只准备入口，未派发、未运行；不是晋级门。** 仅接受冻结 V/G 协议 `93bd80fa…82ccc9`、真实 fixed512 + dev200 complete，以及由最终 CPU verifier `a17905b6…77098` 产生的实际 `PASS_ACTUAL_CPU_FINAL_TENSORS`。默认 `check` 只读来源、小件终点和既有 CPU 审计，不加载 Torch 或预测模型。

仅两个新文件，本入口在 `load_final()` 内复用冻结 verifier 的 CPU 文件/逐张量校验、原 D strict loader，严格载入 V 的 head/readout/conditioner；未修改其他 loader。`--source-root` 放原协议的 21 runtime + 2 analysis 源，以及脚本 `EXTRA` 列明的 verifier 和 4 个冻结数学依赖。也支持原 N/P 两个相邻目录布局。独立部署可复制这些**相同字节**到全新 package，不修改正式训练 package。

```sh
python -B diagnose_object_state_velocity_assignment_v1.py \
  --source-root "$SOURCES" \
  --training-run "$COMPLETED_VG_RUN" --complete-sha256 "$ACTUAL_TRAIN_COMPLETE_SHA" \
  --protocol "$FORMAL_PROTOCOL" \
  --cpu-audit "$COMPLETED_FINAL_CPU_AUDIT" --cpu-audit-complete-sha256 "$ACTUAL_CPU_COMPLETE_SHA" \
  --selection "$SELECTION" --o-reference "$ORIGINAL_O_DEV_RECORDS" \
  --sparse-labels "$SPARSE_ROOT" --raw-labels "$RAW_ROOT"
```

以后若根代理决定执行，在同一命令添加 `--mode run`，并显式给出以下参数，使用独立 detached job_runner 控制外层超时：

```sh
  --config "$CONFIG" --checkpoint "$M0_CHECKPOINT" --o-checkpoint "$O_CHECKPOINT" \
  --repo "$NATIVE_REPO" --runtime-contract "$RUNTIME_CONTRACT" \
  --connected-protocol "$CONNECTED_PROTOCOL" --connected-training-run "$COMPLETED_JD_RUN" \
  --dev-cache "$DEV200_CACHE" --development-predictions "$CENTERED_DEV200_CRN" \
  --device cuda:0 --max-seconds "$REVIEWED_INNER_SECONDS" \
  --max-allocated-gib "$REVIEWED_ALLOCATED_GIB" --out "$NEW_DIAGNOSTIC_DIRECTORY"
```

所有路径取正式 launch 的实际输入。run 必须有有限时间上限和不高于 32 GiB 的 allocated ceiling；此入口不决定预算、不派发、不等待任务或自动重试。没有完成的 CPU 审计时，check 和 run 都拒绝。不提供部分样本或预览成绩选项。

固定全部 dev200/100 scenes、每条件 16,074 个原始对象—时域支持。N 原始分配；P 按 `SHA256("velocity-assignment-v1|"+token+"|"+原框index)` 排序循环分配；Z 清零。仅 clone 后改变 packed R 速度 `24:27`，三者均 `use_velocity=True`，由原 conditioner 重算每时域 `p0+h*v`。每条件独立重跑完整 rollout，t0 只与同一 V 自比，不要求等于 O。N 的正式 V hist、完整 common metrics 与对象误差记录必须精确重现，否则失败不改容差。

每条记录保留 donor 原 index、N=0/1、实际速度数值变化和 byte 变化数量、幅度、多重集保持、五时域 logits/terminal features 及四时域物理场的差异摘要。GT/sparse 用于三条件预测完成后的评分；原预测状态读取器仍仅从 raw carrier 提取当前 G0，其标注和未来 pose 不进入前向。输出 `records.jsonl`、`physical_objects.jsonl`、`loaded_models.json`、`manifest.json`、`summary.json`、`report.md` 与完成 ledger，不保存密集张量。

统计复用冻结的 count_vector/aggregate_counts 和 physical_rows/aggregate_physical，10,000 次 seed11 整场景配对；物理 median/p90 是描述统计，均值差有区间。GT-count 认证内部把 N 临时放在旧 validator 的 D 槽，仅为复用原支持核验，不输出伪 D 成绩。P−N/Z−N 是同权重事后诊断；P/Z 均可能 OOD，区间不包含其他置换或训练种子不确定性。V(Z) 不是独立训练 G。没有新阈值、支持过滤或模型晋级判据。

交付静态验证范围：Python 3.10 AST/help、28 份真实冻结依赖字节、纯 NumPy 空/单对象/重复速度及 hash-cycle 机械测试。合成数组只验证置换算子，不是模型结果。真实最终 CPU 审计和 N 原生复现须在以后实际运行时验证。
