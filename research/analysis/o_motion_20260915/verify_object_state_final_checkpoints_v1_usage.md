# V/G fixed512 最终权重 CPU 核验

仅在正式训练及 final dev200 全部完成后运行。输入终态 `complete.json` 的实际 SHA，不接受中间 checkpoint。脚本与新审计目录可独立部署，`--source-root` 指向原冻结训练 package；不改其中任何文件。

```sh
CUDA_VISIBLE_DEVICES='' python -B verify_object_state_final_checkpoints_v1.py \
  --source-root "$FROZEN_PACKAGE" \
  --training-run "$COMPLETED_VG_RUN" \
  --complete-sha256 "$ACTUAL_FINAL_COMPLETE_SHA256" \
  --protocol "$FROZEN_PACKAGE/object_state_forecast_training_protocol_v1.json" \
  --preflight-protocol "$FROZEN_PACKAGE/object_state_forecast_preflight_protocol_v2.json" \
  --engineering-preflight "$COMPLETED_V2_PREFLIGHT" \
  --o-checkpoint "$ORIGINAL_O_RUN/latest.pth" \
  --o-reference "$ORIGINAL_O_RUN/development_records.jsonl" \
  --connected-training-run "$COMPLETED_JD_RUN" \
  --selection "$FIXED_SELECTION_JSON" \
  --sparse-manifest "$SPARSE_LABEL_ROOT/manifest.json" \
  --raw-manifest "$RAW_LABEL_ROOT/manifest.json" \
  --out "$NEW_CPU_AUDIT_DIRECTORY"
```

使用原训练环境的 CPU PyTorch/NumPy。原冻结 package 需含协议的 21 个 runtime 与 2 个 analysis 源；无原生模型构造、forward、optimizer 恢复或 CUDA 初始化。新目录已存在即拒绝。输出 `audit.json`、`report.md`、`complete.json`，失败写 `failed.json` 并非零退出。

核验原 O checkpoint 文件及 head 张量作为初始/结构参考；按原 seed11 的 readout→conditioner 顺序重建初态。逐一检查 V/G 三组件张量、参数名/数量、真实 AdamW moments 和逐参数 step。实际 step 对照全部 512 日志中该参数 `grad != None` 的次数；零梯度仍计更新，不把 conditioner 的 None-gradient 跳步误判为漏训。最终 optimizer LR 与认证后的最终日志完全相等，不在不同平台重复 libm cosine 公式。

原 O metadata 的参数名来自 `model.named_parameters()`，带 `future_pred_head.`；新 V/G manifest 与原 O 保存的 head state 使用模块内参数名。核验器先确认旧名称全唯一且包含恰好一次前导 `future_pred_head.`，仅去掉此前缀，再要求与新 manifest 及实际原 O state keys 的顺序完全相同。张量与摘要算法不变。

冻结 D 与 O 非 future-head 未写入本轮 final.pth。这里只认证它们的来源及原运行时摘要，不能从本轮 final 权重独立证明它们未变；最终权重也不能单独证明全过程 RNG/reset 行为。脚本不重新统计模型成绩。

交付时仅通过 Python 3.10 AST、CLI/import（未导入 Torch）、真实冻结协议/23 源字节检查和既有真实 preflight 小件认证。正式 V/G 尚在运行，未读取 final.pth 或声称最终张量通过。

后续根代理在 H200 做原 O + 新模块的 CPU 参考 schema 探测，发现原先直接比较两种参数名空间的断言失败；这是核验器接口错误，并非训练失败。此次窄修只加入上述前缀认证/转换；真实最终权重核验仍待终点完成后运行。失败说明见 `receipts/object_state_final_verifier_prefix_repair_v1.json`。
