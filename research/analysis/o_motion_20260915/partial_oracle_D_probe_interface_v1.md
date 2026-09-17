# 固定 train16 的 D / partial-oracle 诊断入口

脚本 `partial_oracle_D_probe_v1.py`；固定原训练缓存 `records[:16]`，可用 `--anchors 2` 做原序首两锚工程检查。只读取已完成 O/D 权重，不创建 optimizer、不训练、不作开发集或全量评价。

```sh
python partial_oracle_D_probe_v1.py \
  --protocol connected_motion_protocol_v2.json \
  --training-run ACTUAL_CONNECTED_MOTION_TRAIN_V2 \
  --training-complete-sha256 f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56 \
  --config ORIGINAL_S0.py --checkpoint ORIGINAL_M0.pth \
  --o-checkpoint ORIGINAL_O_latest.pth --repo ORIGINAL_NATIVE_REPOSITORY \
  --runtime-contract runtime_source_contract.json \
  --train-cache ORIGINAL_NATIVE_train512_CACHE \
  --raw-labels ORIGINAL_motion_targets_v1_712 \
  --sparse-labels ORIGINAL_sparse_motion_v1_712 \
  --o-reference oracle_reference_records.jsonl \
  --anchors 16 --out NEW_DIRECTORY --device cuda:0 \
  --max-seconds EXPLICIT_BUDGET --max-allocated-gib 32
```

原始 O/D 前向使用 `input_only`，先于当前样本 fine GT、raw boxes、sparse NPZ 内容读取。第三臂 `partial_oracle_D` 是明确使用标签的诊断：只将 `valid[h]` 对应的原 XYZ C-order 源 flat indices 替换成 NPZ float32 刚体位移，其余 D 位移保持逐位相同。概率载体、O current argmax 源 mask、D gate 参数不变；gate 激活值依原函数对新的 S/W 重算。未标注点保持 D，不改成静止、不增加 GT source mask。

输出仍是原完整 fine GT 指标与原八类速度/标注状态归因。额外报告：

- 原 NPZ 有效唯一源点/对象中，O coarse source mask 有/无支持的数目，按原三类 physical speed group 分层；不称完整对象可见性。
- O→D、O→partial、D→partial 的 FN 域内/域外及 TP→FN/FN→TP。前两者使用对应 W>0；第三者使用两 W 支持并集。按原 `[N,1,X,Y,Z]` trilinear、`align_corners=False` 放大到 `(512,512,40)` 后 >0，域外预测必须相同。
- coarse 源/保留概率质量、支持/碰撞计数、门值及实际混合 margin 改变量分布。没有 gate=1 条件、最大拼接或伪造可达上界。

这只是当前固定系统对有效标注源位移替换的响应。影响域是保守的插值依赖 support bound；不等于物理可达性、正确关联或可实现的 IoU 上界。若无改善，不能单独否定物理 motion；若改善，也不能把 privileged 分支当新候选。

包内 16 份 Python：本脚本、common_connected_motion_evaluation_v2、common_change_evaluation_v2、common_occupancy_change_metrics_v1、motion_geometry、train_connected_motion_v2、train_supported_fusion_v2、supported_motion_fusion、transport_ops、train_source_motion_v1、motion_prediction_head、learned_transport_probe_v1、native_state_cache、oracle_transport_probe、objective_supervision_adapters、joint_native_evaluation。静态非 Python 资产仅 training protocol v2、runtime_source_contract、oracle_reference_records。权重/完整训练回执/缓存/原712标签均使用现有服务器路径。D strict loader 会核两臂训练来源与固定最终 checkpoint，但不会运行 J 或加载 GT 作为 D 输入。

输出 manifest.json、loaded_models.json、records.jsonl、summary.json、report.md，complete.json 绑定全部五件；失败保留 failed.json，无自动重试。只保存小统计/哈希，不保存 fine masks、logits 或 dense flow。

本机已跑 4 项必要 NumPy 测试：非方 XYZ 布局、valid/空/全无效、未修改字节、源支持缺失及错误数据拒绝。Python3.10 AST、CLI 和 16 份实际源码 SHA/延迟导入通过。尚未执行本脚本 GPU 前向；本机测试不代表真实探针已完成。
