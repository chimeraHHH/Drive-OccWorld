# 完整原生验证图：使用说明

脚本：`plot_objective_full_results.py`。SHA256：`863cfdcdec138b1dcc9c05dacd8b751262ff0534cdf4959da0067efa5c3e9765`。

本文件保留原绘图版说明。真实 full5119/150 已完成；当前交付使用 `plot_objective_full_results_reviewed.py` 和 `plot_objective_full_results_reviewed_usage.md` 所述审阅版。

在本目录运行（路径替换为真实镜像位置）：

```bash
python3 plot_objective_full_results.py \
  --summary REAL_FULL_SUMMARY_DIR/summary.json \
  --full-authorization REAL_FULL_AUTHORIZATION.json \
  --out figures/objective_full_v2
```

默认读取同目录的 `objective_joint_evaluation_protocol_v2.json`、`objective_supervision_protocol_v1.json`、`protocol_v2.json` 和 `full_validation_selection_v1.json`。其他同 schema 冻结版本须显式提供 `--evaluation-protocol`；其余路径也可通过 `--objective-protocol`、`--protocol`、`--selection` 覆盖。

只接受 `objective-joint-full-summary-v2`、`objective-joint-merge-complete-v2` 及 `objective-joint-full-authorization-v2`。Merger 固定 SHA 为 `0394ff12140891f70f38527f290b7f71e3c6b52bfe76bde80002620c20ac003b`；当前协议 SHA 为 `b1b1b046b5acf9dc1207a19e0b7b740d6db38a0982a3794f670fa06862189f28`。

输入目录必须包含完整 merger 绑定的七文件：`summary.json`、`report.md`、`sample_hash_ledger.json`、`development_disclosure.json`、`raw_confusions.npz`、`metrics.csv`、`contrasts.csv`，以及 `complete.json`。脚本核对其哈希、协议/授权/最终权重/原生来源绑定，拒绝 pilot、dev200、缺分片、缺文件或候选集合改变。模型由 `candidate_names` 决定，支持 M0/M0_fp32/native1 加 F、O 或 F+O 两个独立候选；这里的两个候选不是联合干预模型。

脚本重读全部 5,119 anchors、150 scenes 的整数 GT 计数与 confusion；用冻结的纯数学函数重新计算四/五模型全部指标及 10,000 次 scene bootstrap，保持每场景自然数量的全部 anchors。所有点值、区间和重采样收据须与完整汇总完全一致。没有加载神经模型、重新推理、修改 GT 或拟合阈值。另核对 v2 九次 AST 受控变换、真实失败诊断和补充来源收据、逐模型规划关闭状态，以及完整评价每条记录 `engineering_parity=None`、未使用历史缓存投影。Pilot 只比较时排除无效 `sample_traj` 的历史例外不得移入 full 输入。

验证全部通过后，才创建新输出目录并生成：

- `objective_full_results.pdf/.png/.svg`：四未来时域曲线、所有已冻结候选相对 M0_fp32/C 的主比较区间；保留零线与 +0.5 pp 参考线。页脚另列候选相对 M0_fp32 的真实 t0 点差，避免掩盖当前帧代价。
- `horizon_numbers.csv`、`contrast_numbers.csv`：原始精度数值，包含未绘制的 t0 和控制比较，区分 GMO、binary mIoU、比例和 pp。t0 单模型区间来自完整汇总；原图的 t0 相对差只展示点估计；底层完整汇总已经计算配对 t0 区间，审阅版另将全部五个时域的配对区间导出为 `horizon_contrast_numbers.csv`。
- `caption.txt`：完整验证、历史暴露、单训练 seed、未校正区间和开发集晋级边界，并明确未来均值改善不等于全部时域改善。
- `render_receipt.json`：输入/输出/源码 SHA、实际绘制数值，状态 `RENDERED_PENDING_VISUAL_QA`。

真实渲染后须查看 PNG，再另写 `visual_qa.json`；本脚本不预先声称视觉 QA 通过。不会覆盖已有目录，不提供 mock、缺失值填补、平滑或未晋级模型的零分。

已完成的本地检查：AST/编译/import/CLI；四/五模型排序及空集/重复/逆序拒绝；缺输入和误传真实 dev200 时零输出拒绝；用既有真实五模型开发数据核对冻结纯指标 helper。本次 v2 更新另通过真实协议全部源码 SHA、九次 AST 变换与来源不变核验；缺输入和真实 dev200 均被拒绝且无输出。完整输入链和实际图形仍待真实 full 结果验收。
