# 完整验证图的已审阅交付版

当前使用 `plot_objective_full_results_reviewed.py` 与 `figures/objective_full_v2_reviewed/`。该版读取真实 full5119/150 结果，已完成所有整数 confusion 指标和 10,000 次场景配对复算；另一个审计使用独立数学实现复核。

这次修订纠正旧绘图收据的 `current_frame_paired_interval_computed=False`：底层完整汇总事实上计算了全部五个时域（包括 t0）的配对区间。图中 t0 页脚仍仅展示点差；新增 `horizon_contrast_numbers.csv` 完整导出 4 组比较 × 5 时域的差值、配对区间及 FP/FN 差。原始实验、协议、分数和所有曲线均未改变。

审阅版 PNG 与先前实际查看的 PNG 字节一致，视觉检查通过；所有 20 行新增 CSV 与已经复算的完整汇总逐项一致。旧源码及产物保留用于审计，交付以此审阅版为准。

运行方式（输出目录必须尚不存在）：

```bash
python3 analysis/m0_improvement_20260915/plot_objective_full_results_reviewed.py --summary analysis/m0_improvement_20260915/server_results/campaign_objective_joint_full_v2/summary_v1/summary.json --full-authorization analysis/m0_improvement_20260915/objective_joint_full_authorization_v2.json --out NEW_OUTPUT_DIRECTORY
```

科学范围：一个训练种子；历史暴露的完整验证集；未校正场景配对区间；不代表新盲测、跨种子稳健性或 ICLR 发表要求全部满足。
