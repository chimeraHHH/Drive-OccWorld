# Readout transition v3 绘图入口

脚本 `plot_readout_transition_results_v3.py` SHA：`81f6e44ac88adb50d56862043a5754c2734b1ac036c6b305e61873b2311005e0`。

精确绑定冻结 v3 协议 `e760da6c6eb645b0f2862a58bd37a340d97ed94c51d254384fe084c71386695a`、producer `9921c211e477d2f05dc08848436c66d238cc81a27f13d5124f81bd359ffd675b`、wrapper `ddd8fd7536602ed5fc9e6ee66ce4c99f5bb7b8ebcf8e8009312e1f10bde02b05`。相对 v2 绘图入口仅修改这些版本绑定、默认协议名及绘图收据 revision 来源；数学、完整性验证、原 GT/对角检查和 10,000 次场景配对 bootstrap 均不变。

CPU 语法、CLI、22 项源 SHA 与逐字逆变换检查已通过。真实旧 O 完成结果、真实失败 v2b campaign 误传时都被拒绝，没有写出输出。回执：`receipts/readout_transition_plot_v3_static_check.json`，SHA `1ca232a0e1b6c1c45bb4880409d842d7d61e145da4b1a30304c6ebbe95124bf6`。

**尚未对完整 v3 结果做正向验收，未渲染。** v3 job 成功退出且完整 campaign/pilot/development 回执及原 C/O 来源镜像到齐后，再执行：

```bash
readout_project=/Users/yiminghua/2026Summer/WorldModel/dropple
python3 -B "$readout_project/analysis/m0_improvement_20260915/plot_readout_transition_results_v3.py" \
  --campaign "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v3" \
  --development-summary "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/summary_v1/summary.json" \
  --native-runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_memory_v1/runs" \
  --runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/runs" \
  --out "$readout_project/analysis/m0_improvement_20260915/figures/readout_transition_development_v3" \
  --validate-only
```

通过后去掉 `--validate-only` 渲染至全新目录；生成 PNG/PDF/SVG、全部数字 CSV、caption 和 `RENDERED_PENDING_VISUAL_QA` 收据。之后再进行真实视觉 QA，不提前声明图表完成。禁止 mock cross 值或隐藏失败回执。图中保持单 seed11、历史曝光 development、事后条件比较及非唯一因果归因的限制。
