# Readout transition v4 绘图入口

脚本 `plot_readout_transition_results_v4.py` SHA：`086164754182b3cf85d3c592e132bc53e0c25103997cf011935606b6ad1e8acd`。

精确绑定冻结 v4 协议 `4baad13ab4d6fe0ddce740b6a345923668f3bba2e4a8b10e21bb84f7694b0d83`、producer `e7f0f7b44029366cbe842ad3105cac1d0b408ff11fa5c511a39a8c32b5d7b351`、wrapper `b60758605b1f7db3c2a50c4e4f0c67af28b6a33a81bb3150e378325d8f56be70`。

相对保留的 v3 入口，只更新版本/来源绑定、默认协议路径、绘图收据 revision，以及 development 实测预算上限 1500 → 3000 秒。总预算 3600 秒通过冻结协议与 wrapper SHA、campaign manifest.resources 的精确相等绑定；pilot 300 秒 / 32 GiB 和 development 32 GiB 不变。开发阶段的实际时限仍须精确匹配同一次 pilot 的公式 `max(300, ceil(1.75*(initialization_seconds + 200*max_sample_seconds) + 60))`，且不超过 3000 秒；不将资源上限解释为实际耗时。

本地已通过 Python 3.10 语法、compile、CLI `--help`、22 个协议绑定源码 SHA 检查。7 组受控替换逆向恢复后与原 v3 脚本逐字一致，确认数学、完整性、原 GT/对角检查、10,000 次场景配对 bootstrap 和图表内容未改。未启动任何模型计算，未生成图表或模拟结果。

**尚未对完整 v4 结果做正向验收。** 待 v4 campaign、pilot、development 以及原 C/O 来源的完整真实文件镜像到齐后，执行以下本地只验收命令：

```bash
readout_project=/Users/yiminghua/2026Summer/WorldModel/dropple
python3 -B "$readout_project/analysis/m0_improvement_20260915/plot_readout_transition_results_v4.py" \
  --campaign "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v4" \
  --development-summary "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/summary_v1/summary.json" \
  --native-runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_memory_v1/runs" \
  --runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/runs" \
  --out "$readout_project/analysis/m0_improvement_20260915/figures/readout_transition_development_v4" \
  --validate-only
```

通过后去掉 `--validate-only`，渲染至同一尚不存在的新目录；生成 PNG/PDF/SVG、全部时域与效应的数字 CSV、caption 和 `RENDERED_PENDING_VISUAL_QA` 收据。之后再进行真实视觉 QA，不提前声明图表完成。不接受 mock cross 值、部分样本、旧版本结果或失败回执。图中保留 t0、单 seed11、历史曝光 development、事后条件比较和非唯一因果归因的限制；四个未来时域构成 forest 的指标，t0 不计入该均值。
