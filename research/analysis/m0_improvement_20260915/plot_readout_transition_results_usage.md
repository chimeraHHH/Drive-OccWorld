# 真实四组合诊断图

脚本：`plot_readout_transition_results.py`，SHA `9caadb15395832f454df338bf3d712846a659d0b6b9a42aa378d8eebf063e19b`。诊断协议固定为 `b2521df119bc7877729154b45fbc19dd31474afe620a0f864df31a3e68e892b9`。

目前只完成代码准备、Python 3.10/CLI/原来源与 effect 系数核验、真实旧 O 结果误传的拒绝检查。Root 已读实际输入验证、数字导出与渲染代码，对照原五模型 summary 的实际 audit/identity/bootstrap 字段；未发现接口阻断。**尚无完成的四组合结果，不存在已验证的成功渲染或视觉 QA。**

完整诊断结束后，先核服务器 job 终态及 campaign/stage 结果。所需文件见 `goal_state.json` 的 `completed_diagnostic_fetch_files`；原 C/O 小型来源回执和开发 JSONL 已在本地镜像。不要为运行此图下载大 checkpoint 或持久化 logits。

在项目根目录执行：

```bash
python3 analysis/m0_improvement_20260915/plot_readout_transition_results.py \
  --campaign analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v1 \
  --development-summary analysis/m0_improvement_20260915/server_results/campaign_objective_v1/summary_v1/summary.json \
  --native-runs-root analysis/m0_improvement_20260915/server_results/campaign_memory_v1/runs \
  --runs-root analysis/m0_improvement_20260915/server_results/campaign_objective_v1/runs \
  --out analysis/m0_improvement_20260915/figures/readout_transition_development_v1
```

加 `--validate-only` 可只进行真实完整输入和统计复算，不写任何输出。输入缺失、campaign/pilot/200-anchor 收据不符、对角历史混淆不一致或统计复算不符时拒绝，不能用 placeholder 绕过。

输出包含 PNG/PDF/SVG、五时域原始 confusion 与指标 CSV、全部条件效应 CSV、caption 和 `render_receipt.json`。左图显示包含 t0 的全部四组合五时域；右图显示两条路径的四个条件效应、总差与交互。显式保留 development、历史曝光、单 seed 和非唯一因果分解的说明。两条路径点估计之和等于总差，区间端点不能相加。

真正生成后必须打开 PNG 检查排版、全部标签和正负区间，必要时只修改绘图层再渲染新目录；通过后另存真实 `visual_qa.json`。不得提前将 `RENDERED_PENDING_VISUAL_QA` 改写成最终验收通过，也不以此开发诊断图代替 full 原任务比较。
