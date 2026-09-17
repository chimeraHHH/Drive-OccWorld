# Readout transition v2 真实结果绘图入口

`plot_readout_transition_results_v2.py` SHA：`04ecd07accba5fc35577279e36cd861940ec488da2a5f6bd150d29ca033ce5b7`。精确绑定根任务已冻结的 v2 协议 `deba0647052014de50a00e373582806e6dfa814c16f9c4eaa524d93b66bf77cf`，producer `489fccce38cb418b02d6de18164b70c192b8c76d8e9654c4d8293ce9e7bf0b06`，wrapper `e3ec99f45c7e91355cf4d2eef7ace6a92ef94036d0915f61bf0fc23b3f6a7756`。

相对保留的 v1 绘图脚本，只迁移精确协议 SHA、producer/wrapper 文件名、默认协议文件名，以及绘图收据的 v2 schema 和工程 revision 来源。诊断产物结构未变化，仍使用其真实 schema v1。没有改变 campaign、stage、源文件、200 anchors / 100 scenes、完整 GT 计数、原 C/O 对角混淆、逐位工程门或 10,000 次 seed11 场景配对 bootstrap 检查；没有改曲线、forest、数字计算或 caption 内容。

## 当前验收边界

已完成 Python 3.10 语法、CLI、全部 22 项冻结源 SHA、精确逆变换还原旧脚本检查。真实已完成的旧 O 训练结果，以及真实失败的 readout v1 campaign，误作输入时均被拒绝且没有写出图或目录。记录见 `receipts/readout_transition_plot_v2_static_check.json`（SHA `25924931378071050c41b5caebc603484e0b27d296dd55021c4a81e7523de400`）。

本次没有发现有证据支持的旧字段不兼容需要额外修正。尚未对完整 v2 数据做正向验收，没有渲染，没有视觉 QA；不构造 cross 数值预览，也不以两个工程 anchors 代替开发集结果。

## 完整结果到齐后使用

须先确认真实 v2 job 已成功退出。镜像 v2 campaign 根目录的 `complete.json`、`manifest.json`、`dependency_complete.json`、`development_authorization.json`，以及 `pilot/` 和 `development/` 各自的 `complete.json`、`manifest.json`、`records.jsonl`、`summary.json`、`report.md`。保留任何失败回执，不隐藏它们以绕过拒绝门。旧 C/O 小型回执、开发 JSONL 与五模型汇总使用既有真实镜像；本图不重新读取 checkpoint、原始 GT 或 logits。

先只读验证与统计复算，完整输入不足时应失败：

```bash
readout_project=/Users/yiminghua/2026Summer/WorldModel/dropple
python3 -B "$readout_project/analysis/m0_improvement_20260915/plot_readout_transition_results_v2.py" \
  --campaign "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_readout_transition_v2" \
  --development-summary "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/summary_v1/summary.json" \
  --native-runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_memory_v1/runs" \
  --runs-root "$readout_project/analysis/m0_improvement_20260915/server_results/campaign_objective_v1/runs" \
  --out "$readout_project/analysis/m0_improvement_20260915/figures/readout_transition_development_v2" \
  --validate-only
```

通过后去掉 `--validate-only` 才渲染。输出目录必须全新。输出为 PNG/PDF/SVG、`horizon_numbers.csv`、`effect_numbers.csv`、`caption.txt`、`render_receipt.json`。收据状态为 `RENDERED_PENDING_VISUAL_QA`，并显式记录 v2 revision、前版绘图源码 SHA、新协议 SHA 与每个真实输入 SHA。

图左显示四组合全部五时域（含 t0），图右显示两条条件路径的四个效应、总差和指标尺度上的交互。点估计的两条路径均相加得到总差，区间端点不能相加。保留单训练 seed11、已曝光 development、事后诊断、非唯一因果分解、GMO 非实测运动的说明，不冒充 full validation。

真正生成后再打开 PNG 核对排版、全部标签、正负区间与 CSV；通过后另存真实视觉 QA 收据。在此之前不能声称 GPU/数据验收或图表完成。
