# 雷达—视觉未来占据预测：阶段工作报告

报告覆盖 2026-09-14 至 2026-09-17 的主要研究进展。PDF 与中文 LaTeX 正文配套，9页，3张真实聚合数据图。

- [阅读 PDF](main.pdf)
- [LaTeX 正文](main.tex)与[最新结果宏](latest_results.tex)
- [数值数据](data.json)与[来源 SHA 清单](evidence_manifest.json)
- [绘图脚本](make_figures.py)，输出矢量 PDF 和 PNG

包含 O 的完整共同验证、运动连接与冻结视觉诊断、已完成的 train512 T/J/D 结果、固定 T 的训练集拟合、文献调研、Doppler 替代视觉历史计划。所有图表数值均为真实记录；少帧实验和缺失证据用红字标记，没有合成数据混入。

服务器状态取自 2026-09-17 12:42:52 UTC 的只读检查；D 的结果随后取回并独立复算。T/J/D 已正常退出，不代表新路线已实现。完整验证和开发场景存在历史曝光，单 seed11。

## 编译

```sh
sh build.sh
```

需 XeLaTeX、ctex、fontspec、常用 LaTeX 表格/排版包，以及 Times New Roman、Arial、Songti SC、Heiti SC、Menlo 字体（当前 macOS 配置）。其他平台可将 CJK 字体改为对应 Noto CJK 字体。

重新绘图：

```sh
python3 make_figures.py
```

Python 依赖：NumPy、Matplotlib。已有图片随包提供，常规编译不要求重新绘图。`evidence_manifest.json` 指向完整工作区的证据路径，原始大数据和权重不随报告发布；`data.json` 足以重绘本报告三图。

报告不修改历史 ICRA 草稿，不恢复 S3，也不将新研究假设写为已验证贡献。
