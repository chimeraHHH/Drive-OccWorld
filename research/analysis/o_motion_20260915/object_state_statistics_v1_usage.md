# V/G 完整终点统计器（无新成绩）

两入口只接受 `object-state-forecast-training-v1` 的完整512更新/2048 examples 与全部200个开发样本。先核完整 common/两臂 complete、小文件 SHA、冻结训练与分析源、原四轮 seed11 排列、配对样本/RNG/学习率及实际 optimizer step，再读取原始计数和对象记录。原 K/B 目录不能作为 V/G 输入。`.pth` 大文件不重新加载或扫描；其 SHA 与最终 head/readout/conditioner state 声明在 top/arm complete 和 summary 之间核对，实际张量来源由训练生产者负责。

两脚本共同参数：

```text
--training-run 实际完整V_G训练目录
--protocol 实际冻结正式protocol.json
--selection 原selection_v1.json
--o-reference 原campaign_objective_v1/runs/O/development_records.jsonl
--sparse-manifest 原sparse_motion_v1/manifest.json
--out 全新输出JSON路径
```

`--raw-manifest` 默认使用同目录 `motion_targets_v1/manifest.json`，可显式指定同 SHA 镜像。物理入口另需：

```text
--D-training-run 原connected_motion_train_v2完整目录
--crn-cv-root 原crn_object_state_cv_dev200_v2目录
```

`summarize_object_state_common_v1.py` 比较 V−O、G−O、V−G。主量保持每个未来时域先池化完整 confusion、计算 GMO IoU，再对四个未来时域取均值。原 t0、全部逐时域、全局 FP/FN、八个运动归属组和四状态 transition 全部保留。V−G 才是相同容量/对象几何下的速度输入增量；V/G−O 同时包含继续训练、新容量与额外检测状态，不能单因素归因。每个模型 t0 的自身 hash/confusion 须一致，共享的是 GT/mask；transition 的两个预测端都可变，不当作独立物理运动误差。

`summarize_object_state_physical_v1.py` 比较 D/V/G/固定 CRN-CV v2/零位移。逐 key 核当前 D 与旧完整 D、CRN-CV 内嵌 D 的16,074个对象-时域结果完全相同；V/G/CRN-CV 具有相同实例、时域、真实 dt、组、源点数和零位移误差，再与原 sparse manifest 的全部有效/缺失支持计数核对。未检出的对象不删；CRN-CV 未覆盖点仍用零速度，原覆盖统计随结果披露。无唯一当前虚拟源点的对象、缺未来标签等既有缺口明确列出，未来出 ROI 的有效标签仍保留。

物理量为每对象内源点 XY/XYZ EPE 均值，再对 anchor-instance 等权；每个时域、all/静止/模糊/运动组都报 mean、median、p90。均值差使用100个完整场景成对重采样10,000次，seed11仅是 bootstrap RNG，不是额外训练种子。median/p90 是描述性点估计，不冒称已有分位数置信区间。占据统计采用同一 scene bootstrap 及原计数比率公式。所有区间未经多重比较修正，不能估计训练种子不确定性；空分母保持 null。

真实旧数据退化检查回执在 `object_state_statistics_v1_cpu_checks.json`：原 K/B 全部整数计数、指标和10k区间逐值一致，原 CRN-CV v2/D 的全部支持和 XY/XYZ 均值/中位数/p90/10k区间逐值一致。未生成 V/G 分数、未运行模型或训练。正式协议只需随后绑定两份最终统计器 SHA；此交付不填写实际预检 SHA 或训练资源。
