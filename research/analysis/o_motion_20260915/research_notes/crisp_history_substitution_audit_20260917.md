# CRISP history substitution audit

日期：2026-09-17
对象：CRISP（arXiv:2607.04541，v1）
范围：只核对论文正文、项目页和公开官方仓库状态；不训练、不运行 CRISP 代码。

## 结论先行

CRISP 是“历史多视角相机 + 雷达，使用未来 LiDAR 作特权预训练监督，再把预测状态用于下游”的近邻工作。但论文的 `0s/1s/3s` 是历史时长标签，不是一个可复核的原始输入协议。正文只明确写出雷达输入使用 **6 个 radar sweeps**，没有给出三个 history 设置各自的相机帧数，也没有说明 0s 时这 6 个 sweeps 是否仍跨越一段时间。因此，不能据此认证“0s 无相机/雷达隐含历史”，也不能把 0s 解释成“一张当前图像 + 一个当前 radar sweep”。

论文没有报告同一 backbone 下的 camera-only 与 camera+radar 的 history 交叉实验，没有找到 Doppler 移除对照，也没有端到端/streaming 的 FPS、延迟或吞吐证据。官方项目页链接的仓库在本次核对时仍为 “Code Coming Soon”，GitHub 地址返回 404，所以 loader、队列清空、sweep 时间戳、Doppler 通道等实现级问题均为 unknown。

## 逐项核对

| 问题 | 论文实际支持的事实 | 状态与边界 | 位置 |
|---|---|---|---|
| `0s/1s/3s` 的含义 | Table I 按 0s、1s、3s 报告 history horizon；CRISP 三行均为 C+R | **已确认是历史时长标签；未确认原始帧协议** | 论文 §V-B/Table I；HTML [Table I 附近](https://arxiv.org/html/2607.04541v1#S5)（HTML lines 289–302） |
| 雷达输入数量 | “For radar input, we use 6 radar sweeps.” | **已确认总设置写了 6 sweeps；未确认其跨越时长及在 0s 下是否保留** | §V-A2（HTML lines 275–282） |
| 相机帧数量 | 正文描述 historical multi-view images，但没有列出 0s、1s、3s 各取几帧 | **unknown** | 摘要、§V-A1–A2；未见 camera frame count 表或公式 |
| 0s 是否没有隐含历史 | 论文未给出每路数据的时间戳、队列长度、reset 规则或 0s loader 实现 | **不能认证；尤其 radar 6-sweep 聚合可能仍含时间历史** | §V-A2；官方代码当前不可得 |
| 未来输入是否泄漏 | 未来状态解码器使用先前预测 BEV 状态和 future ego-motion embedding；预训练使用记录的未来 ego trajectory；明确“不包括 future sensor observations 或 future object states” | **论文层面支持无未来传感器/物体观测；不等于 0s 的历史输入已被审计** | §IV-D1（HTML lines 249–258） |
| 下游是否仍使用未来 LiDAR | LiDAR 只作 privileged pretraining supervision；下游保留 camera branch、radar encoder 和 CR spatiotemporal BEV backbone | **已确认训练/部署分工** | §IV-E（HTML lines 266–268）、§V-A1 |
| camera-only vs camera+radar 同 backbone、跨 history | Table I 中 CRISP 是 C+R；ViDAR 的 C、HERMES 的 C、LRS4Fusion 的 C+L 是其他方法/骨干或不同模态组合 | **未做或至少未报告可认证的同 backbone 交叉矩阵** | Table I（HTML lines 289–302）；§V-B1（lines 317–321） |
| 纯 camera-only CRISP 行 | 主表没有 CRISP C-only 行；也没有同一 CRISP 权重/训练配方下的 C-only 0/1/3s 对照 | **unknown / 不应从他法行推断** | Table I、§V-B |
| Doppler removal | 正文讨论 radar Doppler/ego-motion-aware 设计，但 Table VIII–XI 只列 radar-enhanced TSA、radar encoder、temporal gating、innovation gating 等累计组件 | **未找到 Doppler 通道移除或 Doppler zeroing 对照** | §IV-B（约 HTML lines 166–179）、§V-D2–D5（lines 402–448） |
| 组件消融能否代替 modality/history 对照 | Table VIII 的消融为累计结构改动；并使用 lightweight BEVFormer-small、160×160、subset 训练等设置 | **不能当作同 backbone、同计算量的 C-only/C+R history 消融** | §V-D1（HTML lines 396–401）、Tables VIII–XI |
| 端到端/streaming 计算 | 给出预训练 24 epochs、8×A100、每 GPU 1 sample 等训练资源信息 | **没有 FPS、单帧延迟、端到端吞吐、显存随 history、streaming cache/update 或实时闭环证据** | §V-A2（HTML lines 275–282） |
| planning 结果是否是运行时证据 | 下游 planning 只报告任务指标和 3s horizon | **是任务指标，不是 wall-clock/streaming 测量** | §V-A3、Table VII（HTML lines 283–288 及后续） |
| 官方实现是否可核查 | 项目页的 Project Repo 指向 `umfieldrobotics/CRISP`，文字为 “Code Coming Soon”；该 GitHub URL 本次访问返回 404 | **loader、帧索引、sweep 构造、Doppler 字段、状态 reset 全部 implementation unknown** | [项目页](https://umfieldrobotics.github.io/CRISP)；[官方仓库](https://github.com/umfieldrobotics/CRISP) |

## history 输入语义的可用表述

| 论文标签 | 可安全写出的内容 | 不能写出的内容 |
|---|---|---|
| 0s | 评测配置标记为 0 秒历史 horizon；模型不使用未来传感器观测 | “只有当前一帧相机和当前一帧雷达”“完全无 latent/queue history” |
| 1s | 评测配置标记为 1 秒历史 horizon | “准确 N 帧相机 + N 个 sweeps”，除非另有代码或作者答复 |
| 3s | 评测配置标记为 3 秒历史 horizon | “准确 N 帧相机 + N 个 sweeps”，以及把三者的雷达窗口默认为相同/不同 |

“6 radar sweeps”与“0s history”并不矛盾：0s 可能只约束历史 keyframe 的标称时长，而雷达仍采用内部多 sweep 聚合；公开正文没有说明各 history 设置对应的具体 sweep 构造，故应标为 unknown，而不是替 CRISP 补全输入协议。

## 对当前 history 替代假设的直接影响

CRISP 可以作为“多模态历史状态预测、未来 LiDAR 特权监督、下游移除预测头”的概念近邻，但只能支撑方法层面的邻近性，不能支撑输入因果边界或 history 公平性的结论。若将其与当前实验比较，最低限度应在自己的协议中固化并记录：相机帧数和每帧时间戳、所用 radar sweeps 的数量和时间戳/聚合窗口、0s 是否清空跨时缓存、Doppler 字段是否存在或被置零、同一 backbone/初始化/训练预算，以及是否使用 future ego-motion 仅作坐标变换。否则，“history=0”与“history=1/3s”的差异会同时混入传感器数量、雷达聚合和队列状态差异。

CRISP 的 §VI 还明确承认其数据驱动目标没有显式建模 causality/intent（HTML lines 462–467）。这使它适合作为预测状态表示的近邻参考，但不能被当作已经解决“无隐含历史”或“物理因果状态”的证据。

## 原始来源

- arXiv 摘要与版本信息：[https://arxiv.org/abs/2607.04541](https://arxiv.org/abs/2607.04541)
- 可读全文（v1）：[https://arxiv.org/html/2607.04541v1](https://arxiv.org/html/2607.04541v1)
- 作者项目页：[https://umfieldrobotics.github.io/CRISP](https://umfieldrobotics.github.io/CRISP)
- 项目页链接的官方仓库（本次核对返回 404）：[https://github.com/umfieldrobotics/CRISP](https://github.com/umfieldrobotics/CRISP)

