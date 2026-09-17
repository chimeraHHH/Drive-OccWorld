# M0 时间建模调研与可移植机制

2026-09-15。仅公共一手文献检索、论文正文和本地作者代码只读检查；未调用 skill、未连接服务器、未训练。结构化来源、代码版本与文件 SHA 见 [sources.json](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/sources.json)。以下“可能改善”均是实验假设，不是结果。

## 当前优先顺序

**首选：持续观测锚＋最近预测的两槽 transition，与最近两帧预测的两槽 transition 作匹配对照。** M0 原一槽在第一次预测后丢弃直接 t0 观测入口；后续只能从生成状态恢复细节。保留 t0 给整个可训练 future head 一条持续读取观测的通路；对照也有两槽，可区分记忆内容与增加容量。第一步均为 `[t0,t0]`，之后 persistent 为 `[t0,F̂(h−1)]`，rolling 为最近两状态。每槽保留自己的参考系变换，不把陈旧观测位置假装成未来物体位置。该实验超出 R2–R4 的冻结输出小头，但不能预先保证提升。

**第二候选：压缩通道后的 history/future 联合 refiner。** 参考 OccProphet 的实测历史与预测未来共同处理，在 M0 的 BEV/低分辨率 3D 特征上修复跨时域空间形状；不必先重建整套相机提升网络。它可与 persistent memory 分开比较。IR-WM 的内部状态残差可作为直接可实现的已知对照，但已有 full-state predictor 不能只加上一帧特征而保持原函数。

这两个优先项针对“预测状态怎样持续读取观测、怎样维持空间细节”，不是重新调整分类阈值。仍需原生 fine-GMO、完整 GT 分辨率、相同未来 ego 条件、相同训练数据和单 seed 验证。

## 六条精读主线

| 方法与阅读范围 | 可借鉴内容 | 对 M0 的适用边界 |
|---|---|---|
| **Drive-OccWorld，AAAI 2025**；正文 §3.1–3.3、§4.1、表1–2，作者 detector/head/config | BEV 历史队列、逐步未来 decoder、语义/运动条件归一化；训练使用多个几何/语义损失 | 这些是 M0 的既有基础。论文 inflated GMO、fine-GMO、GMO+GSO 是不同任务，不能用 30% 的其他表格当我们的目标值。[正文](https://arxiv.org/html/2408.14197v3) |
| **IR-WM，ICRA 2026**；§III B–C、消融 VI–IX，实际 fine-GMO 配置 | 在 latent transition 内预测增量，累加后再入队；语义/ego alignment 降低逐步误差 | 开源 fine-GMO 仍一槽，并设 `sem_norm=False`、`ego_motion_ln=True`；论文完整 alignment 不能视作该配置全启用。未来特征 L2 teacher 消融未稳定获益。[正文](https://arxiv.org/pdf/2510.16729) |
| **OccProphet，ICLR 2025**；§3.2–3.4、组件消融，UNetPredictor/ConditionalGenerator | 历史在 scene/BEV/height 三视角时序融合，条件生成并行未来；refiner 拼接观测历史和预测未来 | 其 voxel 管线不能直接替换 M0 BEV head。需要压缩通道；AR/NAR、历史条件化与 refiner 本身均已有先例。[正文](https://arxiv.org/pdf/2502.15180) |
| **EfficientOCF，CVPR 2025**；v1 §III A–C、作者 detector/config | 2D occupancy＋height 解耦，以及基于 backward centripetal flow 的实例关联、未来掩膜修正 | 需要实例/flow 监督；height 填充改变 3D 形状假设，C-IoU 又改变评估。可借鉴关联机制，不能换指标后宣称超过 M0。[正文](https://arxiv.org/html/2411.14169v1) |
| **COME，NeurIPS 2025**；v1 §3.1–3.4、§4.1 和结果设置，作者 UNet/config | 在共享场景参考系并行预测；历史驱动的 U-Net 再以 ControlNet 约束生成模型 | M0 已有 ego 对齐，不能把它重新包装成新意；可借鉴固定参考系和几何控制，完整 VAE/DiT 多阶段路线成本较高。Occ3D 及 GT occupancy/trajectory 条件与 M0 不同。[正文](https://arxiv.org/html/2506.13260v1) |
| **OccTENS，RA-L 2026 作者接受稿**；v2 §III A–D、表I/III/IV | 分离时间 next-scene 与空间 next-scale，尺度内并行生成；ego pose 作第零尺度 token | 需新 VQ tokenizer，未找到已核验作者代码/许可证。默认六尺度 0.56s 并不快于表中 OccWorld 0.35s，降到两尺度才更快；不能仅依据摘要说全面提速。[正文](https://arxiv.org/html/2509.03887v2) |

日期与发表状态区分：Drive 首稿 2024-08-26；IR 首稿 2025-10-19、v3 2026-02-08；OccProphet 首稿 2025-02-21；EfficientOCF 首稿 2024-11-21；COME 首稿 2025-06-16；OccTENS 首稿 2025-09-04、v2 2026-03-18。OccTENS 的 RA-L 接受状态来自作者 PDF 页眉，尚未另核出版社卷页。EfficientOCF/COME 的正文精读版本与最终会议版本没有逐条 diff，不将它们混称最终全文核验。

## 可执行代码位置、依赖与成本

- **M0 接口**：[future_pred](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:465) 产生状态，580–585 同步滚动 feature 与 `ref_to_history_list`。`[B,M,HW,256]` 输入加 [prev_frame_embedding](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_base.py:406)；[cross-attention](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/modules/world_decoder.py:215) 将每槽当一个 level。这允许 runtime adapter，不要求重写原仓库。
- **两槽初始化**：复制 frame embedding；按 `[head,level,point,(xy)]` 复制 `sampling_offsets` 和 `attention_weights` 的输出行。softmax 跨 level×point，两个相同槽各分一半权重，相加应还原单槽。feature、reference points、ego 条件 normalization 必须同样复制。首步输出等价要实际数值测试，不能只比参数数量。
- **IR-WM**：[递推代码](/Users/yiminghua/2026Summer/WorldModel/dropple/code/vision-baselines/IR-WM/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:374)，Apache-2.0，HEAD `a83e4a24…`。本地有恢复 occupancy 返回字段的兼容补丁，残差行未改。旧依赖 Python3.8/torch1.10.1/mmcv1.4/mmdet3d0.17.1，与 M0 同族；fine-GMO 配置 24 epoch、每 GPU batch1、lr2e−4。
- **OccProphet**：[UNetPredictor.forward](/Users/yiminghua/2026Summer/WorldModel/dropple/code/vision-baselines/OccProphet/projects/occ_plugin/occupancy/necks/unet_trans.py:651)，MIT，HEAD `d41f8ae9…`；核心文件未改。Python3.7/torch1.10.1cu113/mmcv1.4/mmdet3d0.17.1；fine-GMO 配置24 epoch、batch1、lr3e−4。作者称最低一张4090，首轮标签生成会额外耗时，这不是 H200 实测成本。[作者代码](https://github.com/JLChen-C/OccProphet)
- **特别避免膨胀**：[ConditionalGenerator](/Users/yiminghua/2026Summer/WorldModel/dropple/code/vision-baselines/OccProphet/projects/occ_plugin/occupancy/necks/unet_trans.py:187) 的动态核生成层参数约 `T²*Tfuture*C³*k³`；T3、future4、C256、k1 时仅该层约6.04亿参数。先压到C16/32，或另定义静态/低秩预测器。`leave_attn_unused=True` 还有作者明确标注的 legacy 默认，新训练不应无意保留闲置参数。
- **EfficientOCF**：MIT，作者 HEAD `bc2289fa…`；`projects/occ_plugin/occupancy/detectors/efficientocf.py` 及 `projects/configs/baselines/EfficientOCF_V1.1.py`，15 epoch、batch1，README 示例8卡。依赖与 OccProphet 同一旧栈；报告82.33ms是作者推理数字，不是我们训练耗时。[作者代码](https://github.com/BIT-XJY/EfficientOCF)
- **COME**：Apache-2.0，核验 HEAD `d88378b4…`（2026-07-01）；`occforecasting/occforecasting/models/unet.py:187` 与 `unet_aligned_past2s_future_3s.py`。环境 torch2.0.1cu118/mmcv2.0.1/mmdet3d1.4 与 M0 栈不同。论文三阶段标2000/12/1000 epoch，不能仅将中间12 epoch称作整方法成本；可独立借鉴 U-Net 结构。[作者代码](https://github.com/synsin0/COME)
- **OccTENS**：论文两阶段 tokenizer/generation，训练 GPU 小时、作者仓库及许可证均未核实。暂列概念路线。

所有移植后的 H200 显存与训练小时均未知；只允许短 profile 后锁预算。单 seed 是用户约束，不能将一次训练的场景 bootstrap 当作训练随机性复现。

## “固定观测＋生成状态”已有多近的先例

**一般原理已有直接先例。** StreamingT2V（CVPR2025）用最近生成片段的 CAM 和初始 anchor 的 APM，明确同时保留短期动态与长期外观。但其 anchor 可以是生成帧，不等于物理传感器测量。[作者方法页](https://streamingt2v.github.io/)

WorldMem（NeurIPS2025）的记忆单元包含帧、pose、time，通过相对状态 cross-attention读取滚动窗口外的历史；其记忆也会存生成帧。OccProphet 的观测/预测联合 refiner 与 COME 的历史控制则是 occupancy 内最近的功能先例。没有在本轮有界检索中找到与“两槽、按各自 SE(3)、固定真实观测 versus 滚动生成、同容量”完全相同的 M0 实验；**这不是未发表性证明**。[WorldMem §3.3–3.4](https://arxiv.org/html/2504.12369v2)

最强反例是陈旧锚：静态结构可持续利用，但动态车辆/行人的旧位置可能妨碍未来预测。仅提高 GMO IoU 也不能证明运动建模提高，因为 GMO 是可移动语义类而非实际运动标签。实验应优先保持原协议，报告四时域、原始 NLL/Brier/IoU 与可解释的注意力/来源干预；真正运动分组须另有合法速度或实例轨迹来源。若 persistent 与 rolling 同样提升，只支持更大的可学习记忆；若只在固定锚提升，才支持持续观测访问的作用。
