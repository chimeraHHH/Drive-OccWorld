# 持久内容、运动与未建模变化：定向文献和源码核对

日期：2026-09-17。依据当前 T/J 问题进行定向核对，不重复广泛列论文；没有外部模型训练或权重下载。下面分开列官方来源与对本项目的推论。

## DDPAE：共享内容、预测位置

[NeurIPS 2018 原文](https://papers.neurips.cc/paper_files/paper/2018/file/496e05e1aea0a9c4655800e8a7b9ea28-Paper.pdf)与[作者仓库](https://github.com/jthsieh/DDPAE-video-prediction)均已核对。该模型将视频分成组件，并分离内容和位置变化；论文主要验证 Moving MNIST / Bouncing Balls。不是道路三维材料流的证据。

固定 commit `219e68301d24615410260c3d33c80ae74f6f2dc3` 的 [DDPAE.py](https://github.com/jthsieh/DDPAE-video-prediction/blob/219e68301d24615410260c3d33c80ae74f6f2dc3/models/DDPAE.py#L322) 中，decode 将每组件单个 content 沿全部时间重复，再由 decode_components 将解码的组件与 pose 送入 object_to_image；sample_latent 从输入窗口提取 content。PoseRNN 单独预测位置转移。这是本次具体读到的实现，不仅来自摘要。

**对我们的推论：** T 保持 rendered C0、保留独立递推运动状态是有先例的结构约束；不能把“固定内容+预测运动”本身作为新贡献。它适合反驳自由 future-content 是否必需，但我们的 dense C0 没有组件身份认证，含背景与混合 voxel，不能升级为 object-centric 模型。雷达部分可观测速度与有限视觉历史之间的替代关系，需要另外证明。

## PhyDNet：受约束动力与补充变化并存

[CVPR 2020 官方论文页](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html)确认其用物理约束分支和补充分支表达一般视频。官方源码 commit `23a992d771c9eb1d32f52b1873a3c5625f1a8413` 的 [models/models.py](https://github.com/vincent-leguen/PhyDNet/blob/23a992d771c9eb1d32f52b1873a3c5625f1a8413/models/models.py#L261) 在 decoding 时不向物理分支注入新观测，同时保留 ConvLSTM 分支；两路专属解码结果相加后再进入公共解码器。它不是严格物理质量守恒，也没有认证道路物体速度。

**对我们的推论：** 如果固定 C0 限制占据预测，不宜直接认定运动路线失败。需要区分持久内容迁移与未建模变化。不过直接加一个自由补充分支会重新引入绕过运动的路径，因此必须限制其适用范围，并通过真实位移干预和共同物理指标检验；“两分支”不能自动保证分工可辨识。

## GaussianWorld：新观测补全不能借入无观测未来

[CVPR 2025 官方论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html)将场景演化分为自车对齐、动态局部运动和新观测区域补全，并明确条件中有当前 RGB 观测。本次复核该任务边界，未新增其源码深读。

**对我们的推论：** 道路场景需要处理进入视野与遮挡变化，但我们预测 t0 之后时刻时没有未来 RGB。不能沿用 streaming 的观测更新来填未来而称作 forecasting。先在当前信息下验证固定内容的限制，再设计合法的未来新生/不确定状态；不借未来观测作输入。

## 当前决定与可追溯性

先完成 T/J 读出对照：相同 recurrent state、相同参数和原目标，只改变 renderer 使用 C0 还是 Sh。不改成冻结 content_increment，因为那会连带移除 recurrent 状态更新。T 早期 increment 仍可经后续 velocity 收到占据梯度；最后 increment 的直接读出路径才是移除对象。

固定内容对照的负结果只能暴露这一表示与 decoder 下的限制，不足以证明物理运动无用；正结果也仅表明 train4 上的可用性。下一阶段必须在 train512 和单独确认场景检验，同时维护 O 原占据任务和 moving/stationary 物理准确性两条证据线。

官方下载源码及每文件 SHA256 在本地 `material_state_source_audit_20260917/receipt.json`，第三方源码镜像不随研究 snapshot 发布。读取了 DDPAE.py、decoder.py、pose_rnn.py 与 PhyDNet 的 models.py；没有声称通读其全部训练、数据或实验实现。没有新增官方 baseline 重训练任务。
