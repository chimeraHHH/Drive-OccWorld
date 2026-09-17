# 场景状态、观测更新与传感器证据：论文和官方代码复核

日期：2026-09-17。根 agent 独立阅读，配合两位 GPT-5.6-Luna 子 agent；未使用 skill。下表包含世界模型及解决相关几何/融合问题的邻近论文，不能把六篇都称为未来世界模型。GaussianWorld、RaCFormer 阅读方法和实际核心源码；其余四篇是官方论文摘要/相关方法材料与 README 筛选，未完成逐行实现审计。代码快照、commit 和下载摘要位于 `topconf_repo_audit_20260917/`。

## 论文与可迁移边界

| 工作及正式会议 | 有价值的机制 | 与本项目的关键差别 | 官方实现及权重核验 |
|---|---|---|---|
| [GaussianWorld，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html)；[正文](https://arxiv.org/html/2412.10373) | 持久 Gaussian 状态、ego 对齐、局部位置变化、新进入区域填充，利用当前图像修正状态。 | Streaming 当前占据感知，每步有新图像；不是未来无观测 rollout。语义可移动类也包含停着的车。 | [zuosc19/GaussianWorld](https://github.com/zuosc19/GaussianWorld)，commit `b43629eaecffd5a7cbaac1a55517766e6263e4fc`；官方 stream 权重入口 HTTP 200 且标题为 `ckpt_stream.pth`，未下载文件/复现。 |
| [GaussianFlowOcc，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Boeder_GaussianFlowOcc_Sparse_and_Weakly_Supervised_Occupancy_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf) | 稀疏 Gaussian 表示、时间关联与弱 2D 监督。 | 核心任务是占据估计；不能把跨帧 flow 的重建作用直接说成未观测未来预测收益。 | [boschresearch/GaussianFlowOcc](https://github.com/boschresearch/GaussianFlowOcc)，`42b03b3aba47abe7aaee6d868713307450c15423`；有训练/推理代码，README 示例 checkpoint 路径不等于公开下载；本轮未认证其自身模型权重。 |
| [RaCFormer，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html)；[正文](https://arxiv.org/html/2412.12725) | query 同时取 radar BEV、camera BEV 和原生图像特征，以预测速度回查历史位置，减少深度转换造成的融合错位。 | 当前 3D 检测及速度估计；不是未来占据，也不是每个材料点的 3D scene flow。mAVE 受检测匹配支持范围影响。 | [cxmomo/RaCFormer](https://github.com/cxmomo/RaCFormer)，`4d4c534f84acef9ec390523134367aad40619057`；官方 Google Drive 入口 HTTP 200，标题 `racformer_r50_f8.pth`；未下载权重/编译扩展。 |
| [EmbodiedOcc，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_EmbodiedOcc_Embodied_3D_Occupancy_Prediction_for_Vision-based_Online_Scene_Understanding_ICCV_2025_paper.html) | 随观测到来局部更新、保留全局空间记忆。 | 室内在线场景理解，不是道路动态未来。保留未观测状态这一原则可用，场景和动态假设不可照搬。 | [YkiWu/EmbodiedOcc](https://github.com/YkiWu/EmbodiedOcc)，`6cf48a7148136b2a9dd698d0f74f08152b82ea2f`；核到数据链接，未认证自身模型权重入口。 |
| [GaussRender，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Chambon_GaussRender_Learning_3D_Occupancy_with_Gaussian_Rendering_ICCV_2025_paper.html) | 把预测/标注占据渲染成深度与语义视图，利用投影一致性训练几何。 | voxel、ray、表面指标不等价。我们的二元 GMO 后景包含空空间及静态物体，不能直接当几何 opacity 做射线遮挡。 | [valeoai/GaussRender](https://github.com/valeoai/GaussRender)，`3e0d6e74e09fa4482dd43f2ab881d4c748383d1f`；训练/测试与 checkpoint 路径可见，未认证权重 blob。 |
| [SOAP，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_SOAP_Vision-Centric_3D_Semantic_Scene_Completion_with_Scene-Adaptive_Decoder_and_CVPR_2025_paper.html) | 遮挡感知的视图投影与场景适配查询，针对同一射线上重复/错位图像特征。 | SemanticKITTI/SSCBench 当前语义补全，不是 nuScenes 未来预测；图像投影落在画面内不等于点可见。 | [gywns6287/SOAP](https://github.com/gywns6287/SOAP)，`9bebb597e7a099660818b6c1d6c74e283a45f57b`；README 有四个 Google Drive model-zoo 入口，本轮未打开/下载。 |

## GaussianWorld 实际代码给出的边界

固定 commit 下，`model/segmentor/gaussian_segmentor_stream.py::forward` 的当前图像特征进入每步状态更新；历史状态经 decoder ego 对齐，随后 `obtain_anchor` 使用当前图像做更新，返回历史状态还会 detach。不能用这个 streaming 性能证明无图像 future transition 已学到。

`model/decoder/gaussian_decoder/gaussian_decoder_stream.py::warp_anchor` 对历史锚点作 ego 变换，过滤超出范围的锚点，再为新区域填充锚点。这是空间边界进入机制，并非完整的遮挡、出生、消亡估计器。

实际使用的 `model/encoder/gaussian_encoder/refine_layer.py::forward` 对历史 anchor 的语义用 `argmax < 11` 调整可运动更新权重，第一阶段限制非位置属性，后续才联合精炼。它使用“可移动语义类”的先验，并没有认证物理速度。该事实与我们 GMO 的语义边界一致：停车车辆仍可属于 movable 类。

可借鉴的是把**持久状态、ego 对齐、动态变化、观测修正**分开。若迁移，图像更新只允许发生在观测前缀，未来只能递推状态；不能把当前感知模块带着未来图像移植进去。Gaussian 仅是可选承载形式，不是贡献本身。

## RaCFormer 实际代码给出的边界

`models/racformer_transformer.py::RaCFormerSampling.inner_forward` 使用 `query_ray[..., 8:].detach()` 的预测速度与历史 `time_diff` 相乘，回移 sample points，再沿 query ray 设置深度采样；同文件 `BEVSampling` 也有历史速度补偿。decoder 从 radar BEV、camera BEV 和原生视图取特征。说明融合质量不仅取决于两个模态是否同格，还取决于查询地址和时间。

但速度在此采样路径 detach，不能据此声称原生图像采样误差端到端优化了速度；视图投影的画面内判断也不等于遮挡检验。检测输出的 box/velocity 与本项目所有原始材料点的 EPE 分母不同。完整部署还涉及旧版 torch/mmcv 与自定义 CUDA 算子，README 可运行命令不是本环境已成功运行的证据。

对我们的直接含义：目前雷达同格覆盖很低，扩大融合层数不一定增加独立信息。应先验证相机/雷达证据绑定到哪个空间实体、哪个历史时刻，再决定需要怎样的状态更新。没有雷达回波不能自动把速度归零；可见性下降也不能删除已经建立的运动状态。

## 建议与停止条件

优先研究一个问题：**部分观测下，怎样持续维护正确绑定的场景状态，使运动更新被未来占据解码器实际使用？** 将观测更新与无观测预测分开；静态场保留，动态实体携带运动，当前未观测但历史存在的实体仍可递推，新出现区域由独立补全处理。

先在已有冻结结果上验证 query 的几何/表面有效性和证据特异性。若真实历史对应不优于匹配速度、距离、覆盖率的错时/错对象对应，不进入复杂融合训练；若只改善材料点 EPE 而未来 GMO 和移动占据没有改善，则不宣称完成任务连接。新方案需要面对完整八阈值 D-speed 路由的收益—静止代价曲线，不能只胜过 CRN-CV。

本轮没有下载这些论文的模型权重、启动训练或修改服务器任务。所有新方案是研究建议，不是已实现模型或结果。
