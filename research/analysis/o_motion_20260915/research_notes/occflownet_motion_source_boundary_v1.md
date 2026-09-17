# OccFlowNet：运动变换的来源与适用边界

2026-09-16。只读论文摘要和作者官方代码；未安装、训练或运行模型。CVF正式PDF本次打开返回403，不能声称读过其全文。官方仓库锁定commit `27e102e467b771651977e69d3bc0b10177ff6779`，四个实际读取文件及SHA保存在 `occflownet_source_audit_v1/sources.json`。

**代码核对。** `tools/create_flow_data.py` 用跨帧GT instance token对应框，构造目标位姿乘源位姿逆矩阵，生成框内格点搬运位置。`occflownet_stbase_2d_flow.py` 的训练pipeline加载 `LoadFlowGT`。detector的 `forward_train` 把外部voxel_flow/flow_transforms交给renderer；其 `simple_test` 返回语义占据和自由空间。默认renderer的clipped路径按GT流地址搬运匹配类别的预测density/semantics。该已读代码路径支持利用已知运动改善跨帧渲染监督，不能作为“仅凭当前观测预测0.5–2秒未来刚体状态”的直接证据。[GT变换生成](https://github.com/boschresearch/OccFlowNet/blob/27e102e467b771651977e69d3bc0b10177ff6779/tools/create_flow_data.py#L139-L169) · [模型入口](https://github.com/boschresearch/OccFlowNet/blob/27e102e467b771651977e69d3bc0b10177ff6779/mmdet3d/models/detectors/occflownet.py#L80-L141) · [渲染搬运](https://github.com/boschresearch/OccFlowNet/blob/27e102e467b771651977e69d3bc0b10177ff6779/mmdet3d/models/occflownet_modules/renderer.py#L400-L449)

**对本项目的推论。** 刚体变换参与占据构造已有先行工作，本轮不能据此主张新颖性。更有判别力的问题是：在推理端只能使用当前预测框、预测速度和观测特征时，学习出的位姿修正能否同时降低完整支持运动误差并提高原O未来GMO。Cpl/Fix保持相同CV基底和对象信息，只比较学习位姿是否进入空间核；若只在GT位姿下有效，或只改变占据保守程度而运动不优于CV，目标仍未达成。这些是当前实验设计的推论，不是该论文的结论。
