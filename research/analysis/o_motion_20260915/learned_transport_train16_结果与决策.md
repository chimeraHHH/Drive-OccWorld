# 学习位移的稠密传输诊断

2026-09-15。真实运行正常完成，25.23秒；固定16个训练anchors/8场景，零新增训练。新运动头已在包含这些样本的训练集上训练，不能据此声称泛化。

| 条件 | 四未来平均 GMO IoU |
|---|---:|
| O | 21.46860% |
| static_splat_O0 | 14.79517% |
| learned_splat_O0 | 17.04915% |
| blend_static_half | 17.93818% |
| blend_learned_half | 20.67095% |
| blend_reverse_horizons_half | 19.88518% |

**学出的位移确实比零位移传输有用，但固定各半融合仍低于O。** learned transport相对static提高2.25397个百分点；learned各半融合相对static各半提高2.73277个百分点，相对O下降0.79765个百分点。四个未来时域全部低于O，不能挑短时域或训练样本称为成功。

将同一运动头的四时域输出反序后，各半融合降到19.88518%，相对正常顺序下降0.78577个百分点。这是时序连接敏感性的同checkpoint干预，不等同同预算训练对照，也不单独证明物理运动正确。

相对O，正常learned各半融合四未来 pooled FP变化+8862、FN变化+11025。这里两种错误都增加。全部t0输出逐字节保留O，原O五时域混淆逐样本精确复现既有oracle诊断；没有读入GTbox或GTflow作为推理支持。

诊断对所有稠密源点使用预测位移，包括缺乏运动监督的背景/重叠区域；源是O当前softmax概率，无GT安全掩码。后续预先提出的预测前景支撑与局部门控是另一个有训练的候选，不得把这个诊断改名为它的结果。应与同容量、同初始化、同数据/更新的零位移连接门控对照，验证收益是否来自运动。

源和结果位于 `learned_transport_probe_v1.py`、`learned_transport_protocol_v1.json`、`server_results/training/learned_transport_train16_v1/`；complete与全部统计已由根独立复算。O/运动头原权重保持不变。
