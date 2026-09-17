# 固定 T 最终权重的训练集拟合诊断

2026-09-17。当前 T 的 dev200 未来 GMO 为 9.2876%，低于 O 的 14.6640%，也没有稳定胜过保持自己当前输出。在线训练 loss 不能回答最终 checkpoint 是否已经拟合训练数据。因此对同一 T2048 最终权重补充完整 train512/256 scenes 的预测评价，零优化更新。

本诊断不修改正在执行的 T/J/D 服务器序列，不重新训练或选择 checkpoint，不启动多种子。单独使用空闲 GPU1；每个作业有独立目录、进程组、超时和完成记录。两样本执行检查与完整512评价是不同作业，不能将前者计为训练集结果。

## 冻结的输入和评价

复用原 dev evaluator 的模型、参数验证、当前 BEV 输入、五个 fine XYZ 预测、原 GT `[0,2:]`、0.5/1/1.5/2 s horizon、原始 movable occupancy 和 rigid-box 位移代理指标。标签坐标与实际 endpoint 时间不改变。512个训练样本逐一与真实训练日志的 input、target、sparse-label 哈希绑定。原始 box 标签使用 `train/` 路径；不会调用只接受 development 路径的旧 helper。

输出保留每个样本的完整占据/状态转移混淆矩阵，以及按对象等权的物理误差和 GT 分组分母。先取得预测，再读取该样本标签内容。模型参数在评价前后逐字节验 digest，且无 optimizer。没有认证过的 O train512 预测，因此不报告 O 的训练集分数，也不把缓存附带的其他预测当成 O。

## 解释规则

- 如果训练集传播也不能改善相同当前输出的 persistence，则应优先研究表征、目标连接、读出能力与优化是否足以拟合；不能仅归因于开发集泛化。
- 如果训练集收益明显而开发集消失，则进一步检查场景分布和泛化，不能直接把两个不同场景总体的分数差当成纯过拟合因果效应。
- 是否胜过 persistence 需要同一有效 GT 域；若 GT ignore 在时间间变化，联合转移矩阵只能提供共同有效域的 persistence，不能冒充全域值。
- 当前冻住的 codec 只做过当前占据监督，T 小型预测头的失败不等于所有 transport 方法失败；一次2048更新也不证明收敛。
- J/D 按原协议完成后再合并机制解释。新路线应保留 O 的有效预测能力以及协议允许的未来 ego/action/query 输入。减少相机历史的实验仍是独立待验证方向，不以 T 的诊断耗时代替端到端效率。

实现：[evaluate_material512_train_fit_v1.py](evaluate_material512_train_fit_v1.py)。冻结协议：[material512_train_fit_protocol_v1.json](material512_train_fit_protocol_v1.json)。
