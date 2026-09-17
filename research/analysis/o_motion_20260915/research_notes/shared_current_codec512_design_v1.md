# 扩大到train512：共同当前表示训练

日期：2026-09-17。T/J材料读出比较已完成，见结果报告。下一阶段不继续调train4，而从原train512/256场景学习共同current codec，供后续候选和对照使用。

fresh seed11初始化相同DenseTaskState v2，encoder/decoder训练，所有递推和运动参数保持原初始化。仅使用原target的当前时刻[0,2:3]监督，原CE[1,5]+Lovasz、legacy occupied-mode标签下采样，2048 updates=4次完整512遍历。单个RandomState11连续生成四个排列，不每epoch重新seed。CPU LRU最多8样本，不改变输入/目标文件；目标全文件SHA验证不等于将未来target送入模型或loss。

此前同种子J出现约1.29pp IoU差异，下一阶段明确固定数值策略：全FP32关闭两种TF32、cuDNN deterministic、严格deterministic_algorithms=True、CUBLAS_WORKSPACE_CONFIG=:4096:8。实际服务器算子检查发现CUDA trilinear backward不支持严格确定性；因此原数学插值、CE和Lovasz改在CPU执行，通过可微device copy把梯度传回GPU codec。原标签下采样函数也在CPU执行。推理/细网格评分仍使用原XYZ trilinear规则。

已完成一次真实GPU模型图检查：合成稀疏current目标包含占据/空/ignore，CPU与GPU原标签下采样完全一致；同一模型输入重复两次，CPU loss+GPU codec的loss和梯度SHA完全相同，梯度有限，零optimizer updates。该检查只验证算子/数值路径，不是数据性能或跨进程训练完全复现证明。收据在current_codec_determinism_evidence_v1.json。

Root审查修正了helper初稿中的真实索引问题：official_index是原数据集索引，不是0..511；完整输入顺序由原cache index SHA绑定，另外核验512不同sample/256scene。去除每次cache miss的冗余模型forward，保留实际训练/评分forward和输入认证。没有改变原数据或选择更有利样本。

最终保存shared_codec.pth和原512样本current fine-grid指标，不选best、不读development、不声称forecast胜过O。后续T/J及直接未来状态对照均应使用同一个新codec，并采用新的固定数值政策；不能将新政策与旧train4结果直接混为一组。模型在训练集当前占据仍拟合不足时，先报告瓶颈，不靠对比旧O完整future数字声称进步。

内部上限3600s/allocated24GiB，外部runner3660s；共享H200，保留其它任务。训练由服务器独立托管，无自动重启。外部官方baseline不重训，O/M0不改动。
