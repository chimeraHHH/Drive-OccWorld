# Train512 固定渲染内容 T：开发集结果与研究决策

2026-09-17。**当前 T checkpoint 未超过 O，物理运动也未超过 CRN-CV；不进入 full5119 或少帧效率实验。** 原已冻结的 J/D 仍按服务器顺序完成，用于比较渲染内容演化与直接读出的影响。本结果不否定所有运动传播方法，也不把四轮训练视为充分收敛的证明。

## 执行与独立复算

T 从已核验的 shared current codec 开始，seed11、train512/256 scenes、四次完整 permutation，共 2,048 次 AdamW 更新；冻结 encoder/decoder，训练 52,720 个 recurrent/content/velocity 参数。训练用时 4,568.25 s，峰值分配 2.753 GiB。最终一次 dev200/100 scenes 评价用时 265.69 s，零优化更新。没有 best-checkpoint 选择、追加调参或从开发集恢复训练。

[独立 CPU 审计](dense_material512_T_evidence_v1.json)重新核验了 9 个文本文件哈希、全部 2,048 更新顺序、512 样本身份、200 样本的原 O 完整记录与 GT 分母，以及 796 个聚合标量。源脚本为 [audit_dense_material512_results_v1.py](audit_dense_material512_results_v1.py)，不导入 producer 的聚合或指标函数。本轮没有独立重跑原始预测或 CPU 重载 T 权重；原 evaluator 已在真实 GPU 评价前检查最终参数及冻结模块身份。

- T training complete SHA256：`1d6b86e89b097c075d74a7f7cc01365e6b3c18e2addb4b247171ef1ea52f3b2f`
- T evaluation complete SHA256：`705675bfd1e429297a507d19ad52da7e7bb6c90af6d57af46b9964e950de0815`
- T evaluation records SHA256：`243385e57484588486952d8da1073d82c5aace9990d2e2833f458c64f970a14b`

## 原始未来占据任务

主指标为每个 horizon 跨样本池化 confusion 后的 GMO IoU，再对四个未来 horizon 取均值。GMO 是二分类 movable occupancy，不是语义 mIoU。moving/stationary recall 的主报告同样先按 horizon 池化再平均；证据 JSON 另列跨时域一次池化，不能混用两种平均。

| 原 dev200 指标 | O | T |
|---|---:|---:|
| 当前 GMO IoU / % | 15.9191 | 14.2910 |
| 未来 GMO IoU / % | 14.6640 | 9.2876 |
| moving >0.5 m/s recall / % | 43.4521 | 30.2357 |
| stationary ≤0.1 m/s recall / % | 47.4253 | 42.1910 |
| 到达 01 IoU / % | 8.1234 | 3.7572 |
| 离开 10 IoU / % | 9.9818 | 8.4546 |
| 四未来时域 FP 总数 | 13,647,997 | 20,657,517 |
| 四未来时域 FN 总数 | 3,732,899 | 4,242,421 |

T−O future IoU 为 −5.3764 pp，场景配对 bootstrap 95% 区间 [−5.8792, −4.8697]；moving recall 为 −13.2164 pp，区间 [−15.9098, −10.5623]。100 scenes、10,000 次重采样、seed11；这描述已暴露开发场景的抽样变化，不是训练随机性的区间，也不是独立确认。

## 关键诊断：学习传播有没有胜过自己的当前保持

现有转移矩阵保存 `(GT_t0,GT_h)` 与 `(pred_t0,pred_h)` 的完整联合计数，编码 `2*t0+h`。以 `GT_h=GT_state%2` 和 `pred_t0=pred_state//2` 求边际，可以精确得到“保持自己的当前预测”的未来 confusion。所有 200 个样本的共同有效域均等于全网格；重建的当前和正常未来边际全部精确复原。独立 Luna 用另一份 CPU 代码再次确认了 3,200 个矩阵检查及全部 persistence 数字，无新增 GPU 推理。

| 设置 | +0.5 s | +1.0 s | +1.5 s | +2.0 s | 未来均值 / % |
|---|---:|---:|---:|---:|---:|
| 保持 T 当前输出 | 11.1795 | 9.6844 | 8.9714 | 8.5642 | 9.5999 |
| T 学习传播 | 11.7778 | 9.7189 | 8.3066 | 7.3471 | 9.2876 |
| 保持 O 当前输出 | 12.4159 | 10.7345 | 9.8507 | 9.3930 | 10.5985 |
| O 原生未来预测 | 15.7578 | 15.1320 | 14.2952 | 13.4710 | 14.6640 |

T 固定渲染内容的 zero-displacement 干预与当前输出保持恒等，故该边际就是此干预的占据结果；这个解释不适用于内容仍演化的 J/D。O 的行只表示保持 O 当前预测，不意味着 O 有一个被置零的 flow head。

T 学习传播相对自身保持为 **−0.31225 pp**，95% 区间 [−0.63780, +0.04898]，没有显示稳定收益；O 相对自身保持为 **+4.06546 pp**，区间 [+3.33981,+4.87079]。T 传播减少 FN 403,242，却增加 FP 5,591,519。

数值上可以精确拆为：`T−O = (T−persist_T) + (persist_T−persist_O) + (persist_O−O) = −0.31225 −0.99866 −4.06546 pp`。这是不同端点的代数分解，不是三个独立因果效应。它表明差距不能只由两者当前 IoU 的不同解释；O 已有的未来预测能力必须被新方法保留和超越。

## 物理误差：比较强基线，而非只比较零位移

[物理参考审计](dense_material512_T_physical_reference_v1.json)绑定既有 CRN-CV 官方冻结预测、相同 dev200 index 与相同 sparse-label manifest。逐 anchor/horizon/group 的对象数、点数全部一致，zero-target 均值范数的最大数值差为 7.70e−7 m，来自旧参考更高精度的范数计算，低于预设 1e−5 m 边界。

以下均为固定 t0 坐标中的 **XYZ 刚体框代理 EPE**，先对对象内源点平均，再对 anchor-instance 等权；不能与旧报告的 XY EPE 直接混比。

| 第四未来 endpoint（名义 +2 s） | CRN-CV | T | T−CRN-CV，场景 95% CI / m |
|---|---:|---:|---:|
| moving，1,226 个 anchor-instance | 3.4093 m | 4.9610 m | +1.5516 [1.2994,1.8693] |
| stationary，2,136 个 anchor-instance | 0.1697 m | 0.5714 m | +0.4017 [0.3266,0.5065] |
| ambiguous，487 个 anchor-instance | 0.5322 m | 0.7174 m | +0.1852 [0.0987,0.2883] |

T moving EPE 虽优于零位移的 7.5217 m，却不足以称为运动改进：它在四个 horizon 的三组均落后于已有 CRN-CV。O 本身仍没有可比的物理运动头。

## 机制边界和下一步

独立源码审查未发现 XYZ、正负方向或位移累计的明显错误：T/J 用于占据 splat 的位移就是稀疏 physical loss 采样的同一源索引位移；D 的占据路径有意绕过 velocity。T 的 recurrent content 仍会影响未来位移，因此 T/J 检验的是渲染内容是否演化，不是对象身份是否恢复。

模型时间输入和积分采用名义 0.5 s 步长，监督则来自实际未来 endpoint；不能把输出解释为已校准的连续时间速度。GT 时间及评价人口在各臂保持一致，此局限不会取消当前 checkpoint 的负结果。

当前 codec 只为当前占据训练后被冻结，不能由当前 IoU 证明它保留了未来预测所需的运动信息。另一方面，frozen codec 也不构成数学上的未来性能上限；训练平均 occupancy loss 从第一轮 0.3379 降到第四轮 0.3250，仍不足以证明完全收敛。不能把这一次固定预算失败扩大成所有 latent transport 方法失败。

下一步保持已冻结 J/D 流程，先完成三组同输入结果，随后区分“可预测状态本身不足”与“内容/几何传播读出不足”。不对当前 T 扩展全验证或直接执行六条件少帧训练。新模型若推进，应以保留 O 的原生预测能力、保留可用动态信号和终端 motion/occupancy 双收益为标准；不以更好的训练拟合、低于零位移 EPE 或通过工程检查替代目标。
