# O 的 t0 与未来预测：共享读出和监督连接审计

**O 仍训练 t0；t0 和未来共享可训练的 occupancy 读出，但未来状态不读取 t0 语义 logits。** 更重要的是，O 的 t0 相对同预算 native1 控制 C 实际提高，而非下降。不能把相对原始 M0 的下降直接归因于移除 sem/geo。

本报告只读已冻结且 SHA 匹配的本地源码、协议和完整 dev200 汇总；未读取 full validation 的任何部分结果，未连接服务器、训练、改代码或构造混合模型。

## 先核对比较对象

来自完整开发汇总 `server_results/campaign_objective_v1/summary_v1/summary.json`（SHA `46c37b5b074e73896e357e0bc9744a538abb01b288521769a577f459be54ad6b`，已核 completion 的 summary hash）：

| 模型 | t0 GMO IoU，% | 四个未来时域 GMO 算术平均，% |
|---|---:|---:|
| M0，原 TF32 | 16.07823 | 13.90385 |
| M0_fp32，同精度冻结参考 | 16.07828 | 13.90660 |
| native1 / C，同预算原 12 loss 继续训练 | 15.83397 | 13.73856 |
| O，同预算 CE + Lovasz | 15.91914 | 14.66400 |

- O − M0_fp32：t0 **−0.15914 pp**，未来 **+0.75739 pp**。
- O − C：t0 **+0.08517 pp**，未来 **+0.92544 pp**。

同一完整汇总已有 t0 的 scene-paired 95% CI，本次只读复核、没有重新抽样：O − C 为 **[+0.05842, +0.11456] pp**，O − M0_fp32 为 **[−0.20193, −0.11219] pp**。这是 100 场景、200 anchors、固定单训练 seed 的开发集不确定性，未做多重比较校正，不能改写成新主比较或训练 seed 稳健性。整个继续训练过程相对 M0 的当前时刻代价，仍未定位。

## t0 是如何产生的

冻结观察状态记作 z0。原 [`drive_occworld.py:477`][det477] 把 `prev_bev_input[:, -1]` 复制到三个 layer 槽中，初始化 `next_bev_feats=[ref_bev]`。这一步没有调用 future transformer。随后循环生成四个未来特征，每步把最后一层特征写入下一步 memory（[第 498 行][det498]、[第 577 行][det577]）。四步全部完成后，才将包含 t0 的五个时域统一堆叠，交给 `future_pred_head.forward_head`（[第 588 行][det588]）。

实际配置是 `soft_weight=False`、三层、`num_pred_fcs=1`、`pred_history_frame_num=pred_future_frame_num=0`（[`S0.py:98`][cfg98]）。因此第 l 层的形式为：

`logits[t,l] = readout_l(features[t,l])`；其中 `features[0,l] = z0`。

每个 readout 是 `Linear(256,256) → LayerNorm(256) → ReLU → Linear(256,32)`。三个 readout 是独立 deep-copy，**层间不共享参数，同一层跨五个时域共享参数**（[`world_head_v1.py:97`][head97]、[第 170 行][head170]）。单头输出的 `pred_frame_num=1` 不是“只预测 t0”：外层五时域维度已经存在，不能把它与 rollout 时间轴混淆。

原 evaluator 只取最后一个 decoder layer 的输出，再分别插值到原生完整 GT 尺寸并 argmax（[`drive_occworld.py:718`][eval718]）。所以在同一个固定 z0 上，**最终 t0 指标能直接改变的可训练参数只有 `future_pred_head.bev_pred_head.2.{0,1,3}.{weight,bias}`**，共 74,528 个。完整开发 checkpoint 审计中的实际 tensor_shapes 与上述形状一致；三组 readout 共 223,584 个参数，整 future head 为 13,274,016 个。

其它 future transformer、query/位置 embedding、action MLP、条件归一化参数可以直接改变未来状态；它们不处在当前 t0 logits 的前向路径。分类头只有 LayerNorm，没有跨样本运行统计的 BatchNorm 或 dropout；因此不能用分类头 train/eval 的随机模式差异解释固定参数下的 t0 变化。

## t0 logits 不反馈到未来状态

当前机制是**特征递推**。下一时刻 memory 来自 `pred_feat[-1]`，不来自 occupancy logits（[`drive_occworld.py:577`][det577]）。主 `forward_head` 位于四步循环之后，因此 t0 logits 产生时四个未来状态已经计算完毕。

原源码在规划开启时另有循环内 occupancy readout，但该调用严格在 `self.turn_on_plan and occ_flow == 'occ'` 分支内（[第 554 行][det554]）；本轮 M0/native1/F/O 都关闭规划。实际配置也关闭 semantic GT/预测归一化：`prev_render_neck.sem_norm=False`、`sem_gt_train=False`（[`S0.py:132`][cfg132]）。没有另一条 t0 语义反馈路径。

这意味着，**t0 下降本身不是“错误当前语义被递推放大”的证据**。共享 readout 会影响五个时域的最终判别，但其 logits 不作为本轮未来状态的输入。未来收益究竟主要来自 transition 学到更好的特征，还是 readout 对未来特征的判别改变，当前分数本身不能区分。

## O 是否监督 t0，以及真实权重

答案是明确的“是”。训练 replay 把 `valid_frames` 设为 `[1,2,3,4]`，这是未来 transition 的 autograd 开关；它没有删除 t0。统一 readout 位于该分支之外，在训练梯度上下文中执行（[`native_state_cache.py:243`][replay243]；[`drive_occworld.py:534`][det534]、[第 588 行][det588]）。冻结 z0 不需要梯度，作用于它的读出参数仍有梯度。

`compute_occ_loss` 只删除 GT 的前两个历史帧，保留 t0、0.5、1.0、1.5、2.0 秒五帧，再把时间和 batch 合并；没有 `[:,1:]` 或 future-only loss mask（[`drive_occworld.py:596`][loss596]）。O 只按 loss **族和 layer 键**筛选，保留三层 CE、三层 Lovasz 共六个原 tensor；不筛时间、不重缩放（[`objective_supervision_adapters.py:17`][adapter17]、[第 65 行][adapter65]）。

| 维度 | 原 C | O | 真实源码语义 |
|---|---|---|---|
| 时域 | 全五帧，包括 t0 | 相同 | 合并时间和 batch 后计算；没有显式 horizon 权重 |
| layer | 三层各自四族，共 12 个标量 | 三层各自两族，共 6 个标量 | 最终训练直接求和，不做 1/3 层均值 |
| CE | 类权重 `[1,5]` | 原样保留 | reduction=mean，分母为合并后有效标签权重总和 |
| Lovasz | 原 softmax，默认 `per_image=False` | 原样保留 | 五帧有效体素一起排序、按出现类别平均，仍跨时域耦合 |
| sem/geo | 参与梯度 | 原 12 项仍计算和有限性审计，但这六项不返回给优化总和 | 未修改原概率公式，不是数值公式修复 |
| GT | 256×256×20 的原 coarse/ignore 规则 | 相同 | 当前与未来都用同一规则；完整原 GT 留给评测 |

证据：[`world_head_v1.py:199`][head199]、[第 224 行][head224]；[`semkitti_loss.py:148`][ce148]；[`lovasz_softmax.py:156`][lovasz156]；[`memory_experiment.py:233`][train233]。每个 loss 族配置系数均为 1（[`S0.py:124`][cfg124]）。

**不能把五帧监督说成精确的 t0 20%、未来 80%。** CE 的实际占比受每帧有效体素和类权重影响；Lovasz 的全体排序不是逐 horizon 损失平均。C 中 sem/geo 也对合并后的有效体素做全局比例统计。配置里的 `per_frame_loss_weight=(1.0,)` 只被保存和长度检查，`loss_weight=[[1],[1],[0]]` 只被保存/检查；在当前 `WorldHeadV1.loss_occ` 实际路径均未使用。因此后者不代表最后层 loss 被置零。

## 精度和模块状态，以及解释边界

原缓存观察器为 TF32=True；M0_fp32、C、O 使用完全相同缓存 z0，future 计算 matmul TF32=False、cuDNN TF32=True，float32。正式训练只解冻整个 future head；观察器参数和缓存输入均固定（[`memory_experiment.py:26`][prep26]、[第 42 行][prep42]；[`objective_supervision_train.py:223`][otrain223]）。训练 replay 进入 train/enable_grad，最终评价进入 eval/no_grad（[`memory_experiment.py:68`][eval68]）。C 与 O 共享初值、参数名/形状、顺序、512 updates、AdamW/LR、accumulate=4 和全头 clip=35；差异是优化 loss 集。

M0 与 M0_fp32 的 dev t0 相差约 0.0000445 pp，远小于 O 相对 M0_fp32 的 −0.15914 pp；已经有同精度参考，不能把后者简单归于 TF32。

源码支持但尚待验证的机制是：跨时域共享 readout 在继续训练时改变了当前/未来判别的折中；移除 sem/geo 改变了共享参数的梯度组合及全头裁剪后的更新轨迹。训练代理损失、coarse GT 和完整 GT 的 IoU 也不完全同一目标。**尚未测量 t0 与各未来时域在共享读出上的梯度冲突，不能声称这是已证实原因；更不能声称 O 相对 C 牺牲当前性能。** 当前开发证据支持 O 相对同预算 C 的当前和未来指标都更好、相对原始 M0 存在当前/未来方向不同的变化；它不能单独确定机制。

以上引用的 detector、world_head_v1/base、semkitti_loss、Lovasz、memory_experiment、native_state_cache、objective trainer/adapters 和 S0 配置，均在本次逐文件重新计算 SHA，与父/目标协议和实际 config SHA 匹配。只读 dev 汇总的范围为 100 scenes/200 anchors；没有将该结论扩大到 full validation 或其他训练 seed。

[det477]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:477
[det498]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:498
[det577]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:577
[det588]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:588
[det554]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:554
[det534]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:534
[loss596]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:596
[eval718]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:718
[head97]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:97
[head170]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:170
[head199]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:199
[head224]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/dense_heads/world_head_v1.py:224
[cfg98]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py:98
[cfg124]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py:124
[cfg132]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/configs/S0.py:132
[replay243]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:243
[adapter17]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_adapters.py:17
[adapter65]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_adapters.py:65
[ce148]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/losses/semkitti_loss.py:148
[lovasz156]: /Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/losses/lovasz_softmax.py:156
[train233]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:233
[prep26]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:26
[prep42]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:42
[otrain223]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/objective_supervision_train.py:223
[eval68]: /Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:68
