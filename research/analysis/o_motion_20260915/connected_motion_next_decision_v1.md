# 运动监督与占据连接：下一项判别实验

状态：**研究提案，未实现、未训练、未派发**。2026-09-15。本轮只读已完成报告、冻结源码，并联网核验三篇原论文及作者代码；未读取正在运行的 supported A/Z 候选分数。其正式共同评价先完成，本提案不能替代它，也不构成目标已达成。

## 目前真正知道什么

- 已完成 source head 在 dev200／100scene 上，名义2秒较高运动组 XY EPE 从零位移7.509775降至4.358820m，但低运动组从0.054884升至0.202205m；两侧方向都跨场景出现。这是**当前唯一box内虚拟材料点上的刚体位移代理**，不是全场景flow，也不是与没有flow输出的O比较EPE。4.36m约为8.5个0.512m的粗网格，不能说运动已经足够准确。
- dense train16／8scene 中，learned half融合比O低0.79765pp，FP/FN分别增加8862/11025；同head反序时域再下降。此前 GT-oracle half比O高0.6813pp，只证明这些训练样本上的特权互补空间。二者还有GT安全支持与dense无监督背景的差异，不能把约1.48pp缺口全部归为位移误差。
- 当前 A/Z 训练在预测前景支持上，仅优化97参数门，O与motion均冻结。无论A/Z结果怎样，这一阶段**都没有测过占据梯度改善运动头**。也不能将静态源门Z解释为完整“仅辅助监督但motion继续训练”的对照。

依据：[source正式结果](source_motion_train_v1_结果与决策.md)、[dense传输结果](learned_transport_train16_结果与决策.md)、[oracle结果](oracle_transport_train16_结果与决策.md)、[既有连接提案](source_motion_connection_design_v1.md)。原GMO评价与低运动／过渡／较高运动分组全部保留。

## 为什么物理EPE下降仍可能不改善占据

**源码事实。** 当前 source 是 O 的二分类概率及当前argmax支持，传输只保留每个目的体素的两个标量：`S=Σ k·m0·p0`、`W=Σ k·m0`。S再裁到概率区间，门逐体素混合其log-odds与O未来log-odds；不是有独立形状生成能力的空间解码器。来源为 `supported_motion_fusion.py:14–46,80–103`；`transport_ops.py` 使用三线性前向散射，碰撞求和、界外权重丢弃。

**算子层面的推论，不是新的实验结果。** 体素占据是某个空间区域是否被占据的事件，通常不等于可守恒相加的概率质量。孤立源体素若p=1、沿XYZ各移动半格，三线性散射给八个邻居各1/8；即便中心位移完全正确，单独以0.5阈值读取就可能全部消失。反过来，多源碰撞的概率相加再clip可能饱和。这个解析例子解释表示风险，**不证明我们实际失败主要由它造成**，也不授权现在改归一化／阈值。

此外，点EPE评价只覆盖有当前唯一box点和未来标注的对象；实际传输还会移动O的误报与无物理监督点，漏掉当前未检出的物体形状，并且不能单凭t0质量产生未来新生物体。门保留O补全路径，却不能自动识别所有错误覆盖。低运动对象0.20m的抖动接近0.4个粗格，也可能损伤离散形状边界。

**跨目标推论。** 稀疏物理损失让对象／速度组等权；占据CE和Lovász作用于另一个空间、有效支持和归约。物理点均值更低不保证高代价边界、漏检对象和不受监督的源点更好。当前没有同参数处两项梯度的证据，不能先称“梯度冲突”。即使接通，散射对位移的导数主要感受落点附近有限邻居；clamp饱和、W=0回退和关门会削弱通路。错误落点附近的占据梯度不保证指向远处正确对象，也可能奖励把误报质量移出ROI。

## 三项新增一手核验及其适用边界

### FipTR，ECCV 2024：有监督flow真正进入未来特征，但采样offset不是物理flow本身

[原论文§3.3–3.6](https://arxiv.org/html/2404.12867v2)与[作者仓库](https://github.com/TabGuigui/FipTR)一致地把预测backward flow送入未来BEV注意力。实际 `FlowGuidedSelfAttentionV2` 将query与flow拼接后用Linear生成sampling offsets；不是直接强制按flow采样。head中flow→未来特征→实例mask这段没有detach，另有SmoothL1监督。因此它提供“共享特征生成与任务梯度接通”的先例，不证明采样offset等于真实运动。[attention:239–280](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/modules/flow_guided_self_attention.py#L239)、[head:257–294](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/dense_heads/fistr_lss_head_timespecificmaskquery.py#L257)。

实现边界很具体：当前官方配置选 `ConvertMotionLabelsFistr255`，标签是实例BEV栅格中心的相邻帧平移，写在目标实例mask；背景255在flow loss中另监督为零。它不是source-centric三维刚体材料点，不能将我们的米制forward flow直接代入。可复用的是“flow到采样的可导接口”及其对照思想，不能直接搬标签、初始化权重或论文分数。[labels:403–448](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/pipelines/motion_labels.py#L403)、[loss:720–739](https://github.com/TabGuigui/FipTR/blob/3f8b59ab76d444d1b8929b60ea757fb9119db012/projects/mmdet3d_plugin/fiptr/dense_heads/fistr_lss_head_timespecificmaskquery.py#L720)。这条链仅静态审读，未导入／运行它的CUDA算子。

### Softmax Splatting，CVPR 2020：可导传输并不要求把源概率直接当目的预测

[原论文§3](https://arxiv.org/pdf/2003.05534)在散射后使用特征金字塔和synthesis网络，并研究多源落在同一位置的权重选择；这与直接sum占据概率再clip是不同的表示。作者算子明确提供对输入及flow的backward；`soft`模式拼接加权分子与指数权重，散射后除以分母。[算子:232–270,362以后](https://github.com/sniklaus/softmax-splatting/blob/43df87a387440446388f8aec72976c5e7a4a3996/softsplat.py#L232)。

作者完整推理链实际先encode、估计soft metric，再warp多尺度特征后合成图像。[run.py:471–481](https://github.com/sniklaus/softmax-splatting/blob/43df87a387440446388f8aec72976c5e7a4a3996/run.py#L471)。但该发布脚本顶层关闭grad，是推理实现，不能当已审核的完整训练脚本；二维图像插值拥有两侧图像，也不同于我们的未来预测。它支持研究传输表示／碰撞问题，**不说明softmax归一化就是三维占据正确的概率模型**。仓库声明限学术使用；本轮没有安装CuPy或复制CUDA算子进项目。

### TOFlow，IJCV 2019：任务梯度可能学出有用但非物理的位移

[原论文](https://arxiv.org/pdf/1711.09078)明确讨论增强任务更好而flow偏离物体真实运动的情况；这正是我们不能把GMO提升当EPE提升的理由。[作者项目页](https://data.csail.mit.edu/tofu/)提供原Torch7仓库。实际 `WarpFlowNew:updateGradInput` 把采样网格梯度传回flow；README发布范围是权重与demo，并非当前PyTorch可直接运行的训练器。[作者代码:145–167](https://github.com/anchen1011/toflow/blob/941eae530199e646118cc5ec36b4a8896ebeb24b/src/util/nn/WarpFlowNew.lua#L145)。可复用的是“同flow架构，隔离下游梯度”的实验逻辑，不复制其图像任务评价或放弃我们的物理监督。

以上三个版本的核心文件已按commit只读保存到 `connected_motion_source_reads/`，文件SHA与原始URL见其中 `sources.json`。未下载权重或训练。另核了DFIT-OccWorld原论文，但没有确认可用作者核心仓库，故不把它算作已核验实现，也不据其摘要提出新分支。

## 唯一建议的下一实验：同运动监督的 J／D 梯度连接对照

**问题：保持当前source表示不变，允许占据任务共同塑造位移，能否获得既更有用、又不更不物理的运动？** 这比同时换feature carrier、改碰撞归一化和解冻O更可判别。它也不保证解决概率表示上限。

| 项目 | J：接通 | D：切断 |
|---|---|---|
| 初始motion | 相同已完成512步head | 完全相同 |
| motion训练 | 原box物理监督＋占据梯度 | **原box物理监督继续训练** |
| 送入同一transport的值 | `d=motion(z0)` | `d.detach()`，数值不改 |
| fusion门／O／source | 相同门初始化；O冻结；同p0、当前argmax、坐标 | 完全相同 |
| gate训练 | 原O的CE+Lovász | 完全相同 |
| 预算／评价 | seed11，同512train×4pass、acc4、固定512final；同dev200 | 完全相同 |

两臂都从原motion最终权重与**同一新门初始化**开始，不拿A与Z的不同最终门作为各自初值；A/Z只保留为已完成参考。D不是冻结motion，也不是置零flow。唯一有意差异是 `L_occ→d→motion parameters` 是否接通；不加direct-flow快捷特征，不改GT、source支持、概率clip、原六项loss或t0。分组motion loss原归约不变，future GT只能进入监督。两臂的motion／gate参数分别使用同配置优化器和同裁剪分组，避免gate梯度被全局混合裁剪后间接改变D的motion更新。本文件不选择或调优联合loss系数；若决定实施，先固定一组共同系数与学习率，不做开发集系数搜索。

**工程与解释共用的最小观测。** 初始相同样本J/D前向值应一致；D的占据到motion梯度必须为零／不参与，J应观测到有限且非零的实际梯度，而非只有flow辅助梯度。只在预先固定的少量train样本、同一参数状态记录两目标的梯度范数／内积与clamp、support、gate状态；这说明该状态的局部连接，不等于整个训练中的梯度冲突证明。继续保留质量界外／碰撞统计，防止把移走误报当运动变准。两条真正训练路径各自更新，不在同一模型上交替共享optimizer或moments。

**主要结果仍是原任务。** 固定四未来GMO、每h、t0、FP/FN；J−D判断任务梯度连接政策，J−O判断实际占据收益。物理侧同时报告原支持上XY/3D EPE的三个组、均值／中位／p90以及静止代价，比较J、D与原motion head／zero。用同100scene配对区间，注明单训练seed、历史验证曝光。O没有flow，因此这些物理对照不能冒充“J的EPE优于O”；占据更好且头EPE更好，仍未自动补上与O共同输出的运动准确性证据。

## 取决于 A/Z 的前提与明确负结论

1. **先等 A/Z 完整同集结果。** 若A胜Z且胜O，冻结分离训练已经证明一类可用连接，J/D检验的是联训的额外价值；若A胜Z但不胜O，只证明比静态补充分支有用；若A不胜Z，不能先宣布表示失效，J/D仍可区分当前独立辅助训练是否限制连接。现阶段不派发这对新训练。
2. **J无实际占据梯度或全部被门／clamp阻断：** 工程记录该通路未形成，不把未接通的负结果解释为联合监督无效，不自动延长预算。
3. **J不优于D，或依然不优于O：** 本预算下不支持“接通下游梯度解决问题”；停止扩展这条概率搬运路线，不靠换阈值、GT支持掩码、选择时域或更多seed救结论。
4. **J的GMO提高但物理EPE恶化，尤其低运动进一步漂移：** 最多证明任务纠错场有用，不能称物理运动更准，用户的联合目标仍未完成。反过来只有EPE提高而GMO不改善，也只是辅助目标进步。
5. **J相对D/O的GMO与两侧物理风险均改善：** 支持当前条件下任务连接有效，仍需共同输出运动证据及后续固定评价；不是新增ICLR创新性自动证明。

若概率表示的可达输出确实太窄，**feature transport必须被列为下一次独立科学因素**：固定motion及任务梯度政策，才比较carrier/decoder，不能将representation、flow联训、O解冻同时改变。原M0/O readout在 `world_head_v1.py:98–121,162–193` 将每个2D BEV token映射为高度×类别logits；没有一个已经认证为每个XYZ体素的通用latent可直接搬。将BEV特征随意复制到16个高度不是等价三维物理传输。还须保持固定t0输出空间，不再以未来ego变换修正O内部递推。当前不实现这个替代模型；本轮先用J/D回答一个清楚的问题。
