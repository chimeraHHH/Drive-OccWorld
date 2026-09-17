# 冻结视觉特征上的世界模型：对当前路线的补充与反证

2026-09-17。根 agent 原生检索与代码阅读；没有使用 skill、训练模型或下载大权重。本报告补充既有 17 篇调查。正式会议论文、期刊文章和预印本分别标注。代码快照与 SHA 见 `world_model_followup_repo_audit_20260917/root_source_manifest.json`。

## 1. DINO-Foresight：不依赖显式对象运输的有效对照

**正式状态：NeurIPS 2025 主会。** [正式 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/efaca9631eb6bd5df8f6761c5d9a9fe3-Abstract-Conference.html)，[正文 v2](https://arxiv.org/html/2412.11673v2)，[官方仓库](https://github.com/Sta8is/DINO-Foresight)。

冻结视觉编码器的多层 patch 特征，压缩后预测未来特征，再用任务头解码语义、深度等。训练可用未来帧作为目标；预测端屏蔽未来 token，长时域把自己的预测送回上下文。它提供了“未来表征→任务”的有效路线，但没有证明真实三维物体运动或本项目的 GMO 占据改善。正文明确使用确定性预测；一些几何评价是伪标注，不能当独立三维物理真值。

实读 commit `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`：

- [`src/dino_f.py`](https://github.com/Sta8is/DINO-Foresight/blob/d98f765d2b743ec90fb38bc657a39c07c7ca4ab0/src/dino_f.py)：`extract_features` 使用 `no_grad`；`get_mask_tokens` 的 `full_mask` 替换未来 token；`forward_loss` 仅在 masked 位置计损失；`sample_unroll` 将预测特征接入下一轮；`evaluation_step` 把 `pred_feats` 送进 head。
- README 的公开训练命令是 8 GPU、低分辨率 800 epochs 后高分辨率适配，并不是廉价的小 adapter。核到公开权重入口，未下载、未验证文件或在本环境推理。

**对我们的推断：** 不应预先断言“没有显式物理状态就无法预测未来占据”。可以把密集未来特征预测作为强竞争路线，判断对象/表面绑定是否确实值得付出额外复杂度。反过来，好的 2D 未来语义也不能证明其物理运动更准。复制它的 SmoothL1、PCA 或 Transformer 本身没有新颖性。

## 2. DINO-WM：读出好看与动力学学好应分开

**本轮未核实正式会议录用，按预印本补充。** [论文](https://arxiv.org/abs/2411.04983)，[作者项目页](https://dino-wm.github.io/)，[官方仓库](https://github.com/gaoyuezhou/dino_wm)。不把旧版 ICLR 投稿 PDF 当成录用证据。

它学习动作条件下的冻结空间视觉特征预测，用特征距离做规划。与我们的差别是：自车计划只控制自车，不能给其他交通参与者提供已知未来动作。其规划成功也不等价于材料点 EPE 或三维占据正确。

实读 commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c` 的 [`models/visual_world_model.py`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py)：`forward` 的视觉/本体特征目标 detach；图像 decoder 接 `z_pred.detach()` 或 `z.detach()`，重建梯度不负责训练动力学；动作维不作为相同 embedding loss 的预测目标。这是可检查的模块边界，不是“只要 detach 就更好”的普适结论。

**对我们的推断：** 运动 probe、占据 decoder 和世界状态转移应分别评价。尤其要避免占据 decoder 通过静态语义和 O 残差通路获得收益，而运动部分仍无实用贡献。也不能仅凭结构上存在梯度就宣称运动已被使用。

## 3. JEPA-WMs 系统研究：训练 rollout 与最终用途必须对齐

**TMLR 2026，非 ICLR 主会。** 作者 [arXiv v4 摘要及版本说明](https://arxiv.org/abs/2512.24497v4) 明确记录 TMLR 接收；阅读包括 [正文 v2](https://arxiv.org/html/2512.24497v2)，并用 [v4 的 §3–4 与 Table 1](https://arxiv.org/html/2512.24497v4) 复核训练和 rollout 说明，未把新增理论附录计为深读。[官方仓库](https://github.com/facebookresearch/jepa-wms)。

该工作系统比较表示、条件输入、训练与规划配置，表明设计收益依赖任务。它不能给我们一个统一的最优 rollout 步数或 loss；有用的是把训练时的预测误差与最终使用效果分开，并检查模型面对自己生成的状态时是否仍有效。

实读 commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`：[`app/vjepa_wm/video_wm.py`](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/video_wm.py) 的 `rollout` 支持 sequential/parallel、自身预测上下文、可选 stop-gradient 和 scheduled sampling；[`train.py`](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/vjepa_wm/train.py) 实际汇总 rollout loss。代码也有 `use_ground_truth` 本体状态选项，不能把库中所有模式都当纯开环预测。README 提供模型入口；本次没有运行。

**对我们的推断：** 未来四个时域都给监督，仍不等于共享状态在无新观测条件下保持一致；但我们的现有四时域预测也不能未经检验就硬改成递归。直接多时域预测是必要竞争对照，递归累积误差是实质风险。

## 4. 应改变的研究判断

最新 RAFT 诊断弱，只否定当前虚拟点投影评分作为全局路由证据，不否定相机运动信息。已有 V 输入速度重分配几乎不影响终端，已有 Cpl/Fix 又表明“同一刚体状态连接两个任务”本身不足以成功。因此下一步需要同时研究 **状态承载什么、观测如何更新它、未来任务究竟使用什么**；不能把“增加 flow loss”或“接上 warp”当完整研究问题。

我建议保留两个竞争假设，而不是立即叠加全部模块：

| 假设 | 可以解释的现象 | 能推翻它的证据 |
|---|---|---|
| 观测归属与时间持续性是主要限制 | 虚拟体积点无法对应可见表面；速度被错误实体读取或丢失 | 在合法、可靠对应下仍无运动增量；只靠更新未来特征目标即可获得同等联合收益 |
| 未来状态的学习目标与读出是主要限制 | 同权重速度扰动没有实用输出效应；当前占据源稀疏 | 改善未来表征训练后仍不改善运动/占据；而只修正观测绑定就明显有效 |

这两个因素可以共存。需要先比较各自作用，再决定是否合并；不能把弱 AUC 唯一归因于几何，也不能把最终 IoU 小幅提升唯一归因于运动。

## 5. 对下一实验的具体启发（建议，未启动）

在下一模型原型前先做**共享读出下的未来状态可用性检查**：冻结同一个合法 encoder/readout，比较当前状态直接外推、现有预测未来状态、由未来观测提取并严格变换到相同坐标的 teacher 状态。第三者仅是特权诊断/训练目标，绝不能进入推理。所有来源必须保持相同时间、坐标、通道语义；接口不等价就不能直接替换。teacher 并非数学上界，也可能受感知误差限制。

若 teacher 经同一读出明显更好、预测状态差距大，值得学习“可被当前任务读取的未来状态”；若 teacher 也弱，优先解决读出/几何，而不是强制拟合不适用的目标。还要报告 ego 对齐后的静止组与运动组，避免把自车视角改变当物体运动。这个检验补充已有 GT 位移 oracle：它改变的是表征来源，不能冒充同一种运动因果干预。

后续学习仍保留 O/M0、单 seed11；未来特征仅训练目标。先 profile 和检查拟合，再锁定预算，不能把任意短训失败当路线失败。晋级必须同时报告完整未来 GMO、移动召回、原支持上的 moving/static EPE，并面对全部八个速度路由对照；开发集已多轮曝光，最终需要另行审计未参与设计的数据。这里提出的是研究设计，没有产生超过 O 的新结果。
