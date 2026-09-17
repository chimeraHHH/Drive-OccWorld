# J/D 梯度连接设计窄审

状态：只读审查现有冻结 helper 与下一步设计；未运行 GPU、未训练、未修改冻结源码。新的 train_connected_motion_v1.py 尚未落盘，本文件不等于训练器通过代码审查。

**设计可回答的问题。** J/D 从同一预训练 motion、同一新 gate、同一 O 与缓存出发，唯一政策差异是占据损失能否反传到 motion；D 继续相同物理监督，绝不是零位移或冻结 motion。相同 seed11、样本序列、accum4、固定512updates、两个分离优化器和裁剪组足以构成当前预算下的连接政策对照。最终 J−D 包含联训过程中 motion/gate 的共同适应，不能唯一归因为物理 flow 更准。O/M0及原始 motion/A/Z 保留。

## 实质工程风险

1. **两处现成断路。** `train_supported_fusion_v2.shared` 整段 no_grad 且断言 transport 无梯度；只可复用其中冻结 O 的部分。`learned_transport_probe_v1.load_motion:147–149` 会冻结所有 motion 参数，加载后必须重新开启 J/D motion 的 requires_grad，并验证优化器恰好覆盖各自参数。O 仍固定、无优化器。motion 输入缓存 state 可以 detach；motion 输出 J 不可以。
2. **正确切断位置。** J 的 flow 原值进入 supported_transport；D 仅将该输入换为 flow.detach()，原 flow 留给 gather_sparse/object_group_loss。不能把 D 的全部 forward 放 no_grad，也不能把两臂 flow 都 detach 后依靠物理损失制造“非零总梯度”假象。compose_prediction 的 clone/切片赋值仍可保留 CopySlices 梯度；其 debug 整数字节比较不作为训练值。
3. **算子是局部可导，不是远程匹配器。** forward_splat_3d 的 floor/long 决定地址，fraction=destination−floor(destination)仍传递米制位移导数；eight-corner scatter numerator 保留图。支持W>0是硬门；完全出界、没有预测源支持、概率clamp饱和会造成真实零梯度。任务可通过挪走误报质量降低占据损失；保留界外质量、碰撞、覆盖和低运动漂移的现有记录，不能把这种收益称物理改进。
4. **物理归约保持原式。** beta=.5m 的逐XYZ SmoothL1均值→对象均值→当前存在速度组均值→有效时域均值。缺目标保留原图零项，不补造监督，也不跳过训练样本。未来出ROI仍受物理监督。固定两个loss权重均1不代表梯度规模相等；只记录局部范数比/内积，不据dev再调权重。
5. **常数loss与数值不能混淆。** 原O选6项CE/Lovász；前两decoder的4项固定，不产生新梯度。最后decoder的t0仍在5h pooled CE/Lovász中，不能因为t0不变就删掉它或拆成逐h另平均。保留原 loss_terms/select_objective 最安全；重复固定CE沿已验证容差审计、Lovász exact。不要为省常数归约而detach整个total或错误复用另一臂的最终层loss。sem/geo仍计算/检查但不加入objective。
6. **更新公平不等于共用状态。** 每臂独立motion与gate参数、AdamW moments、裁剪；每4样本仅一次step，motion LR1e−4→1e−5/clip10、gate1e−3→1e−4/clip35。不得先phys backward/step再occ backward/step；该操作与一次相加后裁剪/AdamW不同。任何初始VJP只读，不污染.grad、RNG或optimizer，也不继承预检四步权重。

## 一个足够有判别力的实际梯度合同

在预先固定的初始训练样本及相同参数状态，以 autograd.grad 分别取得物理、占据、两者之和对 motion 的未裁剪梯度，并记录：

- J/D 参数、flow、输入和物理target的字节身份相同；初始 gate 相同。传输/标量的浮点比较与源字节身份分别报告。
- J 的 `∂L_occ/∂flow` 与 `∂L_occ/∂theta_motion` 均有限且非零；D 对 motion 为全None（不把显式注入零grad当真正断开）。
- J/D 的物理梯度数值一致；各臂 `g_total≈g_phys+g_occ`；物理对gate全None，初始占据对gate的梯度数值一致；O无梯度。
- autograd.grad 后 .grad仍空，销毁诊断图并正常zero_grad，才开始真正accum4。若固定样本无支持使J为零，记录具体支持/clamp事实，不临时改GT或换样本掩饰。

不要求独立CUDA调用的所有浮点归约bitwise一致：transport scatter_add_ 与 physical loss 的 sums.index_add_ 均可能原子归约。需要实际记录误差，使用预先固定合理数值比较；不能把新的大差异自动归咎于1ULP。该初始合同只证明局部连接和配方执行，不证明训练全程梯度冲突或所有样本有效。原 primitive已有解析梯度/gradcheck测试，本轮无需再建重复测试架构。

## task-correction 与真实运动的界线

这里只窄核已有一手源，不新增泛搜。TOFlow原论文 **§4.1、PDF第4页/Fig.3，§4.3第5页** 明确给出背景非零位移改善遮挡补全的例子；作者 `WarpFlowNew:updateGradInput` 将采样梯度传回flow。这证明“任务好、位移偏离真实运动”在机制上可能，不给本项目因果结论。[论文](https://arxiv.org/pdf/1711.09078)，[固定作者代码](https://github.com/anchen1011/toflow/blob/941eae530199e646118cc5ec36b4a8896ebeb24b/src/util/nn/WarpFlowNew.lua#L159)；本地原代码SHA `7d44467979faa95203e1ed3bab8d237a63d6bdbb4da6794a3def684b46d1bf63`。

本项目最终应并列呈现 J/D/O 的原GMO、change/速度分组recall与全FP，及 J/D/预训练motion/zero 在**同一完整有效标签集合**上的3D/XY EPE、三速度组、四时域、mean/median/p90与scene配对差。若J占据提高但标签代理EPE变差，最多说明连接政策取得任务收益并牺牲了刚体位移代理准确度，可描述为task-correction倾向；不能说真实scene-flow更准，也不能仅凭负局部梯度cos确定整个下降机制。若两侧都提高，只支持当前监督支持上的双目标进步；新生/重叠/无唯一当前box、真实非刚体运动仍未被该EPE覆盖。O没有原生flow，不能把这些head EPE对照写成J胜过O的motion EPE。

可导搬运和J/D断梯度对照均已有先例；这一轮是回答现有方法信息连接问题的受控实验，不是独立ICLR新颖性证明。当前不叠加carrier改变、O解冻、梯度投影或dev系数搜索。

