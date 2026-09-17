# FIERY / BEVerse：运动标签与特征索引审查

2026-09-16。仅查两篇原论文及作者仓库，固定 commit 并保存所读源码/文件 SHA；没有执行外部模型、读取 K/B 中间成绩、修改实验或启动训练。结论是**它们把流的承载位置与消费方式明确配对；不能据此断言 K/B 同索引读出错误或已经失败**。

## 先区分三个问题

1. **ego 参考系**决定同一个空间点的坐标数值；自车坐标变换不等于物体运动。
2. **Eulerian / Lagrangian 索引**决定输出存在哪里。Eulerian 场在每时刻位置 `q` 编址；本项目材料点标签始终按 t0 源点 `p0` 编址，回归 `p_h−p0`，属于源点固定的位移读出。
3. **实例/材料点对应**决定跨时刻是否为同一对象/同一点。相同网格下标、共享参考系或仅预测语义占据都不提供这种对应。

本地 `future_state_motion_v1.py:31–50` 把 `features[0,j]` 与 `features[h,j]` 原索引拼接；没有按 GT 或预测未来位置取特征。P/`coordinate_contract_audit.md` 又表明 native 显式查询/记忆路由采用未来局部 SE(3) 约定，GT 位于 t0 系，可学习偏移可能补偿。因此只能称这些量为**原索引 hidden features**，不能先验认定它们已严格处于共同 R 系，更不能认定 `j` 代表同一材料点。此前 A 坐标路线的负结果不因这次检索而被撤销。

## FIERY：每个未来时刻的实例位置，预测下一步位移

固定作者仓库 `wayveai/fiery@fd03f164935ac172ef6d96117a7a4873abf23919`（2022-09-21）。论文 §3.6–3.7、§4.1 和附录 B.2 区分语义、中心、同帧中心 offset、跨帧 flow；各未来状态分别解码，论文目标参考系为当前 ego。监督来自车辆框，输出是二维实例预测，不是本项目三维材料点 scene flow。[原文 pp.5–6、14](https://arxiv.org/pdf/2104.10490#page=5)

实际源码链：

- `data.py:241–297,378–425`：每帧框先栅格化到该帧 ego 平面；以原 `instance_token` 保持序列身份；随后生成中心/offset/flow。[框与身份](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/data.py#L241)
- `utils/instance.py:12–77`：在相邻帧均存在同一实例时，把下一帧实例 mask 按 ego 变换到前一帧，取栅格质心差；**差值写在前一帧该实例的 mask 上**。是逐实例常量二维前向中心位移，单位格，不含旋转引起的材料点差异。缺帧会断开连接，生成器末帧没有 outgoing label，保留 255。[标签写入位置](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/utils/instance.py#L56)
- `trainer.py:144–184` 将各帧标签空间重采样到当前参考网格；`decoder.py:53–90` 对每个时刻状态分别输出同位置的 segmentation/center/offset/flow；`losses.py:20–37` 对非 ignore flow 做通道 L1 和、时间折扣、有效像素均值。[准备标签](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/trainer.py#L144)；[解码](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/models/decoder.py#L53)；[损失](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/losses.py#L20)
- `make_instance_id_temporally_consistent:199–266` 在**当前预测实例区域**平均 `grid+flow_t`，与下一时刻独立预测实例中心做距离/Hungarian 匹配；未匹配的下一帧实例可获新 ID。没有用旧 t0 mask 去直接读取所有未来 flow。此处是推理后的身份关联，不是训练中对未知实例做 Hungarian 损失。[消费路径](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/utils/instance.py#L199)

因此 FIERY 的空间一致性来自“时刻 j 的实例 mask / 状态 → j 到 j+1 的 flow → 下一帧实例关联”。它没有证明源点固定的多时域回归不可能学习，也没有给出本项目那种 t0 材料点四时域 EPE 保证。

## BEVerse：监督实例 flow 与内部特征 flow 是两条路径

固定作者仓库 `zhangyp15/BEVerse@5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be`（2022-07-29）。原文 §3.4 / Fig.3 / Table3 的 iterative flow 是从当前状态生成下一状态的特征 warp；最终仍接 FIERY 类实例输出头。其消融支持该架构在自己的 IoU/VPQ 任务有效，不能替代米制材料点验证。[原文 pp.4–5](https://arxiv.org/pdf/2205.09743#page=4)

| 路径 | 写入/采样索引与方向 | 监督及消费 |
|---|---|---|
| `instance_flow` | `instance.py:155–164` 仍在前一帧实例 mask 写下一帧质心减前一帧质心；行/列两分量，格单位 | `_base_motion_head.py:422–425` 对该输出头做 L1；随后同类实例关联 |
| 内部 `flow` | `motion_modules.py:434–447` 在**目标输出网格 q** 查询旧状态 `x(q+u(q))`；分量顺序为宽/高 | `offset_pred` 预测取样偏移，进入 warp→GRU→conv→任务头；未接上述 flow 标签的直接回归损失 |

具体链：[标签 helper](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/datasets/utils/instance.py#L87)；[实例损失与推理](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/models/motion_heads/_base_motion_head.py#L414)；[内部生成器](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/models/motion_modules.py#L198)；[取样器](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/models/motion_modules.py#L434)。

`beverse_tiny.py` 的 `IterativeFlow` 默认走 `using_v2=False, detach_state=True`；`ResFuturePrediction` 逐步 warp 后更新状态，再 detach 供下步，不能称所有未来梯度都通过整个 unroll 反传。最终 flow 头由各时刻状态单独解码，和内部 `offset_pred` 不是同一参数头。[实际配置](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/configs/beverse_tiny.py#L25)；[默认与解码](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/models/motion_heads/iterative_flow.py#L19)。

## 不应直接照搬的几何细节

两库的标签生成都先在局部帧计算向量，再通过普通 feature warp 移动两通道。检查 FIERY `geometry.py:219–220,256–280` 和 BEVerse `warper.py:42–73,111–132`：**该调用链只重采样空间位置，没有显式旋转两通道向量分量**。在一般非零 yaw 下，这不是完整的向量场坐标变换；正确变换还需相应的基旋转。这里是源码代数判断，未执行其模型，也没有量化对论文性能的影响。另有栅格质心取整、裁剪、框遮盖的误差；本项目已有 raw SE(3) 材料点合同不应换成这套近似。[FIERY 几何](https://github.com/wayveai/fiery/blob/fd03f164935ac172ef6d96117a7a4873abf23919/fiery/utils/geometry.py#L219)；[BEVerse 几何](https://github.com/zhangyp15/BEVerse/blob/5c6f4f7bb5f2c647e5c2301ef6e6b4e78f2253be/projects/mmdet3d_plugin/datasets/utils/warper.py#L42)

## 对本项目的启发与可证伪边界

- **当前 K/B 能检验的仍只是物理监督梯度接入原生 future head 的增量**。同索引拼接并非形式上非法：当前特征、D 初值、周围卷积与全局编码都可能保存运动信息；但若物体已移开，未来同索引状态的任务信息可能属于另一位置。K/B 差异不能单独区分这种读出困难、ego 隐式补偿、实例对应或 current-state 捷径。
- 若后续要检验“未来状态是否包含该源对象的信息”，最有针对性的是保持同一冻结特征/同容量读出/同源标签，比较**原索引查询**与**由输入可得的预测位移引导的未来查询**。必须单独写清未来 latent 的寻址约定；不能把物理 `p0+D` 无条件当 native token 坐标。GT 未来位置仅能作 train-only oracle 诊断，不能进入普通评价或宣称 deployable 收益。该比较不在本轮执行，也不等于重启旧 A 坐标补丁。
- 若选择 Eulerian 下一步流方案，应在每个 j 的目标位置生成 j→j+1 合法对应标签，再通过明确的物理积分/关联恢复源轨迹；不能继续在 t0 点索引上使用同名 loss、却把输出解释成未来位置局部流。物理 forward displacement 也不能直接当 `grid_sample` 的 backward sampling offset：纯平移时符号相反，非均匀场还涉及逆映射和多对一/空洞。
- “未来状态联合实例/flow 监督”“用 learned flow warp 生成未来特征”已有明确先例；BEVerse 还展示了任务取样偏移与最终运动头分离。我们的可研究问题是**监督对象、承载位置、特征查询和可部署对应的可识别连接**，不是把上述模块组合包装成首次提出。任何方向仍需真实物理 EPE、共同占据/动态风险及同容量控制结果支撑。

来源快照与逐文件 SHA 见 `eulerian_lagrangian_motion_sources_v1.json`；下载源码仅供阅读，没有安装依赖或运行仓库代码。
