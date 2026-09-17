# 历史相机冻结光流：先检查对应假设，再决定是否使用 RAFT

访问日期：2026-09-16。范围仅原论文、作者官方源码与本地已完成统计；未安装依赖、下载权重、运行模型或修改实验程序。本文件是后续候选备忘录，不是已冻结实验。

## 当前证据与优先决策

真实 train512、D-seen、training-internal 的 2 s、当前预测框未覆盖且无同 cell radar 支持人口中，同一合法人口的 AUC 为：ZNCC true **0.5112885325**、broken **0.4964052439**、原 D LSQ speed **0.6008099493**。合法图像证据触达该处完整原分母负代价的 **85.6663%**。因此当前失败不能仅归因于图像没有触达风险人口，也不能据此训练 gate。以上直接读取 [已完成分析](../history_camera_evidence_train_analysis_v1.json)，SHA256 `20b49f30bab4ae8c539e385c288bb8e355feb587cabfdd3327423dc68b2905bd`；位置 `horizons[3].strata.no_radar.all.bins[0]`，不是独立重新汇总。

**优先做明确隔离的历史 GT 刚体姿态 oracle reference/control，再决定 RAFT 是否值得实际试验。** 在有历史同对象标注的原材料点上，用历史刚体变换替换 D 四时域 LSQ 速度向过去的恒速延拓，保持相机、原图、patch、统计权重和当前点身份；与 D 延拓和原 broken 对照在共同合法人口上比较观测残差，同时保留全部原物理分母及历史标注缺失量。标注只属于这个离线诊断分支，绝不进入可部署特征、拟合或 gate。相机异步时刻的姿态插值应单独说明；没有历史标注时间支撑的情形记录 missing，不能暗用未来框补足。

这不是数学 upper bound：GT 框内材料点也未必位于可见表面，且 ZNCC 不随姿态误差单调变化。若 oracle 仍差，只能说明当前投影加固定 patch 的观测假设不够，不能否定相机运动信息；若 oracle 明显更好，则支持时间延拓或 D 运动假设是重要因素，仍不是唯一因果解释，也不保证识别未来 D 优劣的 AUC 接近 1。RAFT 可能改善外观对应，但不能自动修复这两个几何与时间问题。

## 官方 RAFT 的真实接口

原论文为 Teed 与 Deng，ECCV 2020，[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/html/2003.12039v3)。它从两图特征的相关性与学习到的更新规则估计像素位移；论文也明确讨论遮挡、快速运动、模糊和无纹理困难。这里不沿用其 benchmark 分数或吞吐量推测 nuScenes 表现。

作者仓库 [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) 当前读取 commit 为 **`2888e15a51fa41140771d3f498ed8023cff098d1`**（GitHub API 返回提交时间 2025-08-24T23:34:28Z）。下列行号均按原始源码的一基行号。

| 项目 | 已核官方实现与本任务所需处理 |
|---|---|
| 权重来源 | [README L19–27](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/README.md#L19) 指向下载脚本及官方 Google Drive；[脚本 L2–3](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/download_models.sh#L2) 使用 `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`。demo 使用 `raft-things.pth`；[训练脚本 L3–4](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/train_standard.sh#L3) 给出 Chairs→Things 路径。仅据源码不能验证下载包内该权重的实际训练履历。 |
| 权重认证边界 | 已读 README、下载脚本及完整源码树，未发现发布权重的官方 checksum。未请求权重包，链接实际可下载性、包内容、文件大小及权重 SHA 均未知。后续若获准获取，应记录官方链接、重定向、时间、archive/pth 的字节数和 SHA；自己计算的 SHA 是重现指纹，不是官方独立签名。 |
| 输入 | [demo L20–23](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/demo.py#L20) 读 uint8 图像后转 float CHW；[raft.py L89–90](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/raft.py#L89) 内部将 0–255 映射到 −1–1。适配器应显式使用原始 RGB，不用 native BGR 减均值后的 tensor，不先除 255 再交官方 forward。 |
| 尺寸 | [InputPadder L7–24](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/utils/utils.py#L7) 补到 8 的倍数；默认模式下 900×1600 变为 904×1600，上下各复制 2 行。输出必须 `unpad` 回原图；不沿用 CRN 的 resize/crop，也不直接把 padded flow 当原图坐标。 |
| 输出方向及单位 | [raft.py L63–83、116–142](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/raft.py#L63) 给出第一图网格上的 `coords1−coords0`；第二个返回值是全分辨率 `(dx,dy)`，单位为像素，已含 ×8 的上采样比例。[coords_grid L74–77](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/utils/utils.py#L74) 使用 x、y 通道顺序。反向光流需交换图像重新预测，不能直接取负。 |
| 冻结运行配置 | 可提出单一官方 `raft-things`、full RAFT、20 iterations、`eval/no_grad`、`flow_init=None` 的候选，遵循 [demo L43–62](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/demo.py#L43)。20 来自 demo，不是所有官方评测的通用设置；forward 默认是 12。任何实际配置和 strict checkpoint 匹配应先冻结，不能看本任务标签后选权重、次数或尺寸。 |

官方测试环境是旧版 PyTorch/CUDA，当前环境兼容性尚未实际验证。[CorrBlock L12–27、53–60](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/core/corr.py#L12) 的 float32 全相关金字塔，在 904×1600、batch 1 时仅四层相关张量按源码尺寸估算约 **2.52 GiB**：`22600 × (22600+5600+1400+350) × 4 bytes`；不是峰值显存或运行预算。后续先做不看标签的单对资源检查，再固定上限；六相机及双向顺序运行，不据此声称现有显存必够。此次未验证自定义 alternate-correlation 扩展，不默认改实现或缩图解决资源问题。

## 若继续，如何构造 D 与 zero 的观测一致性

复用已认证的原始同相机 current/previous keyframe、实际时间戳和 `camera_to_global`，不使用 `sample_data.prev` 代替已认证上一关键帧，不用文件名排序配对。相机固定为当前 zero 投影选出的原相机，不能由 D、光流质量或 GT 选相机。[现有几何接口](../history_camera_evidence_v1.py) 的 `nominal_lsq_velocity`、`project` 已明确四个名义未来时域与异步相机约定；其源码 SHA256 为 `c7469179d1628979b30eb256e695fdd6479ffd6e2df047d380d890512f57a5ea`。

令 `G0` 为 t0 LiDAR 到 global，`Cτ` 为对应相机到 global，`p0` 为 R 系材料点；沿用全部四个名义 h：

```text
vD = sum_h(h * D_h) / sum_h(h²),  vZ = 0
u_a(τ) = project(Kτ, inverse(Cτ) * G0 * [p0 + (τ−t0)*v_a; 1])
Fcp = RAFT(current_RGB, past_RGB),  Fpc = RAFT(past_RGB, current_RGB)
e_a = norm(u_a(current) + Fcp(u_a(current)) − u_a(past))
score = e_Z − e_D                 # 固定正方向：越大越支持 D
```

`Fcp` 由图像先产生，不能用 D 作为初始化；D 只决定待检验假设的投影地址。双线性查询保持原像素坐标约定（官方 sampler 使用 `align_corners=True`），两端都须有限且位于真实图像内。异步 current 相机下 D 和 zero 的 current 投影也可能不同，不能静默共用其中一个。zero 指世界静止假设：自车运动仍会造成非零图像位移，不能以光流接近零判静止。0.4–0.6 s 必须用逐对真实时间；光流是这对图的位移，此像素残差无需除以固定 0.5。

外加 forward–backward 残差可以是：

```text
rFB(u) = norm(Fcp(u) + Fpc(u + Fcp(u)))
```

这不是 RAFT 返回的 confidence/遮挡头；已核 forward 只返回光流。第一次诊断可保留连续 `rFB` 和出界/missing 计数，不凭本任务标签调阈值。若以后用遮挡阈值，需先冻结规则，D/zero 比较使用共同合法人口，并保留原完整物理分母。双向一致也可能同时匹配到错误的重复纹理，不能证明可见性。若沿用原“过去图横移”的 broken control，须对该破坏后的图像对独立运行对应光流；事后平移正常 flow 不等价于在破坏图上推理。

评价继续使用原 source/object 权重、四时域、GT group、七固定速度 bins，以及无 radar 人口的完整分母收益/负代价触达量；与同人口 D speed 和几何分离控制比较，不仅报有利 AUC。RAFT 也使用 D 已消费过的相机观测：它提供另一种冻结对应估计及外部学习先验，**不是统计独立真值**。任何改善首先只是 training-internal 的观测判别证据，不是新模型 motion gain 或泛化结果。

## 不能由光流补掉的边界；与 CMFlow 的区别

- **材料点与可见像素不等同。** 原虚拟体积内部点投影可能落到遮挡它的前景表面；准确光流也会跟踪那个表面。没有可见性/深度证据，不能把像素残差称作该材料点的可靠物理误差，也不能把低 forward–backward 残差当表面认证。上述 oracle 同样受限。
- **过去一致性不等于未来正确。** 四时域 LSQ 向过去的恒速延拓抹去了加速/转弯；真实未来 D 也可能对应错误的过去假设。反过来，过去 flow 一致的恒速运动不保证未来持续。相隔约半秒、出画/重现、模糊和域差异都仍在，不能用论文短时相邻帧成绩保证本任务。
- **CMFlow 的先例是跨模态训练监督。** [原论文](https://arxiv.org/html/2303.00462v1) 和固定 commit `16a095a250453b4fe1363e4ec9dddd20d0d64e4f` 的 [预处理 L91–102](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/preprocess/utils/get_flow_samples.py#L91) 仅在 train 分支提取 RAFT 光流；[OpticalFlowLoss L207–240](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/losses/radar_loss.py#L207) 将运动后雷达点与光流终点射线比较，且使用训练 mask；[模型 L171–194](https://github.com/Toytiny/CMFlow/blob/16a095a250453b4fe1363e4ec9dddd20d0d64e4f/models/cmflow.py#L171) 推理换为预测 mask。我们的候选直接在诊断推理时消费历史图像，不移植它的 LiDAR 跟踪伪标签/训练 mask，也不因此具备它的监督条件。共享图像对应约束本身已有先例。

## 可复核来源指纹

下列为本轮读取的官方小源码 SHA256；没有保存或请求任何权重。RAFT raw URL 均为 `https://raw.githubusercontent.com/princeton-vl/RAFT/2888e15a51fa41140771d3f498ed8023cff098d1/<path>`，CMFlow URL 同理使用上述作者仓库及固定 commit。原文与上述公开文本可读取；权重端点和实际运行能力未验证。

| RAFT path | SHA256 |
|---|---|
| README.md | `64fe19a2eafcd59075b0e711c8c5e5436ffb20339b302223e6a7954ce74c60d3` |
| download_models.sh | `f5f4b96be22b4743a8f613fcc6a8683ef57fe73f61bf9548028ca1f90dc28b00` |
| demo.py | `23b811268946823705f212bcf6e98cc58989db636fb192703d6b1869730b8e49` |
| core/raft.py | `e7280b82d0e224eff760ae8de5e8ea68923a7928c1e34920378b6f3a039a8633` |
| core/utils/utils.py | `125ca33c84f60c19359c875002e9301fe0404b3256c6efec718b26589d7fdd5a` |
| core/corr.py | `66f85489672e8fbf531135b53cda710594cfd1d900e855200e2bf6268dd7264a` |
| train_standard.sh | `4bdcb24c7c0241321665494e91bad98967ae75ad4fd91244a9122f0df312f4d0` |

CMFlow 三个文件本轮再次读取并核 SHA：`preprocess/utils/get_flow_samples.py` = `e10bd5b6a12bfb615121cb2d9b18ec2382a719588ea47b4f8a456fd8b0257a3c`；`losses/radar_loss.py` = `85a23998c6bd8d042c7fbd269dd1a54d9aee71d135a636fecfc3ac9edc2ff3d2`；`models/cmflow.py` = `cb59b982bd107fbdc5d7f643bafb309762fc1e4000b6034da250a2f3d4b3b576`。

## Addendum：先做不重算图像的标签连接诊断（2026-09-16）

**本补充将上文的优先步骤收窄为复用现有 ZNCC 分数、只增加历史标签关联的诊断；不实施历史 GT 位置上的新 patch 采样。** 因而它能分别检查“旧观测分数与过去代理收益”“过去代理收益与未来收益”的关联，不能称作“真实历史对应下重新评价 ZNCC”，也不是已完成结果。

在 `A0/Aprev` 均为同一 GT instance 的对象局部坐标到 global 的几何中心刚体矩阵、`G0` 为当前 LiDAR 到 global、`p` 为原 R 系齐次材料点时：

```text
H(p) = xyz(inv(G0) Aprev inv(A0) G0 [p;1]) − p
dt_prev = (previous_box_sample_timestamp − current_box_sample_timestamp) / 1e6 < 0
Dpast = dt_prev * frozen_four_horizon_LSQ(D)
Bpast = ||Hxy|| − ||Dpast_xy − Hxy||
```

这个 H 是当前 R 坐标系中材料点的历史位移，零参考为世界静止，不应再减一次自车平移。箱子尺寸不进入刚体矩阵；不能把前后尺寸变化当缩放。原 raw 生成器已分别保存 `timestamp_us`、`lidar_timestamp_us` 和差值，current/previous 的 sequence index 是 2/1（[生成器](../build_motion_targets_v1.py)，L210–221）。因此这里用与箱子姿态对应的 sample 时间差，并保留与旧相机投影时刻的差异；不插值时，“约 50 ms”只能作为已知量级解释，具体偏差应从已有时间戳记账，不能当所有样本的固定修正。H 仍是两箱刚体代理，不是 observed scene flow。

支持应显式区分 `U_h = 原 valid[h] ∩ owner<0 ∩ no_radar`、`L_h = U_h ∩ 历史姿态可认证`、`C_h = L_h ∩ 原 photometric valid`。GT 身份由原 source 的 `object_index→instance_token` 连接，不能用 CRN owner 代替。历史连接只认证 current/previous 同对象及所需直接链；不能补 missing、取最近箱，也不能额外要求该对象七帧全在来删减原 h 的合法支持。原对象点权重和完整对象分母不变，缺历史和缺图像证据分别列账。

| 比较 | 合法解释与限制 |
|---|---|
| 在 C_h 上，原 true/broken/speed 对 `Bpast>0` 的 AUC | 比较旧观测分数是否能排序 D 的过去刚体代理收益；方向不调。若仍弱，不能单独判定图像没有信息：材料点不可见、箱子代理与相机时刻不一致、固定 patch 均未排除。 |
| Bpast 对原 `Bfuture_h>0` 的 AUC | 是使用历史 GT 的 oracle reference，仅检查标签连接。可在 L_h 报完整历史可评分结果，并在 C_h 同人口报配对结果；两者不可混着解释。应在相同人口重报既有 speed 对照，避免把“过去 reference 的子集 AUC”与原总体 speed AUC 比。 |
| past/future 正负四格及零值、missing | 给出原权重占比、未来正收益/负代价的完整分母贡献；正负四格加五个含零的格可形成互斥 3×3，再加历史 missing，精确重构 U_h。不能以共同合法人口重新归一化物理收益。 |

各 AUC 延续旧 helper 的 `benefit>0` 对 `benefit<0` 定义，恰零只单独记账，不静默并入负类；任一类无正权重时返回未定义，不填 0.5。

**关键混杂是两种收益共享 D。** 例如历史 H 为零时 `Bpast=−||Dpast_xy||`；若未来也静止，未来收益同样由 D 幅值决定。于是 Bpast 的高 AUC 可能部分来自共同 D 幅值和动静构成，并非独立证明过去真实运动决定未来可靠性。已有 speed 对照及既定速度 bins 可帮助解释，但不能消除所有混杂。相机分数本身也在 D 生成的投影处查询，broken 是有用控制而非完美随机化。

若第一个关联有用而第二个弱，支持“目前过去代理判别难传到未来目标”的解释；若第一个弱而第二个有用，支持优先检查观测连接；两者都弱则不能定位唯一瓶颈。以上都是同支持上的描述性诊断，不作因果中介分解、不据此宣称 RAFT 必然有效。四个 h 复用同一个 H、LSQ 和 Bpast，变化主要来自未来标签与合法支持，不能当四份独立历史证据或独立显著性。所有结果仍是 D-seen train512 内部分析，历史 GT 只在后评分分支，不进入部署输入、拟合、阈值选择或当前 frozen 程序。
