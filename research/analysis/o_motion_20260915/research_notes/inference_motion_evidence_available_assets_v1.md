# 推理时运动证据：现有资产与需要重新提取的边界

2026-09-16。本轮只读本地源码、缓存索引、完成回执和已经镜像的 CRN 预测；没有 SSH、模型前向、Torch 导入、训练或修改冻结提取任务。这里的“可用”区分已完成远端资产的回执、实际本地文件和仅由源码支持的提取能力；不把未来标签或邻帧 GT 当输入。

**最直接可用的额外观测入口是原 `inputs.pt` 内独立保存的 `radar_bev`，无需重跑 encoder。它仍是 D 已间接消费过的观测，而非新传感器或统计独立证据。** 原缓存没有保留多时刻 BEV 序列；CRN 的预测 JSON 也没有逐扫雷达统计。更细的时间证据存在于另一层 CRN 输入资产，但尚未成为与全部 train512/dev200 对齐的可靠性特征成品。

## 1. 已存在什么，实际还缺什么

| 资产／API | 已保存的内容 | 读取边界与是否需要新模型前向 |
|---|---|---|
| native train512/dev200 `inputs.pt` | 当前 fused `prev_bev_input=[1,1,40000,256]`、独立 `radar_bev`、当前／历史 metadata、原协议给定 ego/action 等 future_pred 参数 | 可认证后只读输入；没有原图、各历史时刻 BEV 或逐返回 radar。读取雷达不需要 native 前向。当前本地镜像只含 index/complete，没有 712 份重型 `inputs.pt`；本轮未重新打开远端 tensor。 |
| 同目录 `native_preds.npy` | **M0** 的 `[5,3,1,1,40000,16,2]` float32 logits，含 t0 和四 future、三个 decoder 层 | 已有原生 replay 字节一致回执；不是 O 输出，也不是历史观测。不能直接替代 O logits。 |
| O checkpoint、`development_records.jsonl` 及后续评估账本 | O 权重、逐样本 confusion／其他评价统计，部分流程有 logits digest | 所核生产者没有保存 O dense logits／t0 probability 成品；digest 或 histogram 无法恢复它们。用 `common_change_evaluation_v2.input_only` 加原 `native.replay` 可从缓存重新生成 O 预测，只跑 future head、不需重跑图像／雷达 encoder。若只取 t0 decoder，仍须另审其窄读取路径，不能冒称已有成品。 |
| CRN centered train512/dev200 `predictions.json` | 每当前 anchor 的框中心、wlh、rotation、velocity、class、score、attribute；保留原框顺序 | 本地实际文件齐全并核过完成链 SHA。没有雷达点数／方差／逐扫证据、跨帧轨迹 ID、独立速度置信度。score 不是已校准的速度质量。 |
| 完整 `GeometryCache(root, complete_sha256, selection).load(split, ordinal, identity)` | 712 个 anchor 的预测框、固定网格 owner、CV field、坐标转换后状态与原 box index | 是当前预测框的确定性展开，不增加独立观测；owner 也不是 GT instance association。完整 712 的 complete/manifest 已镜像，本轮未重新读远端全部 NPZ。 |
| CRN train512 输入准备 v2 | 完整 train metadata，512 个 anchor 所需的 2048 个当前／历史 keyframe；每帧 1 个 radar BEV 点文件和 6 个 PV 文件，共 14,336 个文件、312,208,764 bytes，manifest 含逐文件 SHA | **有完成回执的远端输入资产**，不是仅理论能力。可新增 CPU 输入读取而无需模型前向；本地未镜像这些点文件。metadata 包含作者原 annotations，但 radar worker 只接收 sensor metadata，必须继续隔离 GT。 |

入口证据：[native cache 保存边界](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:315)、[输入白名单](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:174)、[只读输入 API](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/common_change_evaluation_v2.py:86)、[原 O 评价仅落 records](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/memory_experiment.py:68)、[GeometryCache](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/shared_rigid_geometry_cache_v1.py:204)。

单槽不等于没有利用历史：原 detector 先从历史图像迭代求 BEV，将其用于当前图像编码，再融合当前 radar；进入 `future_pred` 前才截取 `memory_queue_len=1`。因此当前 fused BEV 已隐含历史与雷达，无法从一槽反推出每个独立历史时刻。[历史编码](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py:158)、[当前编码及截取](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:996)。`future_state_motion_v1.capture_native_terminal` 得到的是模型预测的 t0+future feature，不会补出丢失的真实历史观测。

## 2. `radar_bev` 的精确语义与 D 对齐

[既有 radar5 完成回执](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/sota_p2_20260911/radar5_cache_complete.json)认证 29,049 个源／目标文件及所有 `.npy` header：float32 `[8,200,200]`。dataset 以 `DataContainer(stack=True)` 送入单 sample native 提取，因此模型边界为 `[1,8,200,200]`（由源 header 与 batching 源码共同确定；本轮没有新 tensor shape 实测）。[装载](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py:468)、[stack](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py:528)。

固定范围是 `[-51.2,-51.2,-5,51.2,51.2,3] m`，栅格 `[C,Y,X]`，XY 各 200 格、边长 0.512 m，扁平地址 `y*200+x`。点按当前 **LIDAR_TOP** 坐标 R 的 xyz 筛进范围，再沿高度聚合。最多 5 sweeps × 5 radar sensor 的全部有效点按 cell 池化，不保留独立 sweep 轴。

| 通道 | 实际编码（空格默认为 0） |
|---|---|
| 0 | `count > 0`，雷达回波存在性 |
| 1 | `log1p(min(count,32))/log1p(32)`；32 以上饱和，不能恢复精确高计数 |
| 2 | `clip(mean(z)/5,-1,1)` |
| 3 | `clip(mean(RCS)/50,-1,1)` |
| 4 | `clip(mean(vx_comp_R)/20,-1,1)` |
| 5 | `clip(mean(vy_comp_R)/20,-1,1)` |
| 6 | `clip(mean(sqrt(vx_comp_R²+vy_comp_R²))/20,0,1)`；不是均值速度向量的模 |
| 7 | `clip(mean(t_lidar_current-t_radar_return)/0.5,0,1)`；均值与截断后没有各 sweep 精确年龄 |

[实际聚合／归一化代码](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/radar_bev.py:262)。不能把 mean speed 与 mean velocity 的差称作已恢复协方差，尤其各通道有截断；`presence=0` 也不能解释为静止。

速度来源是 NuScenes radar point rows 8/9 的 **ego-motion-compensated velocity**。对每个 radar sweep，源码构造 `T(current_LiDAR <- sweep_radar)`，将 `[vx,vy,0]` 乘该完整变换的旋转部分，再取 XY；位移平移不施加于速度。点坐标乘完整刚体变换，但这条原 native loader 路径没有 `position += velocity*lag` 的物体运动补偿。[变换与速度](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/datasets/radar_bev.py:127)。这些是传感器／tracker 提供的补偿速度估计，不是真值完整二维物体速度；8 通道中没有每点 LOS、传感器位置／ID、径向速度约束或质量方差。源码另有可选 11 通道径向扩展，但既有 M0 缓存是 8 通道，不能声称已经拥有额外三通道。

D 标签使用同一当前 LiDAR R 的材料点 forward 位移 `p_h^R-p_0^R`，没有 future ego 坐标项；其输出布局 `[B,4,3,X,Y,Z]`。因此**物理 XY 坐标基一致，内存轴序不同，且雷达只有二维聚合**。若从原 sparse 的 `source_flat_indices` 读取对应 radar cell：`x=idx//(200*16)`、`y=(idx//16)%200`，访问 `radar[0,:,y,x]`；不可直接按 XYZ flattened index 取 2D radar，也不能把同 XY 的 16 个 z 点解释为 16 份独立雷达证据。[R 位移定义](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/motion_geometry.py:74)、[D 转轴](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/motion_prediction_head.py:31)、[原 gather](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:236)。

**真实性绑定已有两层。** 712 条原 cache record 都含 `files.inputs.sha256/bytes`，并另外含 `native_radar_tensor_sha256`；后者由 `_tensor_digest` 按 tensor dtype、shape、contiguous bytes 计算。[digest 定义](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:86)、[写入记录](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/m0_improvement_20260915/native_state_cache.py:338)。可以先认证 inputs 文件，再只读 radar 并核独立 tensor digest，不读取 `targets.npy`。原 `load_tokens` 仅返回 fused tokens，未对外返回 radar；新增只读观测导出应沿相同认证逻辑，但不是改本轮已冻结的 D 提取。

D 本就间接使用雷达：[原融合](/Users/yiminghua/2026Summer/WorldModel/dropple/code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py:385)是 `camera_bev + radar_encoder(radar_bev)`；[D 输入](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/train_source_motion_v1.py:215)仅收其当前 fused tokens。直接显式读取 radar 可提供对 D 输出的另一种观测约束，但与 D 误差可能相关；它不是“D 从没看到过的独立 radar”。本轮核的是冻结源码编码与原资产回执，没有重新从 raw sweeps 生成缓存来证明历史生成过程逐点相等。

## 3. CRN 输入资产比预测 JSON 多保留了什么

train512 官方预测接口保留完整 28,130 个 train infos 的历史索引，通过 `[0,-2,-4,-6]` 选当前／过去四个 keyframe，模型输入屏蔽 GT batch 字段；只保存当前预测。当前 centred JSON 的实际 box keys 均为 `sample_token,translation,size,rotation,velocity,detection_name,detection_score,attribute_name`。[接口与 GT 屏蔽](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/crn_train512_inference_v2.py:144)、[输出](/Users/yiminghua/2026Summer/WorldModel/dropple/analysis/o_motion_20260915/crn_train512_inference_v2.py:200)。`attribute_name` 又由预测速度模是否大于 0.2 和类别规则产生，不是额外动静观测。[作者 exporter](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/evaluators/det_evaluators.py:268)。

输入准备已另存 7×float32 的逐点数组：BEV 为 `[x,y,z,RCS,vx_comp,vy_comp,sweep_index]`；PV 为 `[u,v,depth,RCS,vx_comp,vy_comp,sweep_index]`。最多 8 sweeps，保留 sweep index，可在 CPU 统计逐 sweep 数量／速度分布，无需重新运行 CRN。不过作者 BEV generator 已用 velocity×time_lag 对点作运动补偿，落盘没有实际 time_lag、radar sensor ID、每点关联或 raw quality flags；把这些位置当作不依赖速度的原始运动轨迹，会产生循环解释。[BEV generator](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/scripts/gen_radar_bev.py:27)、[变换与补偿](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/scripts/gen_radar_bev.py:96)、[PV 保存](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/scripts/gen_radar_pv.py:94)。其具体速度变换必须沿作者接口重新核对后才能和 D 的 R 场配对；这里不把它与原 native radar 缓存等同。

模型实际 PV 输入还会把 vx/vy 变成速度模、返回前五列，从而丢掉 sweep index；eval 中超过 1536 点仍随机无放回抽样。seed0 不使多扫统计或抽样轨迹自动成为已导出的结果。[transform_radar_pv](/Users/yiminghua/2026Summer/WorldModel/dropple/code/radar-fusion-baselines/CRN/datasets/nusc_det_dataset.py:328)。因此，应从带 SHA 的**原预处理点文件**读取统计，不能从框 JSON 反演。

原 val6019 CRN 官方完整 export 也有来源回执，但本地 centred dev200 子集不等于已认证的历史检测序列。若复用完整 export 的历史 frame，仍需核过去时间、scene、token 覆盖、origin 转换和预测关联；train512 当前 export 没有对应 2048 历史帧的检测成品。不能用 GT instance IDs 补成推理轨迹。

## 4. 对下一步可行性的约束

- **不需新 native 前向：** 从原认证 `inputs.pt` CPU 导出上述 radar 8 通道，或从已认证 CRN 预处理输入读逐点／逐扫摘要。后者还需完成 train/dev 同定义资产与坐标认证；本轮没有完成这种新导出。两者都超出了“只对 D 四头自身打分”，但尚未证明能区分有用补覆盖与静止漂移。
- **需新的原始观测 CPU 提取：** 若要未被速度补偿的位置、精确每扫时间、传感器 LOS／径向残差和质量标记，要重新读取当前及过去 raw radar PCD 与 sensor metadata；这些信息不能由 8 通道均值或 CRN 7 列反演。已有原始数据路径／loader 能力不是完整 712 的新特征完成回执。
- **需新的 native encoder capture：** 若要各历史 BEV、camera-only／radar-only中间特征、当前和过去的显式视觉对应，需要从原观测路径重新提取；`future_pred` replay 只接已有单槽，无法恢复过去特征。O logits 或预测 future features 则仅需 future-head replay，但仍是模型派生量，不是独立时间观测。

原缓存允许给定 future ego/action，是既定条件预测协议，不是下一轮可擅自增加的运动可靠性证据。GT 动静组、future GT displacement、邻帧 GT、label 的实际 future dt 都不得成为新门控输入；任何输出场应先由合法当前／过去观测定义，再接原 support 做训练／评价。这里不指定新阈值、训练方案或性能结论。

## 5. 本轮实核与可复核绑定

本轮验证 train/dev index 与各自 complete 的 index SHA 相符；712 条记录均有 radar 独立 digest 和原 all5h×3layer replay PASS。native cache heavy payload 不在本地镜像，本轮未重验它们当前远端字节；只引用已有完成回执。CRN train512、centred dev200 的本地 manifest/predictions/原 export（train）逐项与 complete SHA 相符。CRN 输入准备的 manifest/history_plan SHA 相符；本地未含 `reused_prefix.json` 及远端 14,336 个点文件，因此不声称本轮完整重验该资产链。

| 文件／绑定 | SHA-256 |
|---|---|
| `native_state_cache.py`（与 cache extractor 一致） | `41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0` |
| 原 `drive_occworld.py`（与 cache source 一致） | `67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5` |
| 原 `radar_bev.py`（与 runtime_source_contract 一致） | `87ddb0e056163b00abe3962803073090661a2e76e20f382f548fb3134f6c9e77` |
| native train512 index / complete | `1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1` / `69e49dbd8c7667e9c6e17c2ce70b4f879e8f40c33a0018251b9de2ae978d01bc` |
| native dev200 index / complete | `fd42d2e511d754755fa5e2c06527077351869f06e868ecb8acb4404eadaa26c3` / `0128d88e9ed04b85c5bd86a00fc5e273199afe5cedb6fea6586080e6d48b8a84` |
| radar5 29,049 文件 header/传输 receipt | `7a30454a597146e7c9389db71578d69d80ed09f22aa731731d821f76a127b20a` |
| CRN source contract | `e147d9deca1409ee060a1ceb384f89424b55a9c40a866576b638f92f7f414e5a` |
| CRN 官方 `nusc_det_dataset.py` / `gen_radar_bev.py` / `gen_radar_pv.py`，均与 source contract 相符 | `45651e062f2258092f18720890ff7eff77c3f71664d751e04e865e9f55927ed3` / `c7c64b0554a4df28e18bda394265fed9344e465dcaf4a1bce19bdda6d712dfde` / `1cbc331527a69a71a9619ffbd56d528a76ee1b20c20c255d1b42cf25c6841f79` |
| CRN train512 input assets complete / manifest | `b554a6148d543c825ddc282aa1923b914104be08a5500ba02d638e750736ca06` / `0fcb6ef052155731ed4cf27119f4d951afc89d2b1bf269c5e8c88bc03b55e042` |
| CRN train512 predictions / centred dev200 predictions | `b7cab52b8c5c68d6afe1a0dab3728c9878067be826a7eb6e9a0ae7f37e1953ee` / `e1558af2a51ee40e976525d0679135066f93537fc8cfcc873511be44615cf05f` |
| 712 geometry cache complete / manifest | `a2504a0389cb2756032531961e305e22287ec890c96085e5ea29b2e61e4f1c99` / `9952c8fb8edcc15fee607d362e1a5a1396e3048d4ebc09945918f88124675ab2` |

资产定位：native 回执在 `analysis/m0_improvement_20260915/server_results/cache/campaign_cache_v1/{train,development}`；CRN 输入回执在 `analysis/o_motion_20260915/crn_train512_assets_v2_receipt`，其 launch 记录远端资产 `/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/crn_train512_state_v2/assets`；两预测镜像在 `crn_train512_full_official_env_v1` 与 `crn_state_dev200_centered_v1`；geometry 回执在 `server_results/training/shared_rigid_geometry_cache_v1`。以上远端路径只是已有回执记录，本轮未连接服务器确认当前存活或重新读取。
