# 对象状态直接生成未来占据：一项经源码核实的依据

核查于 2026-09-16；仅阅读原论文及作者代码，未安装、运行或部署外部模型。此次只纳入 **MotionPerceiver（RA-L 2024，9(3):2822–2829）**，不为凑数量加入只有相近标题的工作。它与已读 FIERY／BEVerse 的区别是：**位置、朝向、速度、尺寸本身是模型输入，先更新场景状态，再从预测状态查询占据**，而非先从图像生成稠密未来特征，再附加运动读出。

**论文证据。** §IV-A–C、式(7–12)给出观测 token → 状态更新 → 时间传播 → 位置查询的前向链；不要求输入实例 ID。§V-A 的实验使用 WOMD 对象状态；因此不能把其成绩解释为真实检测器误差下的端到端收益。Fig.1／§V-C 展示历史已见对象短时缺测后的保留与衰减，以及后来新观测带来的对象加入；最初未检出的对象并未被可靠恢复。§V-E 的额外 flow 是简化中心位移监督，忽略转向，不能等同本项目刚体材料点或真实 scene flow。[原论文 v2](https://arxiv.org/html/2306.08879v2) · [作者项目与正式出版信息](https://sites.google.com/monash.edu/motionperceiver)

**已读作者源码，固定 commit `cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f`：**

| 必须核清的连接 | 实际代码证据及边界 |
|---|---|
| 对象状态如何进入模型 | `waymo_motion_pipe` 直接读取 WOMD `state/{past,current}/x,y,bbox_yaw,velocity_x,velocity_y,vel_yaw,width,length`，位置与向量分别变换；`TrafficIA.forward` 对位置／朝向 Fourier 编码，并拼接速度和尺寸，不读取 instance ID。所核 `no-ctx` 只开放历史索引 `[0,5,10]`，没有用未来状态作预测输入。[数据 L187–310](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/dataset/waymo.py#L187) · [token L319–346](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/_iadapter.py#L319) |
| 状态是否真正影响未来占据 | `process_timestep` 先 self-attention 传播，再仅在允许观测时刻用对象 token cross-attention 更新；`MotionPerceiver.forward` 把各时刻 latent 交给 decoder。`PerceiverDecoder.aux_forward` 用位置 query 对 latent 做 cross-attention，再输出 heatmap。这是前向条件依赖，不是辅助损失旁路；也没有“对象速度乘时间后硬搬框”的显式运动积分。[更新 L519–546](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/motion_perceiver.py#L519) · [解码 L405–422](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/perceiver_io.py#L405) |
| 覆盖／漏检是否被隐藏 | invalid 对象由 attention mask 排除；输入接口可接受可变有效对象，但所核配置 `only_vehicles: true, filter_future: true`。后者调用 `maskInvalidFuture`：在过去至当前从未 valid 的实例，其**整段未来标签置无效**。因此该配方不是我们保留 future-only、八类 GMO、全 GT 风险的任务，不能复制它的分母或宣称其解决无源对象。[配置](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/cfg/no-ctx.yml#L9) · [过滤 L64–90、L215–226](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/dataset/plugins/maskfrompoly/maskfrompoly.cpp#L64) |
| 训练连接 | occupancy 由标注框 rasterize，`OccupancyFocal` 用 BCE-with-logits 加 focal／类别权重。所核 encoder v7 在未采样监督时刻 `no_grad`，监督时刻后可 detach latent，**不能称为完整跨时域 BPTT**。论文 focal γ=2，而当前源码配置未显式给 γ、注册 dataclass 默认 γ=0.25；不能未经复现称当前默认命令严格重现论文配方。[loss L44–90](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/loss.py#L44) · [v7 L609–667](https://github.com/5had3z/motion-perceiver/blob/cc68a9d1fc0f1b4cfb49d26b6872ba94a723819f/src/model/motion_perceiver.py#L609) |

**可复用程度。** 作者仓库公开完整前向／标签／训练代码；本次 README、Git 文件树和 releases API 未找到训练后模型下载入口（releases 为空），不等于断言作者任何地方都没有权重。仓库未识别 LICENSE；依赖 DALI 自定义 C++ 插件、gcc-13、OpenCV/TBB、Konductor，不能承诺直接拷入当前环境即可使用。[作者仓库](https://github.com/5had3z/motion-perceiver)；本次 15 个源码文件均核 Git blob 身份，SHA 和阅读范围见同名 JSON。

**对本项目的可执行推论（尚未验证）。** 若固定 CRN 同支持诊断表明预测状态确有可用信号，值得优先检验“把可部署的对象状态 token 接入 O 的未来状态更新”，而不是再训练一个只从未来状态读运动的头。保留 O 的稠密场景通路、原完整 GT、t0／全部未来时域和所有运动组；对象条件不能成为占据输出的覆盖硬门。最有识别力的容量／输入对照应保留相同预测框位置、尺寸、类别及全部新参数，仅移除速度等运动分量，避免把额外检测器的静态定位收益误称运动信息收益。CRN 的额外参数、预训练及历史输入必须单列。状态到占据的总体连接、无实例 ID 的集合编码和概率占据解码均已有先例；潜在研究问题是**带真实检测误差和漏检的状态证据，怎样在完整 3D 任务上提供可证实的运动增量**，本卡不能证明答案或新颖性。
