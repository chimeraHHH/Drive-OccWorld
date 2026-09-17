# Cpl/Fix 固定终点评价链独立源码审查

**未发现实质阻断，仅为静态审查结论。** 已全文审阅终点加载器、评价器、统计器、campaign，以及原生评价、物理支持和统计 helper 的实际调用。未修改冻结源，未 SSH、导入 Torch、读取中间开发分数、加载权重或执行评价。

1. **终点与来源。** 加载器要求两臂均为固定 512 update / 2048 example，核完整训练日志、四轮顺序、逐步 LR、paired RNG/input 和真实非 None 参数更新次数。随后评价器才读取绑定的 final.pth，严格加载 head 与对象模块，并核参数摘要、Adam moments/steps 和 RNG；不恢复优化器。统计端跳过大权重重读的范围有明确声明，不把元数据审核写成自己重算张量。
2. **前向与轴序。** 完整当前预测几何与 t0 BEV 先产生 O/Cpl/Fix 原生输出及完整物理场，随后读取 GT 与未来原始标签用于评分。场 `[4,Q,3]` 经冻结 bridge 转为 `[1,4,3,X,Y,Z]`，再按原 XYZ-C-order source 索引取值；原三 decoder / 五时域 occupancy 走原插值和 argmax。用 baseline 的无参数原 evaluator 统计候选 logits，不会替换候选预测。
3. **完整分母。** 固定 dev200 / 100 场景全部保留；O 逐样本五时域 confusion 必须复现旧结果。每臂 t0 对自身 confusion 和共同 GT 检查，不要求 t0 预测跨模型相同。D、CRN-CV、Cpl、Fix 同 16074 个 anchor-instance-horizon 键，原有效点、零位移误差、速度组及真实 dt 一致；漏检、预测框未覆盖点、未来出 ROI 的原有效标签均保留。CRN-CV 重新计算与已认证 v2 记录比较，仅允许既定双精度舍入界；D 采用认证的既有逐对象记录。没有 O 原生 flow/EPE 主张。
4. **统计定义。** 复用冻结纯数学函数，私有模型名参数化不污染原模块全局。原整数先池化、逐时域比率后取四个 future 的均值；全部 t0、FP/FN、八类运动归属和四状态 transition 保留。物理 EPE 先对象内点平均，再 anchor-instance 等权；median/p90 为描述统计。10000 次 seed11 配对整场景 bootstrap 保留该场景全部 anchors/objects，空分母保持 null；无新阈值、选 checkpoint 或自动晋级。
5. **调度接口。** 纯 AST 检查确认三阶段所有必需参数均由冻结 plan 给出或按实际终态动态补入，输出目录和资源参数对应。39 个评价源码、48 个 package 文件及统计源 SHA 与本地实际文件一致；四个入口 Python3.10 AST 通过。train→evaluation→analysis 各一次；子进程继承专用进程组，外层冻结 runner 对所属组执行超时清理。资源为 train 内 4800 / stage 外 4860 s，evaluation 内 1800 / stage 外 1860 s，analysis 外 120 s，campaign 外 6900 s，GPU allocated 上限 32 GiB；这不是本次独审实测耗时或保证。

仍须等待真实固定终点张量加载、200 次评价和汇总完成，才能审查实际数值。全 head 更新使 t0 与未来预测都可能变化，transition 变化不等于孤立运动改善；物理标签是刚体框材料点代理，并非实测稠密 scene flow。单训练种子及历史开发集暴露边界不变。本审查不作性能或 full 晋级结论。

绑定：

| 文件 | SHA256 |
|---|---|
| shared_rigid_final_endpoint_v1.py | `549112b0a8de1fe5dd705da5376aff3622f48a087a67b2f0bef2d1a357191917` |
| evaluate_shared_rigid_state_v1.py | `9785bb3082ac2c805ab54d291152391f0a4943f9948934ce9932cb4e530e9e6a` |
| summarize_shared_rigid_state_v1.py | `9376155e95ac3d17001139b980c7c9fb30e0afda3c4a381e71941b19317ecfc6` |
| shared_rigid_formal_campaign_v1.py | `8f7dbe2b9307b7881d5642a7a964046b2a992b5c7139f8f62bf8f8f6f42a0b1c` |
| 训练 protocol | `9c12d8ba84de797f3e452c0ea6449d881761325241e7fdca807d0cd84aa56199` |
| 评价 protocol | `7b7aeb21c88c905bbe7eb0ed14a5eb3d261d5b8b02d210a1a07bf6c7dddf8353` |
| campaign plan | `412e4a375d9eca1a5d9ba7d442db4f30344f280ac9001172d8be946f4ae56054` |

机器可读审查：同目录 `shared_rigid_fixed_eval_source_audit_v1.json`，SHA `a06a240d9186338ce5066989bead8bc6cba765de0bbf2a449787cb1b2bb852a7`。本地检查脚本首次因试图把 AST 的 `type=float` 当常量解析而停止；修正检查器后完成，生产代码未改，也未运行任何评价。
