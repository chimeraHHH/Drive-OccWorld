# Supported fusion V2：只修复固定 CE 的工程比较

V1 源码 SHA `54bd8747c534b5d0a74474651350398f3d304aade3a23d071a8d3458e44fe4b9` 与失败任务原样保留。失败发生在首样本，A/Z 均为 0 次优化更新，不是候选性能结果。

真实单锚诊断见 `server_results/diagnostics/supported_fusion_loss_diagnostic_v1/result/result.json`，SHA `f57c0cb5aa819f4302bcac6dca1fd16484b96f6db3db489b07fa01da83f5386f`。运行环境 PyTorch `2.1.2+cu121`。A/Z 的固定两层预测与完整 GT 字节相同，四次调用均未修改各自预测或 GT；仅固定 CE inter0/inter1 分别相差 3/1 ULP。两次同一 A 预测的 detached/no_grad 重算，CE 也有 0–3 ULP 波动；固定层 Lovász、sem/geo 均精确相同。诊断正常退出，优化器更新 0。

PyTorch 2.1.2 的 CUDA spatial NLL reduction 使用 `gpuAtomicAdd` 累加，并明确标记 sum/mean reduction 的非确定性。这与当前重复实验吻合，但本次没有做 profiler kernel tracing。参见 [官方 NLLLoss2d.cu](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/native/cuda/NLLLoss2d.cu)。

新文件 `train_supported_fusion_v2.py` SHA `61cb7c96fc8127f4b3b5ef34fff380e0c2076210463cff29e4cee5d3a5d87d7c` 仅把自己的源码绑定名换成 V2，并修改固定项比较及日志：CE 使用 `math.isclose(rel_tol=1e-5, abs_tol=1e-7)`；Lovász 仍要求精确相同。每样本记录双方原值、差值、float32 ULP、比较规则和通过状态；超界仍停止。该容差是经授权的工程阈值，不是从一个样本推得的全数据最大误差界。

没有改写、舍入或缩放用于 backward 的任何损失或梯度。t0、前两层、未覆盖位置与背景 logits 的字节门，原模型、GT、参数范围、A/Z 初始化与独立优化器、顺序、更新数、学习率及预算全部保留。

本地已用真实诊断三对标量核验新门通过；超出 CE 容差的反例及 Lovász 单 ULP 改变均拒绝。Python 3.10 语法通过；其他原函数 AST 未改变。尚未运行 V2 GPU 预检，也未修改或冻结任何协议。
