# CRN train512 传感器输入续接 v2（待审，不派发）

原 v1 源和输出保留。新增 CPU 续接上限 7200 秒，父 runner 上限 7260 秒，仍两个投影 worker，OPENBLAS/OMP 各 1。根据根代理已报告的 162.65→253.52 秒、62→140 帧，最近速度约 0.858 帧/秒，剩余 1908 帧约 2223 秒；这只是资源估算，不是完成承诺。v2 另有复制和完整元数据校验成本。推理预算仍由真实 pilot 后另定。

只有根代理确认旧 runner/child UID、startticks、终态和进程组退出后才能运行。脚本先核停止凭证与旧 launch/spec/state、25 个作者源文件合同、选择，再读取旧 progress。旧 v1 已 fsync 的完成前缀按 needed_history_indices 顺序处理，每帧恰好 1 BEV + 6 PV 文件；未提交或多余旧文件不读取。旧版没有持久化每文件 SHA ledger，因此这里只声称停止后的源/目标逐字节 SHA、有限性、长度与原完成顺序合同认证，不伪称存在旧 ledger。

完整 train infos 不重新生成；复制 SHA 必须精确，重新核全部 28130 样本 / 700 场景的 token、scene、timestamp 顺序，以及实际消费的历史帧全部相机/LiDAR pose、calibration、文件和时间 metadata。原作者雷达两 worker、8 sweeps、过滤和投影不变。GT annotation 不进入 worker，full infos 中原标签保留但原推理仅传 image/matrices/metadata/radar。

CLI（下列实际停止凭证路径及 SHA 由根代理在旧作业终止后填写，当前不构成可派发授权）：

```text
python -B prepare_crn_train512_inputs_v2.py
  --repo <原已验证 CRN repo>
  --source-data-root <原 CRN data view>
  --source-contract crn_train512_source_contract_v1.json
  --source-contract-sha256 e147d9deca1409ee060a1ceb384f89424b55a9c40a866576b638f92f7f414e5a
  --selection selection_v1.json
  --old-assets <crn_train512_state_v1/assets>
  --old-launch <原 flat launch JSON，不是含 stdout 字符串的本地 envelope>
  --old-job <crn_train512_state_v1/job_prepare>
  --old-terminal-receipt <真实终止凭证 JSON>
  --old-terminal-receipt-sha256 <该凭证的实际 SHA256>
  --out <全新 crn_train512_state_v2/assets>
  --max-seconds 7200
```

停止凭证 schema：

```text
schema = crn-train512-stopped-job-v1
status = VERIFIED_STOPPED
launch_sha256 / spec_sha256 / state_sha256 = 三个真实控制文件摘要
runner = {pid, uid, startticks}，与原 launch 相同
child = {pid, uid, startticks}，与原 terminal state.child 相同
boot_id = 当前 /proc/sys/kernel/random/boot_id
runner_missing = child_missing = owned_process_group_missing = true
```

脚本还会现场只读复核两个 PID 均不存在、旧 child PGID 无存活成员；只接受 FAILED/TIMEOUT/USER_SIGNAL 终态且旧 assets 没有 complete。如果 v1 自行成功，应使用原完整产物，不做续接。失败后不自动重试。deadline 是检查点式，父 runner 的硬时间界限及 scoped cleanup 必须保留。

产物仍为 `crn-train512-assets-v1` / `crn-train512-assets-complete-v1`，新增 preparation_revision=2、reuse 和 reused_prefix.json；完整 radar_files ledger 包含已复制及新生成文件。`crn_train512_inference_v2.py` 唯一变化是 prepare helper 文件名，因此可严格核 v2 准备源；模型、原导出、原点修正和输出 schema 不变。新包须包含 prepare v1（纯 helper 复用）、prepare v2、inference v2、origin adapter、25 文件 source contract、selection 及原 job_runner。

本机已验证 Python 3.8 AST、两个 CLI help、25 份实际源 SHA、真实 512 选择、微型文件精确复制/禁止覆盖和 7 路径结构。没有验证真实停止凭证、未复制真实前缀、未运行投影或 GPU。原 v1 文件未改，当前没有派发。
