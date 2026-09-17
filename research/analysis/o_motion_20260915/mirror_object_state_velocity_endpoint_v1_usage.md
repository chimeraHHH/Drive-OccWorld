# N/P/Z 完整终点只读镜像

入口 `mirror_object_state_velocity_endpoint_v1.py` 只接受实际唯一派发回执 SHA `6a972e7c3e821f2a4d6d70cc229fd4872bb658c095f9edb19477cf0d7a6f078c`。默认本地检查，不调用 SSH、Torch、GPU：

```sh
python3 -B analysis/o_motion_20260915/mirror_object_state_velocity_endpoint_v1.py --mode check
```

主代理审核后需要取回时，明确使用 `mirror`，目标目录必须不存在：

```sh
python3 -B analysis/o_motion_20260915/mirror_object_state_velocity_endpoint_v1.py \
  --mode mirror \
  --out analysis/o_motion_20260915/server_results/training/object_state_velocity_assignment_v1
```

远端沿真实 launch 的原 Python 单次执行只读程序，SSH 保留严格主机密钥检查，整体上限 180 秒。先核 launch/spec、37 个源文件及 runner 70988、child 70989 的 UID/startticks；仍在运行或收尾仅返回 `LIVE_OR_FINALIZING_NO_RESULTS_FETCHED`，不读取结果或 output.log，不创建本地输出目录。非成功终态返回 `NOT_SUCCESSFUL_TERMINAL_NO_RESULTS_FETCHED`，不读取部分成绩。

仅在 `EXITED_ZERO`、returncode 0、finished_utc、两 PID 实际不存在、完整200/100/16074支持及六件 complete SHA ledger 均通过后，取回全部7件结果和 `_job/{state.json,launch.json,spec.json,output.log}`。额外 `_mirror_receipt.json` 记录每件原始字节数/SHA、实际进程身份、source/dispatch/helper绑定和观察时间；本地再次核所有字节及 complete ledger，独占创建目录并最后写镜像回执。

不派发、不重试、不写远端、不覆盖目录。若 SSH 或本地写入中断，报错；没有完整镜像回执不能称镜像完成。准备阶段只实际执行本地来源检查与语法检查，尚未 SSH 或断言本次诊断终态成功。
