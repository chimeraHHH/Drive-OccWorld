# M3 H200 GPU/DDP preflight

Run from the M3 code root using the configured H200 environment and the **resolved**
formal configuration. Select the two currently available H200 UUIDs before launch.
Use a new diagnostics directory outside the formal work directory.

```bash
CUDA_VISIBLE_DEVICES="$M3_H200_GPU_UUIDS" "$M3_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/m3_gpu_preflight.py \
  configs_runtime/m3_h200_equivalent.py \
  --diagnostics-dir /storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/diagnostics/gpu_preflight_01 \
  --updates 5
```

`--updates` accepts 1–100 complete optimizer updates (default 5); each update uses
two physical ranks, one sample per forward, and two accumulated microsteps. The
production model builder, real dataset, virtual sampler, MMCV runner, DDP wrapper,
LR hook, AdamW, gradient clipping, and checkpoint initialization are reused. The
optimizer hook is subclassed for observations and verified worker cleanup. Before
prefetch begins, its batch sampler receives a finite prefix of the original
virtual sampler; the prefix keeps the original full `__len__` and index order.
Only the requested `2N` microsteps are dispatched. The loader reaches normal
exhaustion, every worker must exit with code 0, and the hook then stops MMCV before
another epoch begins. The original 24-epoch / 5983-updates-per-epoch schedule and
full dataset remain intact. Validation and checkpoint saving are disabled in the
diagnostic config copy; the input config and formal work directory are untouched.
Existing diagnostics directories and training checkpoint resumes are rejected.

Success creates `preflight_summary.json` with:

```json
{
  "status": "PASS",
  "config": "/absolute/path/to/resolved_config.py",
  "config_sha256": "sha256 of the small input configuration only",
  "world_size": 2,
  "optimizer_updates_completed": 5,
  "dataloader_workers_exited_cleanly": true,
  "effective_global_batch": 4,
  "training_checkpoint_saved": false,
  "reports": ["rank 0 report object", "rank 1 report object"]
}
```

Per-rank JSON includes finite loss/accumulated-gradient checks, camera/M3 forward
call counts, before-clipping gradient norms, actual AdamW step counts, changed M3
parameters, a full comparison of every trainable parameter between ranks, timing,
peak CUDA memory, GPU identity information, and dependency versions. Rank failure
writes `rank_N.json` with `status: FAIL`; no successful summary is written. Logs,
the diagnostic config copy, input protocol, and GPU UUID inventory are retained.
The cleanup report verifies dispatched/sent/received/yielded task counts all equal
`2N`, no outstanding task, and zero exit codes from all workers. Cleanup failures
prevent PASS; they are not silenced or left to an interpreter-exit destructor.
During diagnostic cleanup, PyTorch 2.1's parent-side worker join interval is
temporarily increased from 5 seconds to 60 seconds **per worker**, then restored
even if shutdown raises. This allows large dataset replicas time to release their
objects. A subsequent bounded join (at most 5 seconds per worker) reaps any
processes; timeout/termination or any nonzero exit code still means FAIL. The
report retains `grace_seconds`, `elapsed_seconds`, counters and worker exit codes,
including on failure. The formal training process is not modified by this setting.

This is a full-model compatibility and update test. It is neither an accuracy
result nor a throughput benchmark: startup and synchronization checks affect the
timings. CPU development checks are:

```bash
python -m unittest discover -s tests -p test_m3_gpu_preflight.py -v
python -m py_compile tools/m3_gpu_preflight.py tests/test_m3_gpu_preflight.py
```

The test suite includes a real CPU MMCV runner using the production sampler and
accumulation/LR hooks, two spawn workers with tensor IPC, and 10 complete updates.
It checks that the full 11966-microstep epoch length and 24-epoch schedule remain
unchanged while only 20 microsteps are dispatched and all workers exit normally.
Additional tests call PyTorch's actual shutdown implementation with test-owned
spawn workers: a slow normal exit succeeds within its grace period, a timeout is
reaped and rejected, and the temporary join interval is restored after exceptions.
The 11 targeted tests passed on the H200 CPU runtime. A real large-dataset worker
teardown check and the final GPU/DDP preflight are separate required validations;
the scaled worker tests alone do not establish that the large dataset exits cleanly.
