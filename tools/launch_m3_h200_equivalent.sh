#!/usr/bin/env bash
# Start the reviewed M3 protocol only after the same config passes real DDP.
set -euo pipefail
umask 077
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project=${M3_PROJECT_ROOT:-/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200}
env_dir=${M3_ENV_DIR:-${project}/envs/hym_driveocc_m3}
config=${M3_CONFIG:-${code_dir}/configs_runtime/m3_h200_equivalent.py}
report=${M3_PREFLIGHT_REPORT:?Set M3_PREFLIGHT_REPORT to the successful preflight_summary.json}
export CUDA_VISIBLE_DEVICES=${M3_PHYSICAL_GPUS:-GPU-74b9b73f-c405-55bc-bf76-f8aa85d1e7fe,GPU-000b6236-3632-a001-9667-1f02cbb61c8b}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${env_dir}/bin:${CUDA_HOME}/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=${env_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR=${project}/cache/torch_extensions
export MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=${code_dir}${PYTHONPATH:+:${PYTHONPATH}}
test -x "${env_dir}/bin/python"
cd "$code_dir"
python - "$config" "$report" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from mmcv import Config

config, report_path = map(lambda p: Path(p).resolve(), sys.argv[1:])
report = json.loads(report_path.read_text())
if not (report.get('status') == 'PASS' and report.get('world_size') == 2
        and report.get('optimizer_updates_completed', 0) >= 3
        and report.get('dataloader_workers_exited_cleanly') is True
        and Path(report.get('config', '')).resolve() == config
        and report.get('config_sha256') == hashlib.sha256(config.read_bytes()).hexdigest()):
    raise RuntimeError('The exact resolved config must pass at least three real two-rank updates')
cfg = Config.fromfile(str(config))
if cfg.get('m3_h200_requires_resolution', False) or cfg.get('resume_from'):
    raise RuntimeError('This launcher starts a resolved M3 experiment from the common pretrained weights')
if not (cfg.data.samples_per_gpu == 1
        and cfg.optimizer_config.type == 'EquivalentCumulativeOptimizerHook'
        and cfg.optimizer_config.cumulative_iters == 2
        and cfg.optimizer_config.expected_optimizer_steps_per_epoch == 5983
        and cfg.lr_config.cumulative_iters == 2
        and cfg.lr_config.warmup_iters == 500
        and cfg.runner.max_epochs == 24 and cfg.get('fp16') is None
        and cfg.tf32_policy == dict(matmul=True, cudnn=True)):
    raise RuntimeError('The reviewed H200 training contract has changed')
work_dir = Path(cfg.work_dir)
if work_dir.exists() and any(work_dir.iterdir()):
    raise RuntimeError('The initial formal run requires an empty work directory: ' + str(work_dir))
selected = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(selected) != 2 or len(set(selected)) != 2 or not all(x.startswith('GPU-') for x in selected):
    raise RuntimeError('Select exactly two verified H200 GPU UUIDs')
raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
inventory = {row[0].strip(): [x.strip() for x in row[1:]] for row in csv.reader(raw.splitlines())}
for gpu in selected:
    name, memory, utilization = inventory[gpu]
    if 'H200' not in name or float(memory) > 32 or float(utilization) > 5:
        raise RuntimeError('Selected GPU is not an idle H200: ' + gpu)
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                '--format=csv,noheader'], text=True).splitlines()
if any(gpu.strip() in selected for gpu in apps):
    raise RuntimeError('Another compute process is using a selected GPU')
print(json.dumps(dict(status='READY_TO_START', config=str(config),
                      preflight=str(report_path), gpu_uuids=selected,
                      work_dir=str(work_dir), global_batch=4,
                      optimizer_updates_per_epoch=5983, epochs=24), indent=2), flush=True)
PY
if [[ ${1:-} == --check ]]; then
  exit 0
fi
if (( $# != 0 )); then
  echo 'Usage: launch_m3_h200_equivalent.sh [--check]' >&2
  exit 2
fi
# A standalone rendezvous chooses an available local port for this one node.
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  tools/train.py "$config" --launcher pytorch --seed 0 --deterministic --no-validate
