#!/usr/bin/env bash
# Source in an existing shell; changes process environment variables only.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Use: source runtime_env.sh\n' >&2
  exit 1
fi
M3_RUNTIME_PROJECT=/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200
M3_RUNTIME_ENV=${M3_RUNTIME_PROJECT}/envs/hym_driveocc_m3
M3_RUNTIME_CODE=/home/wangning/Workspace/RadarFlowOcc-m3
if [[ ! -x "${M3_RUNTIME_ENV}/bin/python" || ! -d "${M3_RUNTIME_CODE}" ]]; then
  printf 'The preserved M3 environment/code is unavailable; check the NAS mount and paths.\n' >&2
  return 1
fi
export M3_RUNTIME_PROJECT M3_RUNTIME_ENV M3_RUNTIME_CODE
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${M3_RUNTIME_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${M3_RUNTIME_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="${M3_RUNTIME_PROJECT}/cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Select only the deployed plugin path, avoiding stale MMCV source overrides.
export PYTHONPATH="${M3_RUNTIME_CODE}"
# GPU visibility and numerical training settings remain controlled by the caller.
