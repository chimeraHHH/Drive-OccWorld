#!/usr/bin/env bash
set -euo pipefail

code_dir=/home/huayiming/Workspace/RadarFlowOcc/code/Drive-OccWorld
work_dir=/storage/data/metaiot_data/huayiming/RadarFlowOcc/work_dirs/gmo_radar_v0_l40s720_cached_w2_4g_seed0
eval_dir=${work_dir}/eval_epoch24_official
checkpoint=${work_dir}/epoch_24.pth
config=projects/configs/radarflowocc/action_condition_GMO_radar.py
cache_dir=${code_dir}/data/radar_bev_cache/nuscenes_trainval_5sweeps_200x200_v0_float32
manifest=${cache_dir}/manifest.json
precompute_pid_file=${eval_dir}/precompute_val_radar.pid
expected_val_samples=5119
expected_total_cache_files=29049

mkdir -p "${eval_dir}"
test -r "${checkpoint}"
test -r "${precompute_pid_file}"
precompute_pid=$(cat "${precompute_pid_file}")

while true; do
  if "${code_dir}/../../envs/hym_driveocc_py38/bin/python" - \
      "${manifest}" "${expected_val_samples}" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
try:
    with open(path) as handle:
        manifest = json.load(handle)
    val = manifest['splits']['val']
except (OSError, KeyError, ValueError):
    raise SystemExit(1)

complete = (
    int(val.get('samples', -1)) == expected
    and int(val.get('written', -1)) + int(val.get('already_cached', -1))
    == expected
    and int(val.get('verified_samples', 0)) >= 1
)
raise SystemExit(0 if complete else 1)
PY
  then
    break
  fi

  if ! kill -0 "${precompute_pid}" 2>/dev/null; then
    echo "Validation radar-cache precompute exited without a complete manifest" >&2
    exit 2
  fi
  sleep 30
done

cache_files=$(find "${cache_dir}" -maxdepth 1 -type f -name '*.npy' | wc -l)
if [[ "${cache_files}" -ne "${expected_total_cache_files}" ]]; then
  echo "Unexpected cache count: ${cache_files}, expected ${expected_total_cache_files}" >&2
  exit 3
fi

# GPUs 0 and 1 currently host other users' jobs. Choose four genuinely free
# devices from 2-7 immediately before launching the official distributed test.
while true; do
  mapfile -t app_uuids < <(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader |
      sed '/^[[:space:]]*$/d' | sort -u)
  free_gpus=()
  for index in 2 3 4 5 6 7; do
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
      awk -F', ' -v target="${index}" '$1 == target {print $2}')
    occupied=0
    for app_uuid in "${app_uuids[@]:-}"; do
      if [[ "${uuid}" == "${app_uuid}" ]]; then
        occupied=1
        break
      fi
    done
    if [[ "${occupied}" -eq 0 ]]; then
      free_gpus+=("${index}")
    fi
  done
  if [[ "${#free_gpus[@]}" -ge 4 ]]; then
    break
  fi
  sleep 60
done

selected_gpus=$(IFS=,; echo "${free_gpus[*]:0:4}")
printf '%s\n' "${selected_gpus}" > "${eval_dir}/selected_gpus.txt"

export PATH=/home/huayiming/Workspace/RadarFlowOcc/envs/hym_driveocc_py38/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
export CUDA_VISIBLE_DEVICES="${selected_gpus}"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
export PYTHONPATH="${code_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PORT=16859
# Continuous validation chunks have very different runtimes. The official
# collector synchronizes ranks only after each chunk, so keep the watchdog
# above the maximum observed rank skew without changing any evaluation math.
export DIST_TIMEOUT_SECONDS=10800

cd "${code_dir}"
exec bash ./tools/dist_test.sh \
  "${config}" \
  "${checkpoint}" \
  4
