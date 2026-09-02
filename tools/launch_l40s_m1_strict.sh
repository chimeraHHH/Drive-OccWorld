#!/usr/bin/env bash
set -euo pipefail

workspace=/home/huayiming/Workspace/RadarFlowOcc
code_dir="$workspace/code/Drive-OccWorld-m1"
env_dir="$workspace/envs/hym_driveocc_py38"
extension_dir="$workspace/cache/torch_extensions_m1_l40s"
config=projects/configs/radarflowocc/action_condition_GMO_radar_m1.py
work_dir=/storage/data/metaiot_data/huayiming/RadarFlowOcc/work_dirs/gmo_radar_m1_v0_l40s720_cached_w2_4g_b1_seed0_scratch

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PORT="${PORT:-16860}"
export PATH="$env_dir/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$code_dir${PYTHONPATH:+:$PYTHONPATH}"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.6
export TORCH_EXTENSIONS_DIR="$extension_dir"
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

IFS=, read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#selected_gpus[@]} -ne 4 ]]; then
    echo "Expected exactly four physical GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi

for gpu in "${selected_gpus[@]}"; do
    apps=$(nvidia-smi -i "$gpu" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits)
    if [[ -n "$apps" ]]; then
        echo "GPU $gpu is not idle: $apps" >&2
        exit 3
    fi
done

if ss -ltn | grep -Eq ":${PORT}[[:space:]]"; then
    echo "Master port $PORT is already in use" >&2
    exit 4
fi

if find "$work_dir" -maxdepth 1 -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite an existing checkpoint in $work_dir" >&2
    exit 5
fi

mkdir -p "$extension_dir" "$work_dir"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$work_dir/physical_gpus.txt"
printf '%s\n' "$code_dir" > "$work_dir/code_dir.txt"

cd "$code_dir"
exec nice -n 10 ionice -c 2 -n 7 \
    bash tools/dist_train.sh "$config" 4 \
    --work-dir "$work_dir" \
    --seed 0 \
    --cfg-options \
        optimizer.lr=0.0001 \
        data.samples_per_gpu=1 \
        data.workers_per_gpu=2 \
        'data.train.pipeline.2.data_aug_conf.reisze=[720]' \
        log_config.interval=10
