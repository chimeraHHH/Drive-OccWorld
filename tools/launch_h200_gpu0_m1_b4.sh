#!/usr/bin/env bash
set -euo pipefail

# Independent H200 GPU 0 run: M1, physical batch size 4, from Epoch 1.
workspace=/home/wangning/Workspace/RadarFlowOcc
code_dir=${workspace}/code/Drive-OccWorld-m1m2
env_dir=/home/wangning/.conda/envs/driveocc_h200
work_dir=${workspace}/work_dirs/gmo_radar_m1_v0_h200_gpu0_720_cached_w2_1g_b4_seed0_scratch
pretrained_checkpoint=${code_dir}/pretrained/r101_dcn_fcos3d_pretrain.pth
radar_cache=${code_dir}/data/radar_bev_cache/nuscenes_trainval_5sweeps_200x200_v0_float32
extension_cache=${workspace}/cache/torch_extensions_m1m2
log_file=${work_dir}/strict_h200_gpu0_1g_b4_m1_scratch.log

test -x "${env_dir}/bin/python"
test -r "${pretrained_checkpoint}"
test -d "${radar_cache}"
radar_sample=$(find "${radar_cache}" -maxdepth 1 -type f -name '*.npy' -print -quit)
test -n "${radar_sample}"
test -r "${radar_sample}"
mkdir -p "${work_dir}" "${extension_cache}"

if find "${work_dir}" -maxdepth 1 -type f -name 'epoch_*.pth' -print -quit | grep -q .; then
  echo "Refusing to reuse a work directory containing epoch checkpoints: ${work_dir}" >&2
  exit 1
fi

export PATH="${env_dir}/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR="${extension_cache}"
export MAX_JOBS=2
export PYTHONPATH="${code_dir}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${code_dir}"

exec nice -n 10 ionice -c 2 -n 7 \
  "${env_dir}/bin/python" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=16856 \
  tools/train.py \
  projects/configs/radarflowocc/action_condition_GMO_radar_m1.py \
  --launcher pytorch \
  --work-dir "${work_dir}" \
  --seed 0 \
  --deterministic \
  --no-validate \
  --cfg-options \
  optimizer.lr=0.0001 \
  data.samples_per_gpu=4 \
  data.workers_per_gpu=2 \
  'data.train.pipeline.2.data_aug_conf.reisze=[720]' \
  log_config.interval=10 \
  >>"${log_file}" 2>&1
