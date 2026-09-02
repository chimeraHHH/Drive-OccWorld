#!/usr/bin/env bash
set -euo pipefail

# H200 x2 strict-control run for RadarFlowOcc M1+M2, from Epoch 1.
# Global batch and samples/epoch match the control experiment:
#   L40S: 4 ranks x 1 sample/rank = 4
#   H200: 2 ranks x 2 samples/rank = 4

workspace=/home/wangning/Workspace/RadarFlowOcc
code_dir=${workspace}/code/Drive-OccWorld-m1m2
env_dir=/home/wangning/.conda/envs/driveocc_h200
work_dir=${workspace}/work_dirs/gmo_radar_m1_m2_v1_h200720_cached_w2_2g_b2_seed0_scratch
pretrained_checkpoint=${code_dir}/pretrained/r101_dcn_fcos3d_pretrain.pth
radial_cache=${code_dir}/data/radar_bev_cache/nuscenes_trainval_5sweeps_200x200_v1_radial_float32
extension_cache=${workspace}/cache/torch_extensions_m1m2
log_file=${work_dir}/strict_h200_2g_b2_m1_m2_scratch.log

test -x "${env_dir}/bin/python"
test -r "${pretrained_checkpoint}"
test -d "${radial_cache}"
radial_sample=$(find "${radial_cache}" -maxdepth 1 -type f -name '*.npy' -print -quit)
test -n "${radial_sample}"
test -r "${radial_sample}"
mkdir -p "${work_dir}" "${extension_cache}"

# Never silently continue an older M1+M2 run.
if find "${work_dir}" -maxdepth 1 -type f -name 'epoch_*.pth' -print -quit | grep -q .; then
  echo "Refusing to reuse a work directory containing epoch checkpoints: ${work_dir}" >&2
  exit 1
fi

export PATH="${env_dir}/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR="${extension_cache}"
export MAX_JOBS=2
export PYTHONPATH="${code_dir}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${code_dir}"

exec nice -n 10 ionice -c 2 -n 7 \
  "${env_dir}/bin/python" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=16855 \
  tools/train.py \
  projects/configs/radarflowocc/action_condition_GMO_radar_m1_m2.py \
  --launcher pytorch \
  --work-dir "${work_dir}" \
  --seed 0 \
  --deterministic \
  --no-validate \
  --cfg-options \
  optimizer.lr=0.0001 \
  data.samples_per_gpu=2 \
  data.workers_per_gpu=2 \
  'data.train.pipeline.2.data_aug_conf.reisze=[720]' \
  log_config.interval=10 \
  >>"${log_file}" 2>&1
