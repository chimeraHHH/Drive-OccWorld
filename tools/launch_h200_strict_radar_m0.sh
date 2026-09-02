#!/usr/bin/env bash
set -euo pipefail

# H200 x2 independent strict-control run for the Radar M0 experiment.
# Global batch and samples/epoch are held constant:
#   L40S: 4 ranks x 1 sample/rank = 4
#   H200: 2 ranks x 2 samples/rank = 4

workspace=/home/wangning/Workspace/RadarFlowOcc
code_dir=${workspace}/code/Drive-OccWorld-m1m2
env_dir=/home/wangning/.conda/envs/driveocc_h200
work_dir=${workspace}/work_dirs/gmo_radar_v0_h200720_cached_w2_2g_b2_seed0_scratch
pretrained_checkpoint=${code_dir}/pretrained/r101_dcn_fcos3d_pretrain.pth
extension_cache=${workspace}/cache/torch_extensions_m1m2
log_file=${work_dir}/strict_h200_2g_b2_scratch.log

test -x "${env_dir}/bin/python"
test -r "${pretrained_checkpoint}"
test -d "${code_dir}/data/radar_bev_cache/nuscenes_trainval_5sweeps_200x200_v0_float32"
mkdir -p "${work_dir}" "${extension_cache}"

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
  --master_port=16854 \
  tools/train.py \
  projects/configs/radarflowocc/action_condition_GMO_radar.py \
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
