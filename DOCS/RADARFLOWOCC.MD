# RadarFlowOcc integration

This integration adds a minimal, ablatable radar path to the official
Drive-OccWorld fine-grained GMO baseline. The original camera-only config is
unchanged. Radar is enabled only by:

```text
projects/configs/radarflowocc/action_condition_GMO_radar.py
```

The staged motion variants are:

```text
action_condition_GMO_radar.py        M0 radar fusion
action_condition_GMO_radar_m1.py     M0 + measured Doppler BEV advection
action_condition_GMO_radar_m2.py     M0 + radial-flow auxiliary loss
action_condition_GMO_radar_m1_m2.py  M0 + M1 + M2
```

## Data path

The server repository uses these small symlinks into the approved storage
root:

```text
data/nuscenes          -> /storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc
data/nuScenes-Occupancy -> /storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1
data/can_bus           -> /storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_can_bus/can_bus
data/radar_bev_cache   -> /storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/radar_bev_cache
```

Large work directories and checkpoints must remain under:

```text
/storage/data/metaiot_data/huayiming/RadarFlowOcc/
```

## Radar representation

For the current keyframe, the loader aggregates five sweeps from all five
nuScenes radars. Positions, raw velocities, and ego-motion-compensated
velocities are transformed into the current `LIDAR_TOP` frame. The fixed
`8 x 200 x 200` raster contains:

1. return presence;
2. log-scaled return count;
3. mean height;
4. mean radar cross section;
5. mean compensated x velocity;
6. mean compensated y velocity;
7. mean compensated speed;
8. mean sweep time lag.

A 58,368-parameter convolutional encoder maps the raster to 256-dimensional
BEV tokens. Its final projection is zero-initialized, so the first forward is
exactly the camera-only path; subsequent optimizer steps learn a radar
residual before the feature enters the world-model memory queue.

## M1: measured-motion rollout prior

M1 reads compensated velocity channels 5/6 (zero-based indices 4/5), forward
splats the sparse source-cell velocities to their future locations, and uses a
differentiable `grid_sample` backwarp to advect the measured current BEV. The
prior is then sampled through Drive-OccWorld's future-to-reference ego-pose
transform and added at the future decoder query. A channel-wise `tanh` gate is
initialized to exactly zero, so the initial prediction is bit-identical to M0
while the gate still receives gradient.

## M2: radial-only flow self-supervision

M2 appends three cache channels: compensated measured radial velocity and the
x/y components of the radar-sensor-to-return unit vector. It reuses the
existing `WorldHeadV1` voxel-flow predictor on the current BEV, converts the
legacy backward 0.8 m cell displacement to forward m/s, samples the predicted
flow at the radar return height, and applies L1 only to the line-of-sight
projection. No target is applied to tangential velocity. `turn_on_flow` stays
false, so M2 does not require Cam4DOcc flow labels or instance-flow loading.

## Radar BEV cache

The radar config uses a read-only float32 cache. Populate the training split
before training so every raw 25-sweep aggregation is computed once:

```bash
nice -n 10 ionice -c 2 -n 7 \
  python tools/precompute_radar_bev.py \
  --splits train --workers 4 --verify 5
```

Each cache file is written atomically and indexed by the nuScenes sample
token. The tool verifies selected cached tensors against a fresh online
computation with exact array equality and writes `manifest.json` after the
split is complete. Precompute `val` separately before evaluation.

M2 uses a separate 11-channel cache and never overwrites the M0/M1 cache:

```bash
nice -n 10 ionice -c 2 -n 7 \
  python tools/precompute_radar_bev.py \
  --config projects/configs/radarflowocc/action_condition_GMO_radar_m1_m2.py \
  --splits train --workers 4 --verify 5
```

The first eight channels of each v1 file are exactly equal to its v0 file.
This allows the unchanged eight-channel radar encoder and M0 checkpoints to be
used with M2; only channels 9--11 are consumed by the auxiliary loss.

## Required environment

The CUDA 11.1 extension build is incompatible with the server's default GCC
11. Use GCC 10 for every import, train, or evaluation command:

```bash
cd /home/huayiming/Workspace/RadarFlowOcc/code/Drive-OccWorld
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export MAX_JOBS=4
export PYTHONPATH="$PWD"
```

Data-only integration check:

```bash
nice -n 10 ionice -c 2 -n 7 \
  /home/huayiming/Workspace/RadarFlowOcc/envs/hym_driveocc_py38/bin/python \
  tools/radarflowocc_data_smoke.py
```

## Verified result

On 2026-08-23, one real sample completed forward, backward, and an AdamW
optimizer step on GPU 1:

```text
total loss:                 40.708084
radar projection grad L1:  136.779831
post-step residual max:      0.064621
peak allocated GPU memory:  32898.3 MiB
status:                     PASS
```

The same sample also passed collation with the official
`workers_per_gpu=4` DataLoader setting in a non-distributed process. Note that
Linux uses `fork` in that test, while MMCV 1.4 switches distributed training
to `spawn`. Each spawned worker serializes its own large nuScenes dataset
object, so the official four workers per GPU are not safe on the shared
server.

The complete log is stored at:

```text
/storage/data/metaiot_data/huayiming/RadarFlowOcc/experiments/smoke/radar_one_step_20260823_v1.log
```

## Full training

Do not start a distributed run without checking both `nvidia-smi` and host
memory. The official configuration expects eight GPUs and randomly trains at
heights from 720 through 1080. In FP32, the larger resolutions do not fit a
46-GiB L40S during backward. For an initial L40S run, keep the official lowest
resolution and use at most one spawned DataLoader worker per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nice -n 10 ionice -c 2 -n 7 \
  bash tools/dist_train.sh \
  projects/configs/radarflowocc/action_condition_GMO_radar.py 8 \
  --work-dir /storage/data/metaiot_data/huayiming/RadarFlowOcc/work_dirs/gmo_radar_v0_l40s720 \
  --seed 0 \
  --cfg-options \
    data.workers_per_gpu=1 \
    data.train.pipeline.2.data_aug_conf.reisze='[720]'
```

This hardware adaptation changes only input augmentation and input-pipeline
parallelism. Use the same resize setting for the camera-only control. Restoring
the full 720--1080 multiscale protocol requires additional memory reduction
(for example, a separately validated mixed-precision or checkpointing change)
or GPUs with more memory.

Do not restore the upstream value of four on an eight-GPU run: spawned workers
copy the large dataset object and can exhaust host memory. The cached four-GPU
profile uses two workers per GPU, which keeps the worker count at eight while
removing online radar sweep aggregation.

If only GPUs 1-7 are available, use seven processes and linearly scale the
learning rate from `2e-4` to `1.75e-4`:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  nice -n 10 ionice -c 2 -n 7 \
  bash tools/dist_train.sh \
  projects/configs/radarflowocc/action_condition_GMO_radar.py 7 \
  --work-dir /storage/data/metaiot_data/huayiming/RadarFlowOcc/work_dirs/gmo_radar_v0_7gpu \
  --cfg-options \
    optimizer.lr=0.000175 \
    data.workers_per_gpu=1 \
    data.train.pipeline.2.data_aug_conf.reisze='[720]'
```

For a four-GPU shared-server run, use GPUs 0-3 and linearly scale the learning
rate to `1e-4` while retaining the validated L40S input settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nice -n 10 ionice -c 2 -n 7 \
  bash tools/dist_train.sh \
  projects/configs/radarflowocc/action_condition_GMO_radar.py 4 \
  --work-dir /storage/data/metaiot_data/huayiming/RadarFlowOcc/work_dirs/gmo_radar_v0_l40s720_cached_w2_4g_seed0 \
  --seed 0 \
  --cfg-options \
    optimizer.lr=0.0001 \
    data.workers_per_gpu=2 \
    data.train.pipeline.2.data_aug_conf.reisze='[720]'
```

The upstream repository publishes only the FCOS3D initialization checkpoint,
not a complete Drive-OccWorld `epoch_24.pth`. Published evaluation therefore
requires training the baseline and radar variants under the same protocol.

For a short M0-to-M1/M2 continuation test, use `load_from=<M0 checkpoint>`
rather than `resume_from`: the new gate/flow-head parameters are absent from
the M0 optimizer state. Final ablations should either train all variants from
the same FCOS3D initialization or give an M0 continuation control the same
number of extra epochs.
