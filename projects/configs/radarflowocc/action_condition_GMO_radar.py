_base_ = ['../fine_grained/action_condition_GMO.py']

# Current-frame radar is aggregated from all five nuScenes radar sensors. Each
# sensor contributes five sweeps, transformed into the current LIDAR_TOP frame.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True)

radar_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_h=200,
    bev_w=200,
    nsweeps=5,
    min_distance=1.0,
    count_clip=32.0,
    velocity_norm=20.0,
    rcs_norm=50.0,
    time_lag_norm=0.5,
    cache_dir='data/radar_bev_cache/nuscenes_trainval_5sweeps_200x200_v0_float32',
    cache_readonly=True)

model = dict(
    radar_encoder=dict(
        in_channels=8,
        hidden_channels=64,
        out_channels=256,
        num_groups=8,
        zero_init_residual=True))

data = dict(
    workers_per_gpu=2,
    train=dict(modality=input_modality, radar_cfg=radar_cfg),
    val=dict(modality=input_modality, radar_cfg=radar_cfg),
    test=dict(modality=input_modality, radar_cfg=radar_cfg))
