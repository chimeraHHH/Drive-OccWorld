_base_ = ['./action_condition_GMO_radar_m1.py']

# M2 cache format appends measured radial velocity and the radar-sensor line
# of sight to the original eight M0/M1 channels. The camera-radar encoder still
# consumes only the first eight channels, preserving checkpoint compatibility.
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
    include_radial_features=True,
    cache_dir=(
        'data/radar_bev_cache/'
        'nuscenes_trainval_5sweeps_200x200_v1_radial_float32'),
    cache_readonly=True)

model = dict(
    # Keep the legacy turn_on_flow=False inherited from GMO: M2 constructs the
    # existing WorldHeadV1 flow predictor but does not request Cam4DOcc flow GT.
    doppler_flow_loss=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_h=200,
        bev_w=200,
        radial_velocity_channel=8,
        radial_direction_channels=(9, 10),
        presence_channel=0,
        height_channel=2,
        time_lag_channel=7,
        radar_velocity_norm=20.0,
        radar_height_norm=5.0,
        radar_time_lag_norm=0.5,
        time_decay_tau=0.25,
        # Existing Cam4DOcc flow targets use backward displacement in 0.8 m
        # cells between adjacent 0.5 s frames.
        flow_cell_size=(0.8, 0.8),
        time_step=0.5,
        backward_flow=True,
        max_abs_radial_velocity=30.0,
        loss_weight=0.1))

data = dict(
    train=dict(radar_cfg=radar_cfg),
    val=dict(radar_cfg=radar_cfg),
    test=dict(radar_cfg=radar_cfg))
