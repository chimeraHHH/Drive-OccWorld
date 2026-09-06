_base_ = ['./action_condition_GMO_radar_m1_fast_input.py']

# M3 research prototype. Future ego transforms/actions remain given, as in
# Drive-OccWorld. No future radar or future camera pixels are motion inputs.
# M3 caches are deliberately separate from M0/M1/M2 clipped raster caches.
radar_observation_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_h=200, bev_w=200, nsweeps=1, max_returns=4096,
    max_time_lag=0.15, cache_dir=None, cache_readonly=False)

model = dict(
    turn_on_flow=False, turn_on_plan=False,
    doppler_advection=None, doppler_flow_loss=None,
    scientific_eval=True,
    doppler_nll_weight=0.05,
    doppler_posterior=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_h=200, bev_w=200, embed_dims=256,
        hidden_dims=64, transport_dims=32, time_step=0.5,
        sensor_std=1.5, prior_std_bounds=(0.5, 15.0),
        initial_prior_std=5.0, max_time_lag=0.15,
        holdout_fraction=0.2, gate_init=0.0,
        transport_mode='sigma', use_doppler_update=True))

data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(radar_cfg=None, radar_observation_cfg=radar_observation_cfg),
    val=dict(radar_cfg=None, radar_observation_cfg=radar_observation_cfg),
    test=dict(radar_cfg=None, radar_observation_cfg=radar_observation_cfg))

# Runtime paths, 720-only augmentation, pretrained initialization and the
# common training budget must be resolved with tools/m3_prepare_config.py.
# Do not resume an M1 optimizer into the changed M3 architecture.
resume_from = None
work_dir = 'work_dirs/gmo_radar_m3_observable_transport'
