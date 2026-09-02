_base_ = ['./action_condition_GMO_radar.py']

# M1: Doppler-guided future BEV advection.  The velocity channels are zero-based
# indices 4/5, corresponding to the documented fifth/sixth radar BEV channels.
# The channel-wise gate is initialized to zero, so loading an M0 checkpoint is
# functionally identical before the gate starts learning.
model = dict(
    doppler_advection=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        velocity_channels=(4, 5),
        presence_channel=0,
        radar_velocity_norm=20.0,
        time_step=0.5,
        min_dynamic_speed=0.5,
        max_dynamic_speed=30.0,
        padding_mode='zeros',
        gate_init=0.0))
