_base_ = ['./action_condition_GMO_radar_m1_m2.py']

# Ablation control: keep the M2 radial-flow auxiliary loss and 11-channel
# cache, but remove the M1 rollout prior.
model = dict(doppler_advection=None)
