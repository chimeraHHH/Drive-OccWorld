_base_ = ['./action_condition_GMO_radar_m1.py']

# Future camera images are discarded by NuScenesWorldDatasetV1.union2one.
# Keep their poses/actions and replay image augmentation RNG draws without
# decoding/augmenting those pixels. All model/training settings are inherited.
data = dict(train=dict(future_metadata_only=True))
