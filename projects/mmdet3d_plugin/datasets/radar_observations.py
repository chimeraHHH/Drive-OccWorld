"""Current-frame radar returns for M3, before any velocity rasterization.

The radial scalar is the projection of nuScenes' ego-motion-compensated
velocity onto the sensor-to-return direction.  It is not a raw radar spectrum
measurement.  Each return retains its own direction and *unclipped* scalar.
Support/holdout selection belongs to the model and is never stored here.
"""

import hashlib
import json
import os
from os import path as osp
import re
import tempfile

import numpy as np


class NuScenesRadarObservations:
    """Load padded ``[N, 9]`` observations in current ``LIDAR_TOP`` coordinates.

    Columns are ``x, y, z, rcs, ux, uy, radial_velocity, lag_seconds, valid``.
    Only causal returns inside ``point_cloud_range`` are retained.  The source
    loader removes close returns in the source sensor frame; no distance test
    about the reference LiDAR origin is substituted for that operation.

    The default uses one radar sweep, with at most 0.15 s timestamp lag.
    Positions are ego aligned, not advanced along a target trajectory.  A cap
    is needed for stacked DataContainers; overflow is sampled by return index
    with a stable token-dependent seed, independently of measured velocity.
    """

    NUM_FEATURES = 9
    CACHE_VERSION = 1
    RADAR_CHANNELS = (
        'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT')

    def __init__(self, nusc, point_cloud_range, bev_h, bev_w,
                 nsweeps=1, min_distance=1.0, max_returns=4096,
                 max_time_lag=0.15, radar_channels=None, subsample_seed=0,
                 direction_tolerance=1e-3, cache_dir=None,
                 cache_readonly=False):
        self.nusc = nusc
        self.point_cloud_range = np.asarray(
            point_cloud_range, dtype=np.float64)
        if (self.point_cloud_range.shape != (6,) or
                not np.isfinite(self.point_cloud_range).all() or
                not np.all(self.point_cloud_range[3:] >
                           self.point_cloud_range[:3])):
            raise ValueError('point_cloud_range must have six finite, '
                             'strictly increasing axis bounds')
        for name, value in (('bev_h', bev_h), ('bev_w', bev_w),
                            ('nsweeps', nsweeps),
                            ('max_returns', max_returns)):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError('{} must be a positive integer'.format(name))
            setattr(self, name, int(value))
        self.min_distance = float(min_distance)
        self.max_time_lag = float(max_time_lag)
        self.direction_tolerance = float(direction_tolerance)
        if (not np.isfinite(self.min_distance) or self.min_distance < 0 or
                not np.isfinite(self.max_time_lag) or self.max_time_lag < 0 or
                not np.isfinite(self.direction_tolerance) or
                not 0 < self.direction_tolerance < 0.1):
            raise ValueError('Invalid radar distance, lag, or direction filter')
        self.radar_channels = tuple(
            self.RADAR_CHANNELS if radar_channels is None else radar_channels)
        if (not self.radar_channels or
                len(set(self.radar_channels)) != len(self.radar_channels) or
                any(c not in self.RADAR_CHANNELS for c in self.radar_channels)):
            raise ValueError('radar_channels must be a nonempty unique '
                             'subset of the nuScenes radar channels')
        if int(subsample_seed) != subsample_seed:
            raise ValueError('subsample_seed must be an integer')
        self.subsample_seed = int(subsample_seed)
        self.cache_dir = osp.abspath(cache_dir) if cache_dir else None
        self.cache_readonly = bool(cache_readonly)
        if self.cache_readonly and self.cache_dir is None:
            raise ValueError('cache_readonly requires cache_dir')
        self._source_loader = None
        self._manifest = dict(
            schema='radarflowocc_m3_per_return', version=self.CACHE_VERSION,
            columns=['x', 'y', 'z', 'rcs', 'ux', 'uy', 'radial_velocity',
                     'lag_seconds', 'valid'],
            source='NuScenesRadarBEV._load_channel:v1',
            radial_source='projected_nuscenes_compensated_velocity',
            data_version=str(getattr(nusc, 'version', 'unknown')),
            data_root=osp.realpath(getattr(nusc, 'dataroot', '.')),
            point_cloud_range=self.point_cloud_range.tolist(),
            bev_h=self.bev_h, bev_w=self.bev_w, nsweeps=self.nsweeps,
            min_distance=self.min_distance, max_returns=self.max_returns,
            min_time_lag=0.0, max_time_lag=self.max_time_lag,
            direction_tolerance=self.direction_tolerance,
            radar_channels=list(self.radar_channels),
            subsample_seed=self.subsample_seed,
            subsampling='token_blake2b_seeded_randomstate_return_indices_v1')
        self._manifest_json = json.dumps(
            self._manifest, sort_keys=True, separators=(',', ':'))
        if self.cache_dir is not None and not self.cache_readonly:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def manifest(self):
        """Return a copy so callers cannot alter subsequent cache identity."""
        return json.loads(self._manifest_json)

    @staticmethod
    def _validate_token(sample_token):
        if (not isinstance(sample_token, str) or
                re.fullmatch(r'[A-Za-z0-9_-]+', sample_token) is None):
            raise ValueError('sample_token must be a nonempty safe identifier')

    def cache_path(self, sample_token):
        self._validate_token(sample_token)
        if self.cache_dir is None:
            return None
        return osp.join(self.cache_dir, '{}.npz'.format(sample_token))

    def _validate_observations(self, observations):
        if (observations.shape != (self.max_returns, self.NUM_FEATURES) or
                observations.dtype != np.float32):
            raise ValueError('Observations must have shape ({}, {}) and '
                             'dtype float32'.format(
                                 self.max_returns, self.NUM_FEATURES))
        if not np.isfinite(observations).all():
            raise ValueError('Observations contain nonfinite values')
        valid = observations[:, 8]
        if not np.isin(valid, (0.0, 1.0)).all():
            raise ValueError('Observation valid flags must be binary')
        selected = observations[valid == 1]
        if np.any(observations[valid == 0] != 0):
            raise ValueError('Invalid observation padding must be all zero')
        # Packing puts all real returns first.  Enforce the same schema when
        # reading an externally generated cache instead of silently accepting
        # a different mask convention.
        if np.any(np.diff(valid) > 0):
            raise ValueError('Observation padding must follow all valid rows')
        if not len(selected):
            return
        if (not np.all(selected[:, :3] >= self.point_cloud_range[:3]) or
                not np.all(selected[:, :3] < self.point_cloud_range[3:]) or
                not np.all(selected[:, 7] >= 0.0) or
                not np.all(selected[:, 7] <= self.max_time_lag) or
                not np.all(np.abs(np.linalg.norm(
                    selected[:, 4:6], axis=1) - 1.0) <=
                    self.direction_tolerance)):
            raise ValueError('Observation cache violates spatial, causal, '
                             'or unit-direction filters')

    def _load_cache(self, sample_token):
        cache_path = self.cache_path(sample_token)
        if cache_path is None or not osp.isfile(cache_path):
            return None
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != {'observations', 'manifest', 'sample_token'}:
                raise ValueError('Invalid M3 observation cache fields: {}'.format(
                    cache_path))
            if (payload['manifest'].shape != () or
                    str(payload['manifest'].item()) != self._manifest_json or
                    payload['sample_token'].shape != () or
                    str(payload['sample_token'].item()) != sample_token):
                raise ValueError('M3 observation cache identity mismatch: '
                                 '{}'.format(cache_path))
            observations = payload['observations']
        self._validate_observations(observations)
        return observations

    def _write_cache(self, sample_token, observations):
        self._validate_observations(observations)
        cache_path = self.cache_path(sample_token)
        if cache_path is None:
            return
        if self.cache_readonly:
            raise RuntimeError('Cannot write a readonly radar observation cache')
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb', dir=self.cache_dir,
                    prefix='.{}.'.format(sample_token), suffix='.npz',
                    delete=False) as temp_file:
                temp_path = temp_file.name
                np.savez_compressed(
                    temp_file, observations=observations,
                    manifest=np.asarray(self._manifest_json),
                    sample_token=np.asarray(sample_token))
            os.replace(temp_path, cache_path)
        finally:
            if temp_path is not None and osp.exists(temp_path):
                os.unlink(temp_path)

    def _load_channel(self, sample_rec, radar_channel):
        # Keep devkit imports lazy: packing/cache checks need only NumPy, and
        # workers reading a prepared readonly cache need not create a second
        # radar reader.  This delegates exact sensor-origin/frame semantics.
        if self._source_loader is None:
            from .radar_bev import NuScenesRadarBEV
            self._source_loader = NuScenesRadarBEV(
                nusc=self.nusc, point_cloud_range=self.point_cloud_range,
                bev_h=self.bev_h, bev_w=self.bev_w, nsweeps=self.nsweeps,
                min_distance=self.min_distance,
                radar_channels=self.radar_channels)
        return self._source_loader._load_channel(sample_rec, radar_channel)

    def _pack_returns(self, sample_token, points, time_lags,
                      radial_velocity, radial_direction):
        self._validate_token(sample_token)
        points = np.asarray(points)
        time_lags = np.asarray(time_lags)
        radial_velocity = np.asarray(radial_velocity)
        radial_direction = np.asarray(radial_direction)
        if points.ndim != 2 or points.shape[0] < 6:
            raise ValueError('Radar points must have shape [at least 6, N]')
        n = points.shape[1]
        if (time_lags.shape != (n,) or radial_velocity.shape != (n,) or
                radial_direction.shape != (2, n)):
            raise ValueError('Per-return radar arrays have inconsistent shapes')
        # Convert the physical columns together to float32 before filtering so
        # values that overflow, or land on an excluded bound after rounding,
        # cannot silently become malformed cache entries.
        with np.errstate(over='ignore', invalid='ignore'):
            observations = np.column_stack((
                points[:3].T, points[5], radial_direction.T,
                radial_velocity, time_lags, np.ones(n))).astype(np.float32)
            direction_norm = np.linalg.norm(observations[:, 4:6], axis=1)
        valid = np.isfinite(observations).all(axis=1)
        valid &= np.all(observations[:, :3] >= self.point_cloud_range[:3], axis=1)
        valid &= np.all(observations[:, :3] < self.point_cloud_range[3:], axis=1)
        valid &= observations[:, 7] >= 0.0
        valid &= observations[:, 7] <= self.max_time_lag
        valid &= np.abs(direction_norm - 1.0) <= self.direction_tolerance
        selected_indices = np.flatnonzero(valid)
        if selected_indices.size > self.max_returns:
            seed_text = '{}:{}'.format(self.subsample_seed, sample_token)
            seed = int.from_bytes(hashlib.blake2b(
                seed_text.encode('utf-8'), digest_size=4).digest(), 'little')
            random = np.random.RandomState(seed)
            selected_indices = np.sort(random.choice(
                selected_indices, self.max_returns, replace=False))
        selected = observations[selected_indices].copy()
        if selected.shape[0]:
            selected[:, 4:6] /= direction_norm[selected_indices, None]
        packed = np.zeros(
            (self.max_returns, self.NUM_FEATURES), dtype=np.float32)
        packed[:selected.shape[0]] = selected
        self._validate_observations(packed)
        return packed

    def compute(self, sample_token):
        """Read current/past source sweeps without consulting the cache."""
        self._validate_token(sample_token)
        sample_rec = self.nusc.get('sample', sample_token)
        channels = [self._load_channel(sample_rec, channel)
                    for channel in self.radar_channels]
        return self._pack_returns(
            sample_token,
            np.concatenate([c[0] for c in channels], axis=1),
            np.concatenate([c[1] for c in channels]),
            np.concatenate([c[2] for c in channels]),
            np.concatenate([c[3] for c in channels], axis=1))

    def __call__(self, sample_token):
        cached = self._load_cache(sample_token)
        if cached is not None:
            return cached
        if self.cache_readonly:
            raise FileNotFoundError('Missing M3 radar observation cache: '
                                    '{}'.format(self.cache_path(sample_token)))
        observations = self.compute(sample_token)
        self._write_cache(sample_token, observations)
        return observations
