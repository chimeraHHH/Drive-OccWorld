from functools import reduce
import os
from os import path as osp
import tempfile

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


class NuScenesRadarBEV:
    """Load and rasterize multi-sweep nuScenes radar in the current LiDAR frame.

    The nuScenes devkit transforms radar positions in ``from_file_multisweep``
    but leaves velocity attributes in the source sensor frame. This loader
    explicitly rotates both raw and ego-motion-compensated velocities into the
    current ``LIDAR_TOP`` frame before rasterization.

    The base output channels are:
      0. return presence
      1. log-scaled return count
      2. mean height
      3. mean radar cross section (RCS)
      4. mean compensated x velocity
      5. mean compensated y velocity
      6. mean compensated speed
      7. mean sweep time lag

    With ``include_radial_features=True`` three channels used by the M2
    self-supervised loss are appended:
      8. mean ego-motion-compensated radial velocity
      9. mean sensor-to-return radial unit vector x
     10. mean sensor-to-return radial unit vector y

    Projecting the compensated velocity onto the sensor line of sight keeps
    the directly observed Doppler component (plus the known ego-motion
    correction) and discards the radar tracker's less reliable tangential
    component.
    """

    RADAR_CHANNELS = (
        'RADAR_FRONT',
        'RADAR_FRONT_LEFT',
        'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT',
        'RADAR_BACK_RIGHT',
    )
    NUM_FEATURES = 8
    NUM_FEATURES_WITH_RADIAL = 11

    def __init__(self,
                 nusc,
                 point_cloud_range,
                 bev_h,
                 bev_w,
                 nsweeps=5,
                 min_distance=1.0,
                 count_clip=32.0,
                 velocity_norm=20.0,
                 rcs_norm=50.0,
                 time_lag_norm=0.5,
                 radar_channels=None,
                 cache_dir=None,
                 cache_readonly=False,
                 include_radial_features=False):
        self.nusc = nusc
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        if self.point_cloud_range.shape != (6,):
            raise ValueError('point_cloud_range must contain 6 values')
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.nsweeps = int(nsweeps)
        self.min_distance = float(min_distance)
        self.count_clip = float(count_clip)
        self.velocity_norm = float(velocity_norm)
        self.rcs_norm = float(rcs_norm)
        self.time_lag_norm = float(time_lag_norm)
        self.radar_channels = tuple(radar_channels or self.RADAR_CHANNELS)
        self.cache_dir = osp.abspath(cache_dir) if cache_dir else None
        self.cache_readonly = bool(cache_readonly)
        self.include_radial_features = bool(include_radial_features)
        self.num_features = (
            self.NUM_FEATURES_WITH_RADIAL
            if self.include_radial_features else self.NUM_FEATURES)

        if self.bev_h <= 0 or self.bev_w <= 0 or self.nsweeps <= 0:
            raise ValueError('bev_h, bev_w and nsweeps must be positive')
        if self.cache_dir is not None and not self.cache_readonly:
            os.makedirs(self.cache_dir, exist_ok=True)

    def cache_path(self, sample_token):
        if self.cache_dir is None:
            return None
        return osp.join(self.cache_dir, '{}.npy'.format(sample_token))

    def _load_cache(self, sample_token):
        cache_path = self.cache_path(sample_token)
        if cache_path is None or not osp.isfile(cache_path):
            return None
        bev = np.load(cache_path, allow_pickle=False)
        expected_shape = (self.num_features, self.bev_h, self.bev_w)
        if bev.shape != expected_shape or bev.dtype != np.float32:
            raise ValueError(
                'Invalid radar cache {}: shape={}, dtype={}, expected '
                'shape={} and float32'.format(
                    cache_path, bev.shape, bev.dtype, expected_shape))
        return bev

    def _write_cache(self, sample_token, bev):
        cache_path = self.cache_path(sample_token)
        if cache_path is None:
            return
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb', dir=self.cache_dir,
                    prefix='.{}.'.format(sample_token), suffix='.npy',
                    delete=False) as temp_file:
                temp_path = temp_file.name
                np.save(temp_file, bev, allow_pickle=False)
            os.replace(temp_path, cache_path)
        finally:
            if temp_path is not None and osp.exists(temp_path):
                os.unlink(temp_path)

    def _load_channel(self, sample_rec, radar_channel):
        """Aggregate one radar channel and preserve velocity-frame semantics."""
        ref_sd_rec = self.nusc.get(
            'sample_data', sample_rec['data']['LIDAR_TOP'])
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = self.nusc.get(
            'calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']),
            inverse=True)
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
            inverse=True)

        current_sd_rec = self.nusc.get(
            'sample_data', sample_rec['data'][radar_channel])
        point_chunks = []
        time_chunks = []
        radial_velocity_chunks = []
        radial_direction_chunks = []
        for _ in range(self.nsweeps):
            radar = RadarPointCloud.from_file(
                osp.join(self.nusc.dataroot, current_sd_rec['filename']))
            radar.remove_close(self.min_distance)

            current_pose_rec = self.nusc.get(
                'ego_pose', current_sd_rec['ego_pose_token'])
            current_cs_rec = self.nusc.get(
                'calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            global_from_car = transform_matrix(
                current_pose_rec['translation'],
                Quaternion(current_pose_rec['rotation']), inverse=False)
            car_from_current = transform_matrix(
                current_cs_rec['translation'],
                Quaternion(current_cs_rec['rotation']), inverse=False)
            current_to_ref = reduce(
                np.dot,
                [ref_from_car, car_from_global,
                 global_from_car, car_from_current])

            # PointCloud.transform only transforms xyz. Rotate the two radar
            # velocity vectors explicitly; translation must never affect them.
            rotation = current_to_ref[:3, :3]
            for first_idx in (6, 8):
                velocity = np.vstack((
                    radar.points[first_idx:first_idx + 2],
                    np.zeros((1, radar.nbr_points()), dtype=radar.points.dtype)))
                radar.points[first_idx:first_idx + 2] = (
                    rotation @ velocity)[:2]
            radar.transform(current_to_ref)

            # The line of sight must start at the radar sensor origin for this
            # sweep, not at the reference LiDAR/ego origin.  In the reference
            # LiDAR frame the sensor origin is the transformed local origin.
            sensor_origin_xy = current_to_ref[:2, 3:4]
            # Rows 8/9 are the ego-motion-compensated velocity.  Its radial
            # projection retains the measured Doppler constraint while making
            # it compatible with the world-frame flow predicted by the model.
            radial_velocity, radial_direction = self._radial_measurement(
                radar.points[:2], radar.points[8:10], sensor_origin_xy)

            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            point_chunks.append(radar.points)
            time_chunks.append(np.full(
                (radar.nbr_points(),), time_lag, dtype=np.float32))
            radial_velocity_chunks.append(radial_velocity.astype(np.float32))
            radial_direction_chunks.append(
                radial_direction.astype(np.float32))

            if not current_sd_rec['prev']:
                break
            current_sd_rec = self.nusc.get(
                'sample_data', current_sd_rec['prev'])

        if not point_chunks:
            return (np.zeros((RadarPointCloud.nbr_dims(), 0), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((2, 0), dtype=np.float32))
        return (np.concatenate(point_chunks, axis=1).astype(np.float32),
                np.concatenate(time_chunks),
                np.concatenate(radial_velocity_chunks),
                np.concatenate(radial_direction_chunks, axis=1))

    @staticmethod
    def _cell_mean(indices, values, cell_counts, nonempty):
        sums = np.bincount(
            indices, weights=values,
            minlength=cell_counts.size).astype(np.float32)
        means = np.zeros_like(cell_counts, dtype=np.float32)
        means[nonempty] = sums[nonempty] / cell_counts[nonempty]
        return means

    @staticmethod
    def _radial_measurement(points_xy, compensated_velocity_xy,
                            sensor_origin_xy):
        """Return compensated Doppler scalar and line-of-sight direction."""
        radial_direction = points_xy - sensor_origin_xy
        radial_distance = np.linalg.norm(
            radial_direction, axis=0, keepdims=True)
        valid = radial_distance > 1e-6
        radial_direction = np.divide(
            radial_direction, np.maximum(radial_distance, 1e-6))
        radial_direction = radial_direction * valid.astype(
            radial_direction.dtype)
        radial_velocity = np.sum(
            compensated_velocity_xy * radial_direction, axis=0)
        return radial_velocity, radial_direction

    def compute(self, sample_token):
        """Compute a radar BEV from raw sweeps without consulting the cache."""
        sample_rec = self.nusc.get('sample', sample_token)
        point_chunks = []
        time_chunks = []
        radial_velocity_chunks = []
        radial_direction_chunks = []
        for radar_channel in self.radar_channels:
            points, times, radial_velocity, radial_direction = (
                self._load_channel(sample_rec, radar_channel))
            point_chunks.append(points)
            time_chunks.append(times)
            radial_velocity_chunks.append(radial_velocity)
            radial_direction_chunks.append(radial_direction)

        points = np.concatenate(point_chunks, axis=1)
        time_lags = np.concatenate(time_chunks)
        radial_velocity = np.concatenate(radial_velocity_chunks)
        radial_direction = np.concatenate(radial_direction_chunks, axis=1)
        bev = np.zeros(
            (self.num_features, self.bev_h, self.bev_w), dtype=np.float32)
        if points.shape[1] == 0:
            return bev

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        finite = np.isfinite(points[[0, 1, 2, 5, 8, 9]]).all(axis=0)
        if self.include_radial_features:
            finite = (
                finite & np.isfinite(radial_velocity) &
                np.isfinite(radial_direction).all(axis=0))
        inside = (
            finite &
            (points[0] >= x_min) & (points[0] < x_max) &
            (points[1] >= y_min) & (points[1] < y_max) &
            (points[2] >= z_min) & (points[2] < z_max))
        if not inside.any():
            return bev

        points = points[:, inside]
        time_lags = time_lags[inside]
        radial_velocity = radial_velocity[inside]
        radial_direction = radial_direction[:, inside]
        x_index = np.floor(
            (points[0] - x_min) / (x_max - x_min) * self.bev_w
        ).astype(np.int64)
        y_index = np.floor(
            (points[1] - y_min) / (y_max - y_min) * self.bev_h
        ).astype(np.int64)
        linear_index = y_index * self.bev_w + x_index
        num_cells = self.bev_h * self.bev_w
        counts = np.bincount(linear_index, minlength=num_cells).astype(np.float32)
        nonempty = counts > 0

        mean_z = self._cell_mean(linear_index, points[2], counts, nonempty)
        mean_rcs = self._cell_mean(linear_index, points[5], counts, nonempty)
        mean_vx = self._cell_mean(linear_index, points[8], counts, nonempty)
        mean_vy = self._cell_mean(linear_index, points[9], counts, nonempty)
        speed = np.sqrt(points[8] ** 2 + points[9] ** 2)
        mean_speed = self._cell_mean(linear_index, speed, counts, nonempty)
        mean_lag = self._cell_mean(linear_index, time_lags, counts, nonempty)

        z_norm = max(abs(float(z_min)), abs(float(z_max)), 1.0)
        features = [
            nonempty.astype(np.float32),
            np.log1p(np.minimum(counts, self.count_clip)) /
            np.log1p(self.count_clip),
            np.clip(mean_z / z_norm, -1.0, 1.0),
            np.clip(mean_rcs / self.rcs_norm, -1.0, 1.0),
            np.clip(mean_vx / self.velocity_norm, -1.0, 1.0),
            np.clip(mean_vy / self.velocity_norm, -1.0, 1.0),
            np.clip(mean_speed / self.velocity_norm, 0.0, 1.0),
            np.clip(mean_lag / self.time_lag_norm, 0.0, 1.0),
        ]

        if self.include_radial_features:
            mean_radial_velocity = self._cell_mean(
                linear_index, radial_velocity, counts, nonempty)
            mean_radial_x = self._cell_mean(
                linear_index, radial_direction[0], counts, nonempty)
            mean_radial_y = self._cell_mean(
                linear_index, radial_direction[1], counts, nonempty)
            mean_direction_norm = np.sqrt(
                mean_radial_x ** 2 + mean_radial_y ** 2)
            valid_direction = mean_direction_norm > 1e-6
            mean_radial_x[valid_direction] /= mean_direction_norm[valid_direction]
            mean_radial_y[valid_direction] /= mean_direction_norm[valid_direction]
            features.extend((
                np.clip(
                    mean_radial_velocity / self.velocity_norm, -1.0, 1.0),
                mean_radial_x,
                mean_radial_y,
            ))

        features = np.stack(features, axis=0)
        return features.reshape(
            self.num_features, self.bev_h, self.bev_w).astype(np.float32)

    def __call__(self, sample_token):
        cached = self._load_cache(sample_token)
        if cached is not None:
            return cached
        if self.cache_dir is not None and self.cache_readonly:
            raise FileNotFoundError(
                'Radar BEV cache is missing for sample {}: {}'.format(
                    sample_token, self.cache_path(sample_token)))
        bev = self.compute(sample_token)
        self._write_cache(sample_token, bev)
        return bev
