"""Physical-value, causality, and cache-contract checks needing only NumPy."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / 'projects/mmdet3d_plugin/datasets/radar_observations.py')
SPEC = importlib.util.spec_from_file_location('radar_observations', str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NuScenesRadarObservations = MODULE.NuScenesRadarObservations


class RadarObservationsTest(unittest.TestCase):
    def make_loader(self, **overrides):
        kwargs = dict(
            nusc=types.SimpleNamespace(version='v1.0-test', dataroot='/test'),
            point_cloud_range=(-10, -10, -3, 10, 10, 3),
            bev_h=4, bev_w=4, max_returns=8,
            radar_channels=('RADAR_FRONT',))
        kwargs.update(overrides)
        return NuScenesRadarObservations(**kwargs)

    @staticmethod
    def returns(n=1):
        points = np.zeros((18, n), dtype=np.float32)
        points[0] = 1.0
        points[5] = 7.0
        lag = np.full(n, 0.05, dtype=np.float32)
        radial = np.full(n, 3.0, dtype=np.float32)
        direction = np.zeros((2, n), dtype=np.float32)
        direction[0] = 1.0
        return points, lag, radial, direction

    def pack(self, arrays, **kwargs):
        return self.make_loader(**kwargs)._pack_returns('sample', *arrays)

    def test_physical_radial_velocity_is_not_clipped(self):
        arrays = self.returns(2)
        arrays[2][:] = [45.0, -63.0]
        packed = self.pack(arrays)
        np.testing.assert_array_equal(packed[:2, 6], [45.0, -63.0])
        np.testing.assert_array_equal(packed[:2, 8], [1.0, 1.0])
        np.testing.assert_array_equal(packed[2:], np.zeros((6, 9)))
        self.assertEqual(packed.dtype, np.float32)

    def test_per_return_directions_are_not_averaged(self):
        arrays = self.returns(2)
        arrays[3][:] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        arrays[2][:] = 10 / np.sqrt(2)
        packed = self.pack(arrays)
        np.testing.assert_allclose(
            packed[:2, 4:6] @ np.array([10., 0.]), packed[:2, 6], rtol=1e-6)
        self.assertNotEqual(packed[0, 5], packed[1, 5])

    def test_future_old_and_nonfinite_lags_are_dropped(self):
        arrays = self.returns(6)
        arrays[1][:] = [-0.000001, 0.0, 0.15, 0.1501, np.nan, np.inf]
        packed = self.pack(arrays)
        self.assertEqual(int(packed[:, 8].sum()), 2)
        np.testing.assert_allclose(packed[:2, 7], [0.0, 0.15])

    def test_malformed_directions_and_nonfinite_values_are_dropped(self):
        arrays = self.returns(8)
        arrays[3][:, 0] = [0, 0]
        arrays[3][:, 1] = [2, 0]
        arrays[3][:, 2] = [np.nan, 0]
        arrays[2][3] = np.inf
        arrays[0][0, 4] = np.nan
        arrays[0][5, 5] = np.inf
        arrays[3][:, 6] = [1.0001, 0]
        packed = self.pack(arrays)
        self.assertEqual(int(packed[:, 8].sum()), 2)
        np.testing.assert_allclose(packed[:2, 4:6], [[1, 0], [1, 0]])

    def test_bounding_box_is_lower_inclusive_upper_exclusive(self):
        arrays = self.returns(5)
        arrays[0][0] = [-10, 10, -10.001, 9.999, 1]
        arrays[0][2, 4] = 3
        packed = self.pack(arrays)
        self.assertEqual(int(packed[:, 8].sum()), 2)
        np.testing.assert_allclose(packed[:2, 0], [-10, 9.999])

    def test_empty_returns_give_zero_padding(self):
        packed = self.pack(self.returns(0))
        np.testing.assert_array_equal(packed, np.zeros((8, 9)))

    def test_inconsistent_array_shapes_fail(self):
        arrays = list(self.returns(2))
        arrays[1] = np.zeros((1, 2))
        with self.assertRaises(ValueError):
            self.pack(arrays)

    def test_capped_selection_is_stable_independent_of_radial_values(self):
        arrays = self.returns(40)
        arrays[0][5] = np.arange(40)
        loader = self.make_loader()
        first = loader._pack_returns('sample', *arrays)
        np.testing.assert_array_equal(
            first, loader._pack_returns('sample', *arrays))
        arrays[2][:] = np.arange(40) * -3.0
        changed = loader._pack_returns('sample', *arrays)
        np.testing.assert_array_equal(first[:, :6], changed[:, :6])
        self.assertFalse(np.array_equal(
            first[:, 3], loader._pack_returns('other_sample', *arrays)[:, 3]))
        self.assertEqual(len(np.unique(first[:, 3])), 8)

    def test_compute_delegates_each_configured_sensor(self):
        loader = self.make_loader(radar_channels=(
            'RADAR_FRONT', 'RADAR_BACK_LEFT'))
        sample_rec = {'token': 'sample'}
        calls = []
        loader.nusc.get = lambda table, token: sample_rec

        def load_channel(sample, channel):
            self.assertIs(sample, sample_rec)
            calls.append(channel)
            arrays = self.returns()
            arrays[2][:] = len(calls) * 31
            return arrays

        loader._load_channel = load_channel
        result = loader.compute('sample')
        self.assertEqual(calls, ['RADAR_FRONT', 'RADAR_BACK_LEFT'])
        np.testing.assert_array_equal(result[:2, 6], [31., 62.])

    def test_cache_roundtrip_does_not_read_raw_sensors(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_loader(cache_dir=directory)
            observations = writer._pack_returns('sample', *self.returns())
            writer._write_cache('sample', observations)
            reader = self.make_loader(cache_dir=directory, cache_readonly=True)
            np.testing.assert_array_equal(reader('sample'), observations)
            self.assertEqual(list(Path(directory).iterdir()),
                             [Path(directory) / 'sample.npz'])

    def test_cache_identity_mismatch_fails_instead_of_reusing(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_loader(cache_dir=directory)
            writer._write_cache(
                'sample', writer._pack_returns('sample', *self.returns()))
            for kwargs in (dict(bev_w=5), dict(nsweeps=2),
                           dict(max_time_lag=0.1), dict(max_returns=9),
                           dict(subsample_seed=1), dict(min_distance=2),
                           dict(direction_tolerance=0.002),
                           dict(radar_channels=('RADAR_BACK_LEFT',)),
                           dict(point_cloud_range=(-9, -10, -3, 10, 10, 3)),
                           dict(nusc=types.SimpleNamespace(
                               version='v1.0-other', dataroot='/test'))):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                        self.make_loader(cache_dir=directory, **kwargs)('sample')

    def test_missing_readonly_cache_and_old_bev_cache_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(str(Path(directory) / 'sample.npy'), np.zeros((8, 4, 4)))
            loader = self.make_loader(cache_dir=directory, cache_readonly=True)
            with self.assertRaises(FileNotFoundError):
                loader('sample')
            with self.assertRaises(RuntimeError):
                loader._write_cache('sample', self.pack(self.returns()))

    def test_cache_validates_content_even_when_manifest_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = self.make_loader(cache_dir=directory)
            packed = loader._pack_returns('sample', *self.returns())
            for change in ('float64', 'bad_valid', 'nan', 'padding', 'future'):
                with self.subTest(change=change):
                    malformed = packed.copy()
                    if change == 'float64':
                        malformed = malformed.astype(np.float64)
                    elif change == 'bad_valid':
                        malformed[0, 8] = 0.5
                    elif change == 'nan':
                        malformed[0, 6] = np.nan
                    elif change == 'padding':
                        malformed[-1, 0] = 1
                    else:
                        malformed[0, 7] = -0.1
                    np.savez(
                        loader.cache_path('sample'), observations=malformed,
                        manifest=np.asarray(loader._manifest_json),
                        sample_token=np.asarray('sample'))
                    with self.assertRaises(ValueError):
                        loader('sample')

    def test_manifest_copy_and_unsafe_sample_tokens(self):
        loader = self.make_loader()
        original = loader.manifest
        original['max_returns'] = 100
        self.assertEqual(loader.manifest['max_returns'], 8)
        with self.assertRaises(ValueError):
            loader.cache_path('../sample')

    def test_invalid_constructor_settings_fail(self):
        for kwargs in (dict(max_returns=0), dict(max_returns=1.5),
                       dict(max_time_lag=-1), dict(direction_tolerance=0),
                       dict(radar_channels=()), dict(cache_readonly=True),
                       dict(point_cloud_range=(1, -10, -3, 1, 10, 3))):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.make_loader(**kwargs)


class RadarObservationDatasetAttachmentTest(unittest.TestCase):
    def test_observations_are_only_loaded_for_current_occ_frame(self):
        # Execute the production method with lightweight pipeline substitutes;
        # this checks the field's current-frame gate without loading mmcv,
        # PyTorch, annotation assets, or the full nuScenes dataset.
        path = ROOT / ('projects/mmdet3d_plugin/datasets/'
                       'nuscenes_world_dataset_template.py')
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
                   n.name == 'NuScenesWorldDatasetTemplate')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and
                      n.name == '_prepare_data_info_single')
        # The middle of this existing method loads unrelated occupancy and
        # planning annotations.  Remove only those blocks, retaining original
        # observation loading, pipeline flow, and DataContainer attachment.
        keep = []
        for statement in method.body:
            if isinstance(statement, ast.If):
                serialized = ast.dump(statement)
                if ('get_future_bboxes' in serialized or
                        'get_sdc_planning_label' in serialized or
                        'record_instance' in serialized):
                    continue
            keep.append(statement)
        method.body = keep
        scope = dict(
            copy=__import__('copy'),
            torch=types.SimpleNamespace(from_numpy=lambda value: value),
            DC=lambda value, stack, pad_dims=2: (value, stack, pad_dims))
        executable = ast.Module(body=[method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(executable), str(path), 'exec'), scope)
        calls = []
        output = np.zeros((8, 9), dtype=np.float32)
        fake = types.SimpleNamespace(
            get_data_info=lambda index: {'sample_idx': 'sample'},
            radar_bev_loader=None,
            radar_observation_loader=lambda token: calls.append(token) or output,
            pre_pipeline=lambda value: None,
            pipeline=lambda value: {})
        prepare = scope['_prepare_data_info_single']
        for flag in (False, None):
            result = prepare(fake, 0, flag)
            self.assertNotIn('radar_observations', result)
        self.assertEqual(calls, [])
        result = prepare(fake, 0, True)
        self.assertEqual(calls, ['sample'])
        self.assertIs(result['radar_observations'][0], output)
        self.assertTrue(result['radar_observations'][1])
        self.assertIsNone(result['radar_observations'][2])


if __name__ == '__main__':
    unittest.main()
