"""Fork/cache-state checks with the production NPZ loader and synthetic returns."""
import contextlib
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NAME = '_m3_cache_builder_test_module'
SPEC = importlib.util.spec_from_file_location(NAME, ROOT / 'tools/m3_build_observation_cache.py')
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = MODULE  # Worker task functions must be importable by name.
SPEC.loader.exec_module(MODULE)


@unittest.skipIf('fork' not in multiprocessing.get_all_start_methods(), 'fork required')
class CacheBuilderTest(unittest.TestCase):
    def loader(self, directory, fail_token=None):
        counter = multiprocessing.get_context('fork').Value('i', 0)
        # The lambda is deliberately unpickleable: children must inherit the
        # loaded object via fork, without serializing/reloading its tables.
        nusc = types.SimpleNamespace(version='v1.0-trainval', dataroot='/synthetic',
                                     no_serialization=lambda: None)
        loader = MODULE.load_loader()(
            nusc=nusc, point_cloud_range=(-10, -10, -3, 10, 10, 3),
            bev_h=4, bev_w=4, max_returns=8, cache_dir=directory)

        def compute(token):
            status = json.loads(Path(directory, 'build_status.json').read_text())
            if status['status'] != 'BUILDING':
                raise RuntimeError('Cache was marked complete before all work finished')
            if token == fail_token:
                raise RuntimeError('synthetic read failure')
            with counter.get_lock():
                counter.value += 1
            # Production writes and validates the resulting NPZ, including its
            # manifest and token. Raw PCD access alone is replaced for this test.
            return np.zeros((8, 9), dtype=np.float32)

        loader.compute = compute
        return loader, counter

    def build(self, loader, tokens, directory, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return MODULE.build_cache(loader, tokens, directory, ['train', 'val'],
                                      progress_interval=1, **kwargs)

    def test_parallel_build_completes_all_tokens_using_production_cache_files(self):
        with tempfile.TemporaryDirectory() as directory:
            loader, counter = self.loader(directory)
            tokens = ['sample_{:02d}'.format(i) for i in range(19)]
            record = self.build(loader, tokens + tokens[:2], directory, workers=4)
            self.assertEqual(record['status'], 'COMPLETE')
            self.assertEqual(record['tokens_processed'], 19)
            self.assertEqual(record['tokens_required'], 19)
            self.assertEqual(counter.value, 19)
            self.assertEqual(len(list(Path(directory).glob('*.npz'))), 19)
            for token in tokens:
                np.testing.assert_array_equal(loader(token), np.zeros((8, 9)))
            self.assertEqual(counter.value, 19)
            self.assertEqual(json.loads(Path(directory, 'build_status.json').read_text()), record)

    def test_resume_validates_cache_without_recomputing_raw_returns(self):
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                loader, counter = self.loader(directory)
                tokens = ['sample_{}'.format(i) for i in range(10)]
                self.build(loader, tokens, directory, workers=workers)

                def forbidden(token):
                    raise AssertionError('A valid cached token reread raw PCD')

                loader.compute = forbidden
                resumed = self.build(loader, tokens, directory, workers=workers)
                self.assertEqual(resumed['status'], 'COMPLETE')
                self.assertEqual(counter.value, 10)

    def test_failure_replaces_old_complete_and_records_failure(self):
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                loader, _ = self.loader(directory, fail_token='bad')
                MODULE.write_status(directory, dict(status='COMPLETE', tokens_processed=100))
                with self.assertRaisesRegex(RuntimeError, 'synthetic read failure'):
                    self.build(loader, ['bad', 'good'], directory, workers=workers)
                record = json.loads(Path(directory, 'build_status.json').read_text())
                self.assertEqual(record['status'], 'FAILED')
                self.assertEqual(record['error_type'], 'RuntimeError')
                self.assertEqual(record['tokens_required'], 2)
                self.assertLess(record['tokens_processed'], 2)

    def test_corrupt_existing_entry_fails_without_recomputing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            loader, counter = self.loader(directory)
            np.savez(Path(directory, 'bad.npz'), wrong=np.zeros(1))
            with self.assertRaisesRegex(ValueError, 'Invalid M3 observation cache fields'):
                self.build(loader, ['bad'], directory, workers=2)
            self.assertEqual(counter.value, 0)
            self.assertEqual(json.loads(Path(directory, 'build_status.json').read_text())['status'], 'FAILED')

    def test_explicit_limit_never_publishes_complete_even_if_limit_covers_all_tokens(self):
        for limit in (1, 10):
            with tempfile.TemporaryDirectory() as directory:
                loader, _ = self.loader(directory)
                result = self.build(loader, ['one', 'two'], directory, workers=2, limit=limit)
                self.assertEqual(result['status'], 'LIMITED_SMOKE_ONLY')
                self.assertEqual(result['tokens_processed'], min(limit, 2))

    def test_empty_requested_splits_fail_instead_of_publishing_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            loader, _ = self.loader(directory)
            with self.assertRaisesRegex(ValueError, 'no sample tokens'):
                self.build(loader, [], directory)
            self.assertEqual(json.loads(Path(directory, 'build_status.json').read_text())['status'], 'FAILED')

    def test_invalid_workers_and_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            loader, _ = self.loader(directory)
            for workers in (0, 9, True, 1.5):
                with self.assertRaisesRegex(ValueError, 'workers'):
                    self.build(loader, ['one'], directory, workers=workers)
            for limit in (0, -1, True):
                with self.assertRaisesRegex(ValueError, 'limit'):
                    self.build(loader, ['one'], directory, limit=limit)

    def test_metadata_preparation_failure_clears_a_previous_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            MODULE.write_status(directory, dict(status='COMPLETE'))
            arguments = ['builder', '/missing/config.py', '--cache-dir', directory]
            with patch.object(sys, 'argv', arguments):
                with patch.dict(sys.modules, {'mmcv': None}):
                    with self.assertRaises(ImportError):
                        MODULE.main()
            record = json.loads(Path(directory, 'build_status.json').read_text())
            self.assertEqual(record['status'], 'FAILED')
            self.assertEqual(record['phase'], 'loading_metadata')


if __name__ == '__main__':
    unittest.main()
