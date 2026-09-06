"""Synthetic protocol tests; these do not measure model performance."""

import copy
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'projects/mmdet3d_plugin/datasets/m3_metrics.py'
SPEC = importlib.util.spec_from_file_location('m3_metrics', str(MODULE_PATH))
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
HORIZONS = [0.0, 0.5, 1.0, 1.5, 2.0]


def record(token, scene, confusion=None):
    if confusion is None:
        confusion = [[3.0, 1.0], [2.0, 4.0]]
    hist = np.repeat(np.asarray(confusion, dtype=np.float64)[None], 5, axis=0)
    return {'sample_token': token, 'scene_token': scene,
            'horizon_seconds': HORIZONS[:], 'hist_by_horizon': hist}


class M3MetricsTest(unittest.TestCase):
    def test_padding_is_deduplicated_before_any_aggregate(self):
        first = record('a', 'scene_a')
        second = record('b', 'scene_b', [[10, 0], [0, 2]])
        padded = [first, second, copy.deepcopy(first), copy.deepcopy(second)]
        actual = METRICS.aggregate_records(padded, expected_samples=2)
        np.testing.assert_array_equal(
            actual['hist_for_iou'][0],
            (first['hist_by_horizon'] + second['hist_by_horizon']).sum(axis=0))
        self.assertEqual(len(actual['occ_records']), 2)
        self.assertEqual(actual['num_scenes'], 2)

    def test_conflicting_duplicate_counts_or_scene_fail(self):
        first = record('same', 'scene_a')
        changed = copy.deepcopy(first)
        changed['hist_by_horizon'][3, 0, 1] += 1
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            METRICS.aggregate_records([first, changed])
        changed = copy.deepcopy(first)
        changed['scene_token'] = 'different_scene'
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            METRICS.aggregate_records([first, changed])

    def test_expected_unique_sample_count_catches_missing_records(self):
        first = record('a', 'scene_a')
        with self.assertRaisesRegex(ValueError, 'expected 2 unique'):
            METRICS.aggregate_records([first, first], expected_samples=2)

    def test_known_iou_and_legacy_horizon_index_weighting(self):
        sample = record('a', 'scene_a')
        # Explicitly vary counts with horizon to detect incorrect frame weights.
        sample['hist_by_horizon'] *= np.arange(1, 6)[:, None, None]
        result = METRICS.aggregate_records([sample])
        metric = result['horizon_ious']['0.5']
        np.testing.assert_allclose(metric['per_class_iou'], [3 / 6, 4 / 7])
        self.assertAlmostEqual(metric['mIoU'], (3 / 6 + 4 / 7) / 2)
        expected = np.array([[3, 1], [2, 4]]) * (2 + 3 / 2 + 4 / 3 + 5 / 4)
        np.testing.assert_allclose(
            result['hist_for_iou_future_time_weighting'][0], expected)

    def test_missing_class_is_null_and_not_averaged_as_zero(self):
        result = METRICS.aggregate_records([record('a', 'scene_a', [[5, 0], [0, 0]])])
        metric = result['horizon_ious']['0']
        self.assertEqual(metric['per_class_iou'], [1.0, None])
        self.assertEqual(metric['mIoU'], 1.0)
        report = METRICS.summarize_records(result['occ_records'], bootstrap=10)
        absent = report['metrics']['future/class_1_iou']
        self.assertEqual(absent['ci95'], [None, None])
        self.assertEqual(absent['defined_replicates'], 0)
        json.dumps(METRICS.json_safe(report), allow_nan=False)

    def test_completely_empty_confusion_has_undefined_mean(self):
        metrics = METRICS.confusion_ious(np.zeros((2, 2)))
        self.assertIsNone(metrics['mIoU'])
        self.assertEqual(metrics['per_class_iou'], [None, None])

    def test_incompatible_horizons_shapes_and_invalid_counts_fail(self):
        first = record('a', 'scene_a')
        variants = []
        changed = record('b', 'scene_b')
        changed['horizon_seconds'][-1] = 2.5
        variants.append(changed)
        changed = record('b', 'scene_b')
        changed['hist_by_horizon'] = np.zeros((5, 3, 3))
        variants.append(changed)
        for bad_value in (-1.0, float('nan'), float('inf')):
            changed = record('b', 'scene_b')
            changed['hist_by_horizon'][0, 0, 0] = bad_value
            variants.append(changed)
        for changed in variants:
            with self.subTest(changed=repr(changed)):
                with self.assertRaises(ValueError):
                    METRICS.aggregate_records([first, changed])

    def test_imbalanced_scenes_are_resampled_as_whole_clusters(self):
        # Nine perfect frames in one scene, one fully incorrect frame in another.
        # A scene bootstrap has substantial probability of drawing either scene
        # twice: its 95% interval must reach both 0 and 1. A frame bootstrap would
        # effectively never draw ten copies of the single incorrect frame.
        records = [record('large_{}'.format(i), 'large_scene', [[1, 0], [0, 1]])
                   for i in range(9)]
        records.append(record('small_0', 'small_scene', [[0, 1], [1, 0]]))
        report = METRICS.summarize_records(records, bootstrap=300, seed=0)
        metric = report['metrics']['future/mIoU']
        self.assertAlmostEqual(metric['estimate'], 9 / 11)
        self.assertEqual(metric['ci95'], [0.0, 1.0])
        self.assertEqual(report['num_scenes'], 2)

    def test_identical_models_have_exact_zero_paired_delta(self):
        records = [record('a', 'one'), record('b', 'two', [[7, 2], [1, 3]])]
        report = METRICS.summarize_records(
            records, compare_records=list(reversed(records)), bootstrap=50)
        for metric in report['paired_difference']['metrics'].values():
            self.assertEqual(metric['estimate'], 0.0)
            self.assertEqual(metric['ci95'], [0.0, 0.0])
            self.assertEqual(metric['defined_replicates'], 50)

    def test_paired_delta_direction_and_matching_sampling(self):
        model = [record('a', 'one', [[2, 0], [0, 2]])]
        baseline = [record('a', 'one', [[1, 1], [1, 1]])]
        report = METRICS.summarize_records(
            model, compare_records=baseline, bootstrap=10)
        metric = report['paired_difference']['metrics']['future/mIoU']
        self.assertAlmostEqual(metric['estimate'], 2 / 3)
        np.testing.assert_allclose(metric['ci95'], [2 / 3, 2 / 3])
        self.assertEqual(report['paired_difference']['direction'], 'input_minus_comparison')

    def test_paired_tokens_scene_or_protocol_mismatch_fails(self):
        first = record('a', 'scene_a')
        variants = [record('b', 'scene_a'), record('a', 'scene_b')]
        changed = record('a', 'scene_a')
        changed['horizon_seconds'][-1] = 2.5
        variants.append(changed)
        for variant in variants:
            with self.assertRaisesRegex(ValueError, 'paired'):
                METRICS.summarize_records(
                    [first], compare_records=[variant], bootstrap=2)

    def test_bootstrap_is_reproducible_across_gather_order_and_padding(self):
        records = [record('b', 'two'), record('a', 'one', [[7, 2], [1, 3]])]
        first = METRICS.summarize_records(records, bootstrap=40, seed=7)
        second = METRICS.summarize_records(
            list(reversed(records)) + [records[0]], bootstrap=40, seed=7)
        self.assertEqual(first, second)

    def test_json_and_pickle_cli_export_real_input_records(self):
        records = [record('synthetic_a', 'synthetic_scene')]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            json_input = directory / 'synthetic.json'
            pickle_input = directory / 'synthetic.pkl'
            output = directory / 'summary.json'
            json_input.write_text(json.dumps(METRICS.json_safe(
                {'occ_records': records})), encoding='utf-8')
            with pickle_input.open('wb') as stream:
                pickle.dump({'occ_records': records}, stream)
            command = [sys.executable, str(ROOT / 'tools/m3_summarize_metrics.py'),
                       '--input', str(json_input), '--compare', str(pickle_input),
                       '--output', str(output), '--bootstrap', '5', '--seed', '0',
                       '--expected-samples', '1']
            subprocess.run(command, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['num_unique_samples'], 1)
            self.assertEqual(report['paired_difference']['metrics']['future/mIoU']['ci95'],
                             [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
