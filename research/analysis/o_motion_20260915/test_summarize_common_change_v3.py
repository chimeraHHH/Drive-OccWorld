"""CPU regression on completed J/D data; renamed arms are TEST_ONLY in memory.

No predictions or candidate scores are created or written. Each v3 bootstrap
is compared with the previously frozen v2 result, without rerunning v2.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import unittest

import summarize_common_change_v3 as v3


ROOT = Path(__file__).resolve().parent
V2_SHA = '318dee1a19fbd5b0863f50d2e0346723d7050329b55170cb8efac60929b962ae'
RECORDS_SHA = '5699c554924db0a96b9fb21a12cf3398ace28b30f0a8c1eafa17bde9d25cfc92'
POOLED_SHA = 'e4f832e95a903310ac95e032b78d473651bc8b23a6c7cc154eec9d4183326040'
METRIC_SHA = '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951'
EVIDENCE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_leaves(value):
    if isinstance(value, dict):
        return sum(numeric_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(numeric_leaves(v) for v in value)
    return int(isinstance(value, (int, float)) and not isinstance(value, bool))


class SummaryV3Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT/'server_results/training/common_connected_motion_dev200_v2'
        complete_path = directory/'complete.json'
        complete = json.loads(complete_path.read_text())
        assert complete['status'] == 'COMPLETE_FINAL_DEVELOPMENT_EVALUATION'
        assert complete['samples'] == 200 and complete['scenes'] == 100
        assert complete['optimizer_updates'] == 0 and complete['arms'] == ['O', 'J', 'D']
        for filename, digest in complete['files_sha256'].items():
            assert sha(directory/filename) == digest
        assert sha(directory/'records.jsonl') == RECORDS_SHA
        pooled_path = ROOT/'common_connected_motion_dev200_v2_pooled.json'
        assert sha(pooled_path) == POOLED_SHA
        assert sha(ROOT/'summarize_common_change_v2.py') == V2_SHA
        assert sha(ROOT/'common_occupancy_change_metrics_v1.py') == METRIC_SHA
        cls.expected = json.loads(pooled_path.read_text())
        assert cls.expected['records_sha256'] == RECORDS_SHA
        assert cls.expected['analysis_source_sha256'] == V2_SHA
        cls.rows = [json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
        cls.actual = v3.summarize(cls.rows, bootstrap_repetitions=10000, seed=11)
        EVIDENCE.update(input_complete_sha256=sha(complete_path), input_files_sha256=complete['files_sha256'],
            reference_pooled_sha256=POOLED_SHA, source_v2_sha256=V2_SHA,
            source_v3_sha256=sha(ROOT/'summarize_common_change_v3.py'), metric_source_sha256=METRIC_SHA,
            real_samples=len(cls.rows), real_scenes=len({r['scene_token'] for r in cls.rows}),
            bootstrap=dict(repetitions=10000, seed=11, unit='paired scene'),
            v2_bootstrap_executed=False, new_model_predictions_read=False,
            new_candidate_results_written=False)

    def test_01_exact_source_inverse_and_python310(self):
        text = (ROOT/'summarize_common_change_v3.py').read_text()
        ast.parse(text, feature_version=(3, 10))
        inverse = text.replace('Pool common outputs with J-minus-D and R-minus-R0 comparisons;',
                               'Pool common outputs and add the predeclared J-minus-D comparison;', 1)
        new_pair = "    if 'R' in arms and 'R0' in arms:\n        pairs.append(('R', 'R0'))\n"
        self.assertEqual(inverse.count(new_pair), 1)
        inverse = inverse.replace(new_pair, '', 1)
        inverse = inverse.replace("schema='common-change-pooled-summary-v3'",
                                  "schema='common-change-pooled-summary-v2'", 1)
        self.assertEqual(inverse.encode(), (ROOT/'summarize_common_change_v2.py').read_bytes())
        EVIDENCE['exact_source_inverse_passed'] = True

    def test_02_all_existing_real_values_and_intervals_exact(self):
        expected = {k: v for k, v in self.expected.items()
                    if k not in ('schema', 'records_sha256', 'analysis_source_sha256')}
        actual = {k: v for k, v in self.actual.items() if k != 'schema'}
        self.assertEqual(self.actual['schema'], 'common-change-pooled-summary-v3')
        self.assertEqual(actual, expected)
        self.assertEqual(len({r['sample_token'] for r in self.rows}), 200)
        scene_counts = {s: sum(r['scene_token'] == s for r in self.rows) for s in self.actual['scene_tokens']}
        self.assertEqual(set(scene_counts.values()), {2})
        EVIDENCE.update(existing_output_exact=True, compared_numeric_leaves=numeric_leaves(expected),
                        original_comparisons=list(self.expected['comparisons']),
                        maximum_numeric_difference=0.0, undefined_counts_and_nulls_exact=True)

    def test_03_TEST_ONLY_renaming_preserves_every_pair_metric(self):
        # These are authenticated J/D dictionaries with TEST_ONLY aliases;
        # no metric/count/identity field is fabricated or edited, and none of
        # these renamed summary values is written into an artifact.
        renamed = [dict(row, metrics_by_arm={'O': row['metrics_by_arm']['O'],
                    'R': row['metrics_by_arm']['J'], 'R0': row['metrics_by_arm']['D']})
                   for row in self.rows]
        mapped = v3.summarize(renamed, bootstrap_repetitions=10000, seed=11)
        self.assertEqual(mapped['comparisons']['R-minus-R0'], self.actual['comparisons']['J-minus-D'])
        for new, old in [('R', 'J'), ('R0', 'D'), ('O', 'O')]:
            self.assertEqual(mapped['values'][new], self.actual['values'][old])
            self.assertEqual(mapped['pooled_counts'][new], self.actual['pooled_counts'][old])
        self.assertEqual(mapped['comparisons']['R-minus-O'], self.actual['comparisons']['J-minus-O'])
        self.assertEqual(mapped['comparisons']['R0-minus-O'], self.actual['comparisons']['D-minus-O'])
        self.assertEqual(mapped['scene_tokens'], self.actual['scene_tokens'])
        self.assertEqual(mapped['protocol'], self.actual['protocol'])
        EVIDENCE.update(TEST_ONLY_alias_comparison_exact=True,
            TEST_ONLY_alias_mapping={'R': 'existing J', 'R0': 'existing D'},
            TEST_ONLY_fixture_persisted=False,
            pair_metric_count=len(mapped['comparisons']['R-minus-R0']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError('Do not overwrite test acceptance')
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryV3Regression))
    receipt = dict(schema='common-change-v3-regression-test-v1',
        status='PASS_CPU_REGRESSION_TEST_ONLY' if result.wasSuccessful() else 'FAILED_CPU_REGRESSION',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), seconds=time.monotonic()-started,
        evidence=EVIDENCE, test_source_sha256=sha(__file__))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
