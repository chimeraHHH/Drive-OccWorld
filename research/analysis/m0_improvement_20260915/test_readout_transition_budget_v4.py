"""Bounded CPU checks for v4 time limits using the real v3 engineering pilot.

No model, optimizer, synthetic predictions, or successful v4 receipts are made.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BudgetRevision(unittest.TestCase):
    def test_exact_runtime_differences_and_python310(self):
        before = (ROOT/'readout_transition_diagnostic_v3.py').read_text()
        after = (ROOT/'readout_transition_diagnostic_v4.py').read_text()
        self.assertEqual(after.count('<= 3600'), 2)
        self.assertEqual(after.replace('<= 3600', '<= 1800'), before)
        before = (ROOT/'run_readout_transition_campaign_v3.py').read_text()
        after = (ROOT/'run_readout_transition_campaign_v4.py').read_text()
        for old, new in [("policy['total_max_seconds'] == 1800", "policy['total_max_seconds'] == 3600"),
                         ("'max_seconds': 1500", "'max_seconds': 3000"),
                         ("package/'readout_transition_diagnostic_v3.py'", "package/'readout_transition_diagnostic_v4.py'")]:
            self.assertEqual(after.count(new), 1)
            after = after.replace(new, old)
        self.assertEqual(after, before)
        for name in ('readout_transition_diagnostic_v4.py', 'run_readout_transition_campaign_v4.py'):
            ast.parse((ROOT/name).read_text(), feature_version=(3, 10))

    def test_protocol_science_and_26_file_package(self):
        old = read(ROOT/'readout_transition_diagnostic_protocol_v3.json')
        new = read(ROOT/'readout_transition_diagnostic_protocol_v4.json')
        allowed = {'source_sha256', 'status', 'engineering_revision', 'engineering_repair',
                   'previous_engineering_repair', 'previous_frozen_utc', 'prepared_utc', 'frozen_utc',
                   'resources', 'server_sequence'}
        for key, value in old.items():
            if key not in allowed:
                self.assertEqual(new[key], value, key)
        restored = copy.deepcopy(new['resources'])
        self.assertEqual(restored['total_max_seconds'], 3600)
        self.assertEqual(restored['development_ceiling']['max_seconds'], 3000)
        restored['total_max_seconds'] = 1800
        restored['development_ceiling']['max_seconds'] = 1500
        self.assertEqual(restored, old['resources'])
        restored = copy.deepcopy(new['server_sequence'])
        self.assertEqual(restored['execution_deadline_seconds'], 3600)
        restored['execution_deadline_seconds'] = 1800
        self.assertEqual(restored, old['server_sequence'])
        self.assertEqual(new['resources']['pilot'], {'max_seconds': 300, 'max_allocated_gib': 32})
        self.assertEqual(new['resources']['max_allocated_gib'], 32)
        files = dict(new['source_sha256'])
        for name, digest in files.items():
            self.assertEqual(sha(ROOT/name), digest, name)
        self.assertEqual(len(files), 22)
        files.update({'protocol_v2.json': new['parent_protocol_sha256'],
                      'objective_supervision_protocol_v1.json': new['objective_protocol_sha256'],
                      'readout_transition_diagnostic_protocol_v4.json': sha(ROOT/'readout_transition_diagnostic_protocol_v4.json'),
                      'selection_v1.json': read(ROOT/'protocol_v2.json')['selection_sha256']})
        self.assertEqual(len(files), 26)
        for name, digest in files.items():
            self.assertEqual(sha(ROOT/name), digest, name)

    def test_real_pilot_budget_and_cross_protocol_rejection(self):
        root = ROOT/'server_results/campaign_readout_transition_v3'
        policy = read(ROOT/'readout_transition_diagnostic_protocol_v4.json')
        done = read(root/'pilot/complete.json')
        self.assertEqual(done['status'], 'PASS_READOUT_TRANSITION_ENGINEERING')
        for name, digest in done['files_sha256'].items():
            self.assertEqual(sha(root/'pilot'/name), digest)
        for name, digest in policy['engineering_repair']['source_evidence_sha256'].items():
            self.assertEqual(sha(ROOT/'server_results'/name), digest)
        resources = read(root/'pilot/summary.json')['resources']
        cap = max(300, math.ceil(1.75*(resources['pre_sample_initialization_seconds']
                                     + 200*resources['max_sample_seconds']) + 60))
        self.assertEqual(cap, 2388)
        self.assertGreater(cap, 1500)
        self.assertLessEqual(cap, policy['resources']['development_ceiling']['max_seconds'])
        self.assertLess(done['seconds']+cap, policy['resources']['total_max_seconds'])
        self.assertIn('Measured development budget exceeds frozen ceiling', read(root/'failed.json')['error'])
        spec = importlib.util.spec_from_file_location('_v4_budget_gate', ROOT/'readout_transition_diagnostic_v4.py')
        producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
        args = SimpleNamespace(mode='development', max_seconds=cap, max_allocated_gib=32,
            pilot=str(root/'pilot'), authorization=str(ROOT/'readout_transition_diagnostic_protocol_v3.json'),
            diagnostic_protocol=str(ROOT/'readout_transition_diagnostic_protocol_v4.json'))
        with self.assertRaisesRegex(ValueError, 'Pilot gate is not a bound PASS'):
            producer.phase_gate(args, policy)


if __name__ == '__main__':
    unittest.main(verbosity=2)
