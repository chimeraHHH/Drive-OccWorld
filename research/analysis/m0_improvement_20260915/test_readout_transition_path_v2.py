"""CPU-only regression for the actual v1 Path failure; no model or mock results.

The real local native1 512-update log is checked against its certified receipt.
Checkpoint loading and GPU pilot success are deliberately not simulated.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(filename):
    spec = importlib.util.spec_from_file_location('_path_regression_'+Path(filename).stem, ROOT/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class PathRegression(unittest.TestCase):
    def test_exact_minimal_source_and_protocol_changes(self):
        old = (ROOT/'readout_transition_diagnostic.py').read_text()
        new = (ROOT/'readout_transition_diagnostic_v2.py').read_text()
        for name in ('check_log', 'check_payload'):
            fixed = 'audit.'+name+'(Path(a.control_run),'
            self.assertEqual(new.count(fixed), 1)
            new = new.replace(fixed, 'audit.'+name+'(a.control_run,')
        self.assertEqual(new, old)
        old = (ROOT/'run_readout_transition_campaign.py').read_text()
        new = (ROOT/'run_readout_transition_campaign_v2.py').read_text()
        self.assertEqual(new.count("package/'readout_transition_diagnostic_v2.py'"), 1)
        self.assertEqual(new.replace("package/'readout_transition_diagnostic_v2.py'",
                                     "package/'readout_transition_diagnostic.py'"), old)
        before = json.loads((ROOT/'readout_transition_diagnostic_protocol_v1.json').read_text())
        after = json.loads((ROOT/'readout_transition_diagnostic_protocol_v2.json').read_text())
        allowed = {'source_sha256', 'status', 'frozen_utc'}
        for key, value in before.items():
            if key not in allowed:
                self.assertEqual(after[key], value, key)
        for oldname, newname in [('readout_transition_diagnostic.py', 'readout_transition_diagnostic_v2.py'),
                                 ('run_readout_transition_campaign.py', 'run_readout_transition_campaign_v2.py')]:
            self.assertEqual(before['source_sha256'][oldname], sha(ROOT/oldname))
            self.assertEqual(after['source_sha256'][newname], sha(ROOT/newname))
        for name, digest in after['source_sha256'].items():
            self.assertEqual(sha(ROOT/name), digest, name)

    def test_real_control_log_old_call_fails_corrected_call_matches_receipt(self):
        audit = module('frame_aggregate.py')
        helper = module('frame_train.py')
        parent = json.loads((ROOT/'protocol_v2.json').read_text())
        directory = ROOT/'server_results/campaign_memory_v1/runs/native1'
        expected_sha = '0f7b650d3ab4f9fd0a1612943191e1411cb313586a2037e3de4b92b1a88de53f'
        self.assertEqual(sha(directory/'training.jsonl'), expected_sha)
        with self.assertRaisesRegex(TypeError, 'unsupported operand type'):
            audit.check_log(str(directory), helper, parent['training'])
        actual = audit.check_log(Path(str(directory)), helper, parent['training'])
        summary_path = ROOT/'server_results/campaign_objective_v1/summary_v1/summary.json'
        self.assertEqual(sha(summary_path), '46c37b5b074e73896e357e0bc9744a538abb01b288521769a577f459be54ad6b')
        expected = json.loads(summary_path.read_text())['source_receipts']['native1']['training']
        self.assertEqual(actual, expected)

    def test_checkpoint_path_expression_and_all_external_audit_calls(self):
        # Execute the exact pure path expression, not the Torch/checkpoint body.
        source = ast.parse((ROOT/'frame_aggregate.py').read_text())
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'check_payload')
        value = next(n.value for n in fn.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'path' for t in n.targets))
        code = compile(ast.Expression(value), '<actual-check_payload-path>', 'eval')
        directory = ROOT/'server_results/campaign_memory_v1/runs/native1'
        with self.assertRaisesRegex(TypeError, 'unsupported operand type'):
            eval(code, {}, {'directory': str(directory)})
        self.assertEqual(eval(code, {}, {'directory': Path(str(directory))}), directory/'latest.pth')
        producer = ast.parse((ROOT/'readout_transition_diagnostic_v2.py').read_text())
        calls = [n for n in ast.walk(producer) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'audit' and n.func.attr in ('check_log', 'check_payload')]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(ast.unparse(call.args[0]), 'Path(a.control_run)')

    def test_python310_grammar_and_actual_failure_provenance(self):
        for name in ('readout_transition_diagnostic_v2.py', 'run_readout_transition_campaign_v2.py'):
            ast.parse((ROOT/name).read_text(), filename=name, feature_version=(3, 10))
        evidence = ROOT/'receipts/readout_transition_v1_failure_evidence'
        policy = json.loads((ROOT/'readout_transition_diagnostic_protocol_v2.json').read_text())
        for name, digest in policy['engineering_repair']['failure_sources_sha256'].items():
            self.assertEqual(sha(evidence/name), digest)
        failure = json.loads((evidence/'campaign_readout_transition_v1/pilot/failed.json').read_text())
        self.assertEqual(failure['error_type'], 'TypeError')
        self.assertEqual(failure['optimizer_updates'], 0)
        trace = (evidence/'campaign_readout_transition_v1/pilot.log').read_text()
        self.assertIn("audit.check_log(a.control_run, helper, parent['training'])", trace)
        self.assertIn("rows = jsonl(directory/'training.jsonl')", trace)


if __name__ == '__main__':
    unittest.main(verbosity=2)
