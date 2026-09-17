"""CPU representation regression from the two actually audited C/O receipts.

This does not create predictions, completion receipts, or mock pilot success.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT/'server_results/campaign_objective_v1/summary_v1/summary.json'
TYPE_AUDIT = ROOT/'receipts/readout_transition_v2b_receipt_types.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(version):
    tree = ast.parse((ROOT/('readout_transition_diagnostic_'+version+'.py')).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'require'
             and len(node.args) == 2 and isinstance(node.args[1], ast.BinOp)
             and isinstance(node.args[1].left, ast.Constant)
             and node.args[1].left.value == 'Actual final source audit differs: ']
    assert len(calls) == 1
    return compile(ast.Expression(calls[0].args[0]), '<actual-receipt-comparison-'+version+'>', 'eval')


class ReceiptRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert sha(SUMMARY) == '46c37b5b074e73896e357e0bc9744a538abb01b288521769a577f459be54ad6b'
        cls.historical = json.loads(SUMMARY.read_text())['source_receipts']
        cls.audit = json.loads(TYPE_AUDIT.read_text())
        cls.old_gate, cls.new_gate = comparison('v2'), comparison('v3')

    def gate(self, actual, name, new=True):
        return eval(self.new_gate if new else self.old_gate,
                    {'json': json, 'receipt': actual, 'name': name,
                     'development': {'source_receipts': self.historical}})

    def actual_representation(self, name):
        # Restore only the actual tuple recorded by the real CPU comparison.
        receipt = copy.deepcopy(self.historical[name])
        receipt['checkpoint']['optimizer_groups'][0]['betas'] = tuple(
            receipt['checkpoint']['optimizer_groups'][0]['betas'])
        return receipt

    def test_real_audit_has_only_verified_betas_representation_difference(self):
        audit = self.audit
        self.assertEqual(audit['historical_summary_sha256'], sha(SUMMARY))
        self.assertEqual(audit['forward_calls'], 0)
        self.assertEqual(audit['optimizer_updates'], 0)
        self.assertIs(audit['cuda_initialized'], False)
        self.assertEqual([row['model'] for row in audit['checks']], ['native1', 'O'])
        for row in audit['checks']:
            self.assertEqual(row['differences'], [dict(path='$.checkpoint.optimizer_groups[0].betas',
                actual_type='tuple', historical_type='list', actual='(0.9, 0.999)', historical='[0.9, 0.999]')])
            self.assertIs(row['json_equal'], True)
            self.assertEqual(row['normalized_remaining_differences'], [])
            actual = self.actual_representation(row['model']); before = copy.deepcopy(actual)
            self.assertIs(self.gate(actual, row['model'], new=False), False)
            self.assertIs(self.gate(actual, row['model']), True)
            self.assertEqual(actual, before)  # comparison does not mutate its input

    def test_no_value_hash_field_or_float_ulp_is_discarded(self):
        for name in ('native1', 'O'):
            def changed(fn):
                actual = self.actual_representation(name); fn(actual)
                self.assertIs(self.gate(actual, name), False)
            changed(lambda x: x.pop('training'))
            changed(lambda x: x.update(unexpected_field=True))
            changed(lambda x: x['checkpoint'].update(final_head_state_sha256='0'*64))
            changed(lambda x: x['training'].update(updates=513))
            changed(lambda x: x['checkpoint']['optimizer_groups'][0].update(
                betas=(math.nextafter(.9, math.inf), .999)))
            for nonfinite in (float('nan'), float('inf'), -float('inf')):
                actual = self.actual_representation(name)
                actual['checkpoint']['optimizer_groups'][0]['betas'] = (.9, nonfinite)
                with self.assertRaises(ValueError):
                    self.gate(actual, name)

    def test_producer_wrapper_exact_inverse_and_python310_syntax(self):
        previous = (ROOT/'readout_transition_diagnostic_v2.py').read_text()
        current = (ROOT/'readout_transition_diagnostic_v3.py').read_text()
        old = "require(receipt == development['source_receipts'][name], 'Actual final source audit differs: '+name)"
        new = "require(json.loads(json.dumps(receipt, allow_nan=False)) == development['source_receipts'][name], 'Actual final source audit differs: '+name)"
        self.assertEqual(current.count(new), 1)
        self.assertEqual(current.replace(new, old), previous)
        previous = (ROOT/'run_readout_transition_campaign_v2.py').read_text()
        current = (ROOT/'run_readout_transition_campaign_v3.py').read_text()
        old, new = "package/'readout_transition_diagnostic_v2.py'", "package/'readout_transition_diagnostic_v3.py'"
        self.assertEqual(current.count(new), 1)
        self.assertEqual(current.replace(new, old), previous)
        for name in ('readout_transition_diagnostic_v3.py', 'run_readout_transition_campaign_v3.py'):
            ast.parse((ROOT/name).read_text(), feature_version=(3, 10))

    def test_protocol_sources_and_scientific_scope_unchanged(self):
        old = json.loads((ROOT/'readout_transition_diagnostic_protocol_v2.json').read_text())
        new = json.loads((ROOT/'readout_transition_diagnostic_protocol_v3.json').read_text())
        allowed = {'status', 'source_sha256', 'engineering_revision', 'engineering_repair',
                   'previous_frozen_utc', 'prepared_utc', 'frozen_utc'}
        for key, value in old.items():
            if key not in allowed:
                self.assertEqual(new[key], value, key)
        self.assertEqual(new['previous_engineering_repair'], old['engineering_repair'])
        self.assertEqual(new['engineering_repair']['real_cpu_receipt_type_audit_sha256'], sha(TYPE_AUDIT))
        for name, digest in new['source_sha256'].items():
            self.assertEqual(sha(ROOT/name), digest, name)
        for name, digest in new['engineering_repair']['failure_sources_sha256'].items():
            self.assertEqual(sha(ROOT/'server_results'/name), digest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
