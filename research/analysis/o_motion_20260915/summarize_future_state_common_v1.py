"""Fixed O/K/B dev200 pooling, with each model's own native t0 prediction.

Original common count/ratio mathematics comes from SHA-bound v3 helpers.
Only the model/record contract and permission for different predicted t0
change. This computes no new metric and never selects a model or threshold.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V3_SHA = '338f9c0bb28b800711e6527d9208c3046132a5c652805a92333e4c209c529918'
METRIC_SHA = '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951'
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
ARMS = ('O', 'K', 'B')
PAIRS = (('K', 'O'), ('B', 'O'), ('K', 'B'))
REPETITIONS = 10000
SEED = 11


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_bound(name, digest):
    path = HERE / (name + '.py')
    require(sha(path) == digest, 'Frozen math source changed: ' + name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path and sha(module.__file__) == digest,
                'Different module already imported: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prior_math():
    load_bound('common_occupancy_change_metrics_v1', METRIC_SHA)
    return load_bound('summarize_common_change_v3', V3_SHA)


def scene_weights(scenes, repetitions=REPETITIONS, seed=SEED):
    require(len(scenes) > 0 and type(repetitions) is int and repetitions > 0, 'Invalid scene bootstrap')
    draws = np.random.RandomState(seed).randint(len(scenes), size=(repetitions, len(scenes)))
    weights = np.zeros((repetitions, len(scenes)), dtype=np.int64)
    np.add.at(weights, (np.arange(repetitions)[:, None], draws), 1)
    return weights


def collect(rows, selection_path):
    """Validate fixed identities and common GT; predicted t0 may differ."""
    require(sha(selection_path) == SELECTION_SHA, 'Fixed selection changed')
    selection = json.loads(Path(selection_path).read_text())
    selected = [r for r in selection['records'] if r['split'] == 'development']
    require(len(rows) == len(selected) == 200 and len({r['scene_token'] for r in selected}) == 100,
            'Require complete fixed dev200/100 scenes')
    require(len({r['sample_token'] for r in rows}) == len(rows), 'Duplicate anchor')
    prior = prior_math()
    scenes = sorted({r['scene_token'] for r in selected})
    lookup = {s: i for i, s in enumerate(scenes)}
    length = prior.N_OCC + prior.N_CHANGE + prior.N_GROUP
    sums = {a: np.zeros((len(scenes), length), dtype=np.int64) for a in ARMS}
    for ordinal, (row, expected) in enumerate(zip(rows, selected)):
        require(type(row['ordinal']) is int and row['ordinal'] == ordinal and
                all(row[k] == expected[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')),
                'Anchor identity/order differs from fixed selection')
        require(set(row['metrics_by_arm']) == set(row['hist_by_arm']) == set(ARMS), 'Incomplete O/K/B row')
        require(row['t0_prediction_policy'] == 'model_specific_native' and
                row['native_CPU_hist_exact'] is True and row['O_reference_hist_exact'] is True,
                'Native evaluator/reference or t0 policy differs')
        require(all(valid_sha(row[k]) for k in ('inputs_sha256', 'targets_sha256',
                'raw_label_sha256', 'sparse_label_sha256')), 'Missing common input/label SHA')
        vectors = {a: prior.count_vector(row['metrics_by_arm'][a], row) for a in ARMS}
        reference = row['metrics_by_arm']['O']
        ro, rc, rg = prior.unpack(vectors['O'])
        for arm in ARMS:
            current = row['metrics_by_arm'][arm]
            mo, mc, mg = prior.unpack(vectors[arm])
            require(current['shape_hxyz'] == reference['shape_hxyz'] == [5, 512, 512, 40] and
                    current['extent_xyz_m'] == reference['extent_xyz_m'], 'Original output grid differs')
            native_hist = np.asarray(row['hist_by_arm'][arm])
            require(native_hist.dtype.kind in 'iu' and np.array_equal(mo, native_hist),
                    'CPU counts differ from this arm native evaluator')
            require(np.array_equal(mo.sum(-1), ro.sum(-1)), 'Original GT denominator differs')
            require(np.array_equal(mc.sum(-1), rc.sum(-1)), 'Transition GT denominator differs')
            require(np.array_equal(mg[..., 0], rg[..., 0]), 'Speed-group GT support differs')
            t0 = current['t0_boundary']
            require(all(valid_sha(t0[k]) for k in ('prediction_full_binary_sha256',
                    'prediction_on_valid_sha256', 'gt_valid_mask_sha256')), 'Incomplete native t0 hashes')
            require(t0['gt_valid_mask_sha256'] == reference['t0_boundary']['gt_valid_mask_sha256'],
                    'Common t0 GT mask differs')
            require(t0['valid_voxels'] == int(mo[0].sum()) and
                    t0['predicted_foreground_on_valid'] == int(mo[0, :, 1].sum()) and
                    t0['equals_GT_on_valid'] == bool(mo[0, 0, 1] + mo[0, 1, 0] == 0),
                    'This arm t0 boundary disagrees with its own confusion')
            require([r['domain'] for r in current['transitions']] ==
                    [r['domain'] for r in reference['transitions']], 'Transition GT masks differ')
            require([r['actual_dt_seconds'] for r in current['horizons']] ==
                    [r['actual_dt_seconds'] for r in reference['horizons']], 'Actual horizon times differ')
            sums[arm][lookup[row['scene_token']]] += vectors[arm]
    return ARMS, scenes, sums


def aggregate_counts(arms, scenes, sums, pairs, repetitions=REPETITIONS, seed=SEED):
    """Pure v3 arithmetic; parametrized names permit truthful old-data checks."""
    prior = prior_math()
    weights = scene_weights(scenes, repetitions, seed)
    points, boot, counts = {}, {}, {}
    for arm in arms:
        total = sums[arm].sum(0)
        counts[arm] = dict(zip(('occupancy_5x2x2', 'transition_4x4x4', 'groups_4x8x3_GT_TP_FN'), prior.unpack(total)))
        points[arm] = prior.metric_values(total)
        boot[arm] = prior.metric_values(weights @ sums[arm])
    differences = {}
    for a, b in pairs:
        result = {}
        for key in points[a]:
            deltas = boot[a][key] - boot[b][key]
            valid = np.isfinite(deltas)
            result[key] = dict(difference=points[a][key] - points[b][key],
                unit='percentage_points' if key.endswith('_percent') else 'voxels',
                lower95=None if not valid.any() else np.quantile(deltas[valid], .025),
                upper95=None if not valid.any() else np.quantile(deltas[valid], .975),
                finite_bootstrap_repetitions=int(valid.sum()),
                undefined_bootstrap_repetitions=int((~valid).sum()))
        differences[a + '-minus-' + b] = result
    return prior.finite_json(dict(pooled_counts=counts, values=points, comparisons=differences,
        bootstrap_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))


def summarize(rows, selection_path):
    arms, scenes, sums = collect(rows, selection_path)
    result = aggregate_counts(arms, scenes, sums, PAIRS)
    result.update(schema='future-state-common-pooled-summary-v1', samples=len(rows), scenes=len(scenes),
        scene_tokens=scenes, arms=list(arms), comparisons_order=[a + '-minus-' + b for a, b in PAIRS],
        source_contract=dict(v3_math_sha256=V3_SHA, common_metric_sha256=METRIC_SHA, selection_sha256=SELECTION_SHA),
        protocol=dict(primary='four-future mean of per-horizon pooled GMO IoU; original task unchanged',
            pooling='sum original confusion/GT/TP/FN before ratios; never average sample IoU',
            t0_prediction_policy='model_specific_native; own hashes/confusion retained; same GT/masks',
            legacy_metric_t0_caller_note='O/A/Z cross-model prediction equality note does not apply to O/K/B',
            transition_interpretation='both predicted endpoints may change; not an isolated future-motion effect',
            bootstrap=dict(unit='scene', preserve_all_anchors_within_scene=True, paired=True,
                repetitions=REPETITIONS, seed=SEED, interval='percentile95', multiplicity_adjusted=False),
            no_training_seed_uncertainty=True, historical_development_exposure=True,
            no_flow_EPE_or_instance_identity_claim=True, no_candidate_selection=True,
            zero_denominator='null; no perfect empty score',
            artifact_authentication_scope='input record SHA and fixed identities; checkpoint/completion ledger audited by producer/caller'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'selection', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.out)
    require(not output.exists(), 'Do not overwrite an earlier analysis')
    source = Path(args.records); raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = summarize(rows, args.selection)
    result['records_sha256'] = hashlib.sha256(raw).hexdigest()
    result['analysis_source_sha256'] = sha(__file__)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(samples=result['samples'], scenes=result['scenes'], arms=result['arms'], out=str(output))))


if __name__ == '__main__':
    main()
