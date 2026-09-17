"""Pool common outputs and add the predeclared J-minus-D comparison; paired bootstrap resamples scenes.

The original future GMO IoU remains primary. Occupancy transitions and
speed-attributed recall describe common outputs, not flow or instance EPE.
No model, threshold, GT mask, or candidate is selected by this program.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common_occupancy_change_metrics_v1 import POSITIVE_GROUPS, SCHEMA


N_OCC = 5 * 2 * 2
N_CHANGE = 4 * 4 * 4
N_GROUP = 4 * len(POSITIVE_GROUPS) * 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack(vector):
    a = np.asarray(vector)
    prefix = a.shape[:-1]
    return (a[..., :N_OCC].reshape(prefix + (5, 2, 2)),
            a[..., N_OCC:N_OCC+N_CHANGE].reshape(prefix + (4, 4, 4)),
            a[..., N_OCC+N_CHANGE:].reshape(prefix + (4, len(POSITIVE_GROUPS), 3)))


def count_vector(metric, identity):
    require(metric['schema'] == SCHEMA, 'Metric version differs')
    require(all(metric['identity'][k] == identity[k] for k in ('sample_token', 'scene_token')),
            'Raw-label and prediction identities differ')
    require([h['horizon_index'] for h in metric['horizons']] == list(range(5)), 'Horizon order differs')
    require([h['horizon_index'] for h in metric['transitions']] == list(range(1, 5)), 'Transition order differs')
    occ = np.asarray([h['occupancy']['confusion'] for h in metric['horizons']])
    change = np.asarray([h['confusion'] for h in metric['transitions']])
    group = np.asarray([[[h['motion_positive_attribution']['groups'][g][k] for k in ('GT', 'TP', 'FN')]
                         for g in POSITIVE_GROUPS] for h in metric['horizons'][1:]])
    require(occ.shape == (5, 2, 2) and change.shape == (4, 4, 4)
            and group.shape == (4, len(POSITIVE_GROUPS), 3), 'Count array shape differs')
    require(all(a.dtype.kind in 'iu' and np.all(a >= 0) for a in (occ, change, group)), 'Invalid counts')
    require(np.array_equal(group[..., 0], group[..., 1] + group[..., 2]), 'Group GT != TP+FN')
    require(np.array_equal(group[:, :, 0].sum(1), occ[1:, 1, :].sum(1)), 'GT positive groups do not partition GT')
    require(np.array_equal(group[:, :, 1].sum(1), occ[1:, 1, 1]), 'GT positive TP attribution differs')
    for h, row in enumerate(metric['horizons']):
        neg = row['global_negative']
        require([neg['GT'], neg['FP'], neg['TN']] == [int(occ[h, 0].sum()), int(occ[h, 0, 1]), int(occ[h, 0, 0])],
                'Global negatives differ from original confusion')
    for h, row in enumerate(metric['transitions']):
        domain = row['domain']
        require(int(change[h].sum()) == domain['both_valid'], 'Joint-domain count differs')
        require(sum(domain[k] for k in ('both_valid', 't0_valid_h_ignored', 't0_ignored_h_valid', 'both_ignored'))
                == domain['grid_voxels'], 'Joint-domain exclusions do not conserve voxels')
    return np.concatenate([a.reshape(-1) for a in (occ, change, group)]).astype(np.int64)


def collect(rows):
    require(bool(rows), 'No completed sample records')
    arms = tuple(sorted(rows[0]['metrics_by_arm']))
    require('O' in arms, 'Fixed O must be present')
    require(len({r['sample_token'] for r in rows}) == len(rows), 'Duplicate anchors')
    scenes = sorted({r['scene_token'] for r in rows})
    lookup = {s: i for i, s in enumerate(scenes)}
    sums = {a: np.zeros((len(scenes), N_OCC+N_CHANGE+N_GROUP), dtype=np.int64) for a in arms}
    for row in rows:
        require(tuple(sorted(row['metrics_by_arm'])) == arms, 'Partial candidate row')
        vectors = {a: count_vector(row['metrics_by_arm'][a], row) for a in arms}
        reference = row['metrics_by_arm']['O']
        ro, rc, rg = unpack(vectors['O'])
        for arm in arms:
            mo, mc, mg = unpack(vectors[arm])
            current = row['metrics_by_arm'][arm]
            require(np.array_equal(mo.sum(-1), ro.sum(-1)), 'Original GT denominator changed between arms')
            require(np.array_equal(mc.sum(-1), rc.sum(-1)), 'Transition GT denominator changed between arms')
            require(np.array_equal(mg[..., 0], rg[..., 0]), 'Speed group support changed between arms')
            require(current['t0_boundary']['prediction_full_binary_sha256'] ==
                    reference['t0_boundary']['prediction_full_binary_sha256'], 'Common t0 prediction changed')
            require(current['t0_boundary']['gt_valid_mask_sha256'] ==
                    reference['t0_boundary']['gt_valid_mask_sha256'], 'Common t0 GT domain changed')
            require([t['domain'] for t in current['transitions']] ==
                    [t['domain'] for t in reference['transitions']], 'Transition masks changed')
            sums[arm][lookup[row['scene_token']]] += vectors[arm]
    return arms, scenes, sums


def ratio(numerator, denominator):
    n, d = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def metric_values(vector):
    occ, change, groups = unpack(vector)
    oiou = 100 * ratio(occ[..., 1, 1], occ[..., 1, :].sum(-1) + occ[..., :, 1].sum(-1) - occ[..., 1, 1])
    ciou = 100 * ratio(np.diagonal(change, axis1=-2, axis2=-1),
                      change.sum(-1) + change.sum(-2) - np.diagonal(change, axis1=-2, axis2=-1))
    result = {'future_GMO_IoU_percent': oiou[..., 1:].mean(-1), 't0_GMO_IoU_percent': oiou[..., 0],
              'future_FP': occ[..., 1:, 0, 1].sum(-1), 'future_FN': occ[..., 1:, 1, 0].sum(-1)}
    for h in range(4):
        result[f'h{h+1}_GMO_IoU_percent'] = oiou[..., h+1]
    for k, name in enumerate(('00_background', '01_arrival', '10_vacating', '11_persistent')):
        result[f'future_{name}_IoU_percent'] = ciou[..., :, k].mean(-1)
        for h in range(4):
            result[f'h{h+1}_{name}_IoU_percent'] = ciou[..., h, k]
    for k, name in enumerate(POSITIVE_GROUPS):
        recall = 100 * ratio(groups[..., :, k, 1], groups[..., :, k, 0])
        result[f'future_{name}_recall_percent'] = recall.mean(-1)
        for h in range(4):
            result[f'h{h+1}_{name}_recall_percent'] = recall[..., h]
    moving = groups[..., :, 2:4, :].sum(-2)
    recall = 100 * ratio(moving[..., 1], moving[..., 0])
    result['future_speed_gt_0.5_recall_percent'] = recall.mean(-1)
    for h in range(4):
        result[f'h{h+1}_speed_gt_0.5_recall_percent'] = recall[..., h]
    return result


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summarize(rows, bootstrap_repetitions=10000, seed=11):
    require(bootstrap_repetitions > 0, 'Bootstrap count must be positive')
    arms, scenes, sums = collect(rows)
    # Every sampled scene keeps all of its anchors; paired arms share weights.
    draws = np.random.RandomState(seed).randint(len(scenes), size=(bootstrap_repetitions, len(scenes)))
    weights = np.zeros((bootstrap_repetitions, len(scenes)), dtype=np.int64)
    np.add.at(weights, (np.arange(bootstrap_repetitions)[:, None], draws), 1)
    points = {}; boot = {}; counts = {}
    for arm in arms:
        total = sums[arm].sum(0)
        counts[arm] = dict(zip(('occupancy_5x2x2', 'transition_4x4x4', 'groups_4x8x3_GT_TP_FN'), unpack(total)))
        points[arm] = metric_values(total)
        boot[arm] = metric_values(weights @ sums[arm])
    pairs = [(a, 'O') for a in arms if a != 'O']
    if 'A' in arms and 'Z' in arms:
        pairs.append(('A', 'Z'))
    if 'J' in arms and 'D' in arms:
        pairs.append(('J', 'D'))
    differences = {}
    for a, b in pairs:
        result = {}
        for key in points[a]:
            deltas = boot[a][key] - boot[b][key]
            valid = np.isfinite(deltas)
            result[key] = dict(difference=points[a][key]-points[b][key],
                unit='percentage_points' if key.endswith('_percent') else 'voxels',
                lower95=None if not valid.any() else np.quantile(deltas[valid], .025),
                upper95=None if not valid.any() else np.quantile(deltas[valid], .975),
                finite_bootstrap_repetitions=int(valid.sum()),
                undefined_bootstrap_repetitions=int((~valid).sum()))
        differences[a+'-minus-'+b] = result
    return finite_json(dict(schema='common-change-pooled-summary-v2', samples=len(rows), scenes=len(scenes),
        scene_tokens=scenes, arms=list(arms), pooled_counts=counts, values=points, comparisons=differences,
        protocol=dict(primary='four-future mean of per-horizon pooled GMO IoU; unchanged original task',
            secondary='all four occupancy transitions; physical endpoint-speed-attributed positive recall with global FP/FN',
            pooling='sum confusion/GT/TP/FN before ratios; never average sample ratios',
            speed_group_order=list(POSITIVE_GROUPS),
            bootstrap=dict(unit='scene', preserve_all_anchors_within_scene=True, paired=True,
                repetitions=bootstrap_repetitions, seed=seed, interval='percentile95', multiplicity_adjusted=False),
            no_training_seed_uncertainty=True, historical_development_exposure=True,
            no_flow_EPE_or_instance_identity_claim=True, zero_denominator='null; no perfect empty score',
            no_candidate_selection=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    source = Path(a.records); target = Path(a.out)
    require(not target.exists(), 'Do not overwrite a prior analysis')
    raw = source.read_bytes()
    result = summarize([json.loads(line) for line in raw.splitlines() if line.strip()])
    result['records_sha256'] = hashlib.sha256(raw).hexdigest()
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('samples', 'scenes', 'arms', 'values')}, allow_nan=False))


if __name__ == '__main__':
    main()
