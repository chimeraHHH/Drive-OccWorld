"""Fixed D/K/B rigid-box proxy EPE on original dev200 source-point support.

No flow is invented for O. This preserves original per-object point means,
equal anchor-instance aggregation, and paired whole-scene bootstrap. Median
and p90 are descriptive point estimates; intervals apply to mean differences.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON_SHA = '23f6fc0a6d7a9c63a1e77f90ca94a2aedc72b68b82b201df4f34c6d3f3123e8a'
PRIOR_SHA = '3b0822c7d4c9218211dfec7c7d6854fd67b28f9b43640134dfe2228690a2c40e'
SPARSE_MANIFEST_SHA = 'cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe'
ARMS = ('D', 'K', 'B')
PAIRS = (('K', 'D'), ('B', 'D'), ('K', 'B'), ('D', 'zero'), ('K', 'zero'), ('B', 'zero'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def math_modules():
    name = 'summarize_future_state_common_v1'
    path = HERE / (name + '.py')
    require(sha(path) == COMMON_SHA, 'Frozen K/B common contract changed')
    if name in sys.modules:
        common = sys.modules[name]
        require(Path(common.__file__).resolve() == path and sha(common.__file__) == COMMON_SHA,
                'Different common module imported')
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        common = importlib.util.module_from_spec(spec); sys.modules[name] = common
        spec.loader.exec_module(common)
    prior = common.load_bound('summarize_connected_motion_v1', PRIOR_SHA)
    return common, prior


def physical_rows(objects, anchor_scenes, arms):
    """Original row/key/support checks, with parameterized prediction names."""
    _, prior = math_modules()
    grouped = {a: {} for a in arms}
    for row in objects:
        arm = row['arm']; require(arm in arms, 'Unknown physical prediction arm')
        require(row['sample_token'] in anchor_scenes and
                row['scene_token'] == anchor_scenes[row['sample_token']], 'Physical row outside common anchor/scene')
        require(row['horizon_seconds'] in prior.HORIZONS and row['group'] in prior.GROUPS[1:],
                'Invalid physical horizon/group')
        require(type(row['source_points']) is int and row['source_points'] > 0, 'Invalid object source count')
        require(np.isfinite(row['dt_seconds']) and row['dt_seconds'] > 0, 'Invalid actual dt')
        for key in (*prior.EPE, 'zero_epe_xy_m', 'zero_epe_3d_m'):
            require(np.isfinite(row[key]) and row[key] >= 0, 'Invalid physical EPE')
        key = (row['sample_token'], row['instance_token'], row['horizon_seconds'])
        require(key not in grouped[arm], 'Duplicate physical identity')
        grouped[arm][key] = row
    reference = arms[0]
    require(bool(grouped[reference]) and all(set(grouped[a]) == set(grouped[reference]) for a in arms),
            'Physical identity/valid support differs')
    keys = sorted(grouped[reference])
    equality = ('scene_token', 'source_points', 'dt_seconds', 'group', 'zero_epe_xy_m', 'zero_epe_3d_m')
    for key in keys:
        for arm in arms:
            require(all(grouped[arm][key][k] == grouped[reference][key][k] for k in equality),
                    'Physical label/source/zero-reference mismatch')
    return grouped, keys


def check_manifest(occupancy, grouped, sparse_manifest_path):
    require(sha(sparse_manifest_path) == SPARSE_MANIFEST_SHA, 'Original sparse manifest changed')
    manifest = json.loads(Path(sparse_manifest_path).read_text())
    descriptors = {r['identity']['sample_token']: r for r in manifest['records'] if r['identity']['split'] == 'development'}
    require(len(descriptors) == 200 and set(descriptors) == {r['sample_token'] for r in occupancy},
            'Sparse label selection differs')
    _, prior = math_modules()
    by_token = {}
    for row in grouped['D'].values():
        by_token.setdefault(row['sample_token'], []).append(row)
    for row in occupancy:
        desc = descriptors[row['sample_token']]
        require(all(row[k] == desc['identity'][k] for k in ('sample_token', 'scene_token', 'official_index', 'split')),
                'Sparse identity differs')
        require(row['sparse_label_sha256'] == desc['sha256'] and row['raw_label_sha256'] == desc['label_source_sha256'],
                'Physical/common label provenance differs')
        for h, horizon in enumerate(prior.HORIZONS):
            chosen = [r for r in by_token.get(row['sample_token'], []) if r['horizon_seconds'] == horizon]
            counts = desc['counts']
            require(len(chosen) == counts['valid_objects_by_horizon'][h] and
                    sum(r['source_points'] for r in chosen) == counts['valid_points_by_horizon'][h],
                    'Missing/extra original valid objects or source points')
            for g, name in enumerate(prior.GROUPS[1:]):
                group = [r for r in chosen if r['group'] == name]
                require(len(group) == counts['object_speed_group_counts_by_horizon'][h][g] and
                        sum(r['source_points'] for r in group) == counts['point_speed_group_counts_by_horizon'][h][g],
                        'Physical group support differs from frozen labels')
    return dict(sparse_manifest_sha256=SPARSE_MANIFEST_SHA,
        t0_objects_without_unique_grid_point=sum(d['counts']['t0_boxes_without_unique_grid_point'] for d in descriptors.values()),
        valid_objects_by_horizon=[sum(d['counts']['valid_objects_by_horizon'][h] for d in descriptors.values()) for h in range(4)],
        missing_objects_by_horizon=[sum(d['counts']['missing_objects_by_horizon'][h] for d in descriptors.values()) for h in range(4)],
        valid_points_by_horizon=[sum(d['counts']['valid_points_by_horizon'][h] for d in descriptors.values()) for h in range(4)],
        missing_points_by_horizon=[sum(d['counts']['missing_points_by_horizon'][h] for d in descriptors.values()) for h in range(4)],
        future_out_of_ROI_points_retained_by_horizon=[sum(d['counts']['valid_future_points_outside_half_open_extent_by_horizon'][h]
            for d in descriptors.values()) for h in range(4)])


def aggregate_physical(grouped, keys, arms, scenes, pairs, repetitions=10000, seed=11):
    """Original connected-motion mean/percentile/CI arithmetic, unchanged."""
    common, prior = math_modules()
    weights = common.scene_weights(scenes, repetitions, seed)
    scene_index = {s: i for i, s in enumerate(scenes)}
    reference = arms[0]; motion = []
    for horizon in prior.HORIZONS:
        for group in prior.GROUPS:
            selected = [k for k in keys if k[2] == horizon and (group == 'all' or grouped[reference][k]['group'] == group)]
            ids = np.asarray([scene_index[grouped[reference][k]['scene_token']] for k in selected], dtype=int)
            counts = np.bincount(ids, minlength=len(scenes))
            sampled_counts = weights @ counts
            item = dict(horizon_seconds=horizon, group=group, object_anchor_pairs=len(selected),
                source_point_occurrences=sum(grouped[reference][k]['source_points'] for k in selected),
                scenes_with_support=int((counts > 0).sum()), metrics={})
            for metric in prior.EPE:
                values = {a: np.asarray([grouped[a][k][metric] for k in selected], dtype=float) for a in arms}
                values['zero'] = np.asarray([grouped[reference][k]['zero_' + metric] for k in selected], dtype=float)
                point, boot, descriptions = {}, {}, {}
                for arm, value in values.items():
                    sums = np.bincount(ids, weights=value, minlength=len(scenes))
                    point[arm] = float(value.mean()) if len(value) else np.nan
                    boot[arm] = prior.ratio(weights @ sums, sampled_counts)
                    descriptions[arm] = dict(count=len(value), mean=point[arm],
                        median=float(np.median(value)) if len(value) else None,
                        p90=float(np.quantile(value, .9)) if len(value) else None)
                item['metrics'][metric] = dict(values=descriptions, comparisons={
                    a + '-minus-' + b: prior.interval(point[a] - point[b], boot[a] - boot[b], 'metres') for a, b in pairs})
            motion.append(item)
    return prior.clean(dict(physical=motion, bootstrap_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))


def summarize(occupancy, objects, selection_path, sparse_manifest_path):
    common, _ = math_modules()
    _, scenes, _ = common.collect(occupancy, selection_path)
    grouped, keys = physical_rows(objects, {r['sample_token']: r['scene_token'] for r in occupancy}, ARMS)
    support = check_manifest(occupancy, grouped, sparse_manifest_path)
    result = aggregate_physical(grouped, keys, ARMS, scenes, PAIRS)
    result.update(schema='future-state-physical-pooled-summary-v1', anchors=len(occupancy), scenes=len(scenes),
        scene_tokens=scenes, arms=list(ARMS), physical_rows_per_arm=len(keys), support=support,
        comparisons_order=[a + '-minus-' + b for a, b in PAIRS],
        source_contract=dict(common_summarizer_sha256=COMMON_SHA, original_physical_math_sha256=PRIOR_SHA,
            selection_sha256=common.SELECTION_SHA, sparse_manifest_sha256=SPARSE_MANIFEST_SHA),
        bootstrap=dict(repetitions=10000, seed=11, unit='scene', keep_all_anchors_objects=True,
            paired=True, interval='percentile95', multiplicity_adjusted=False),
        contract=dict(physical='mean point EPE inside each object, then equal anchor-instance weight; not voxel/scene weighted',
            physical_labels='original t0 source-point rigid-box proxy; not directly observed dense scene flow',
            D='unchanged actual final D physical head; baseline displacement, not O',
            O_has_no_native_flow=True, EPE_lower_is_better=True,
            zero_denominator='null; never zero error', median_p90='point estimates; intervals apply to mean differences',
            single_training_seed=11, historical_development_exposure=True, model_selected=False,
            future_out_of_ROI_targets_retained=True,
            artifact_authentication_scope='record SHA, fixed selection and sparse support; tensor/checkpoint ledger audited by producer/caller'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('occupancy-records', 'motion-objects', 'selection', 'sparse-manifest', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); output = Path(args.out)
    require(not output.exists(), 'Do not overwrite an earlier analysis')
    paths = [Path(args.occupancy_records), Path(args.motion_objects)]
    raw = [p.read_bytes() for p in paths]
    rows = [[json.loads(line) for line in value.splitlines() if line.strip()] for value in raw]
    result = summarize(*rows, args.selection, args.sparse_manifest)
    result['input_sha256'] = {str(p): hashlib.sha256(value).hexdigest() for p, value in zip(paths, raw)}
    result['analysis_source_sha256'] = sha(__file__)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(anchors=result['anchors'], scenes=result['scenes'], arms=result['arms'], out=str(output))))


if __name__ == '__main__':
    main()
