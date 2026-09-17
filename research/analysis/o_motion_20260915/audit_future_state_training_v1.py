"""Bounded independent CPU audit; reads records only after a terminal complete.

No torch, model loader, training module, or production summary helper is imported.
Checkpoint tensors and optimizer contents belong to the separate tensor audit.
This audits receipt-bound small files, integer counts, and recorded object errors.
It does not recompute bootstrap intervals or impose a model-selection threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


SCHEMA = 'independent-future-state-training-audit-v1'
GROUPS = ('speed_le_0.1', 'speed_gt_0.1_le_0.5', 'speed_gt_0.5_le_5',
          'speed_gt_5', 'annotated_future_only', 't0_missing_with_history',
          'unknown_no_current_box', 'overlap_current_boxes')
PHYSICAL_GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
EPE_KEYS = ('epe_3d_m', 'epe_xy_m', 'zero_epe_3d_m', 'zero_epe_xy_m')
HORIZONS = (.5, 1., 1.5, 2.)
IDENTITY = ('sample_token', 'scene_token', 'official_index', 'split')
OPTIMIZED = tuple(f'loss_voxel_{f}_inter_{i}' for i in range(3) for f in ('ce', 'lovasz'))
ALL_LOSSES = {f'loss_voxel_{f}_inter_{i}' for i in range(3)
              for f in ('ce', 'lovasz', 'sem_scal', 'geo_scal')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def terminal(root, schema, status):
    """The first read of a run is its terminal receipt, never a score/progress file."""
    root = Path(root)
    require((root / 'complete.json').is_file() and not (root / 'failed.json').exists(),
            f'NOT_TERMINAL_NO_RECORDS_READ: {root}')
    done = read(root / 'complete.json')
    require(done['schema'] == schema and done['status'] == status and
            done['updates'] == 512 and done['examples'] == 2048 and
            done['evaluated_samples'] == 200 and done['mode'] == 'train',
            f'Not a complete fixed-final training/evaluation endpoint: {root}')
    return done


def bounded_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Receipt path escapes its run')
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), 'Receipt symlink escapes its run')
    return result


def verify_small_ledger(root, done):
    """Authenticate small files; intentionally do not load or hash final.pth."""
    verified, deferred = {}, {}
    for relative, digest in done['files_sha256'].items():
        path = bounded_path(root, relative)
        require(path.suffix not in ('.pth', '.pt'), 'Unexpected top-level tensor ledger entry')
        require(sha(path) == digest, 'Small artifact hash mismatch: ' + str(path))
        verified[str(path)] = digest
    for arm, digest in done['arm_complete_sha256'].items():
        folder = root / 'runs' / arm
        require(sha(folder / 'complete.json') == digest, 'Arm complete changed: ' + arm)
        ac = read(folder / 'complete.json')
        require((ac['updates'], ac['examples'], ac['evaluated_samples']) == (512, 2048, 200),
                'Incomplete arm receipt: ' + arm)
        verified[str(folder / 'complete.json')] = digest
        for name, expected in ac['files_sha256'].items():
            path = bounded_path(folder, name)
            if path.suffix in ('.pth', '.pt'):
                deferred[str(path)] = expected
                continue
            require(sha(path) == expected, 'Arm small artifact changed: ' + str(path))
            verified[str(path)] = expected
    return {'verified_small_files': verified, 'tensor_files_deferred_to_tensor_auditor': deferred}


def integer_array(value, shape, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'iu' and np.all(a >= 0),
            'Invalid nonnegative original integer counts: ' + name)
    return a.astype(np.int64, copy=False)


def array_equal(a, b, name):
    require(np.array_equal(np.asarray(a), np.asarray(b)), name)


def finite(value, name):
    require(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite scalar: ' + name)


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, np.float64),
                                                np.asarray(denominator, np.float64))
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def metric_values(occ, change, groups):
    """Direct ratios of independently pooled integer arrays; no helper import."""
    oiou = 100 * ratio(occ[:, 1, 1], occ[:, 1, :].sum(-1) + occ[:, :, 1].sum(-1) - occ[:, 1, 1])
    diag = np.diagonal(change, axis1=-2, axis2=-1)
    ciou = 100 * ratio(diag, change.sum(-1) + change.sum(-2) - diag)
    result = dict(future_GMO_IoU_percent=oiou[1:].mean(), t0_GMO_IoU_percent=oiou[0],
                  future_FP=occ[1:, 0, 1].sum(), future_FN=occ[1:, 1, 0].sum())
    for h in range(4):
        result[f'h{h+1}_GMO_IoU_percent'] = oiou[h+1]
    for i, name in enumerate(('00_background', '01_arrival', '10_vacating', '11_persistent')):
        result[f'future_{name}_IoU_percent'] = ciou[:, i].mean()
        for h in range(4):
            result[f'h{h+1}_{name}_IoU_percent'] = ciou[h, i]
    for i, name in enumerate(GROUPS):
        recall = 100 * ratio(groups[:, i, 1], groups[:, i, 0])
        result[f'future_{name}_recall_percent'] = recall.mean()
        for h in range(4):
            result[f'h{h+1}_{name}_recall_percent'] = recall[h]
    moving = groups[:, 2:4, :].sum(1)
    recall = 100 * ratio(moving[:, 1], moving[:, 0])
    result['future_speed_gt_0.5_recall_percent'] = recall.mean()
    for h in range(4):
        result[f'h{h+1}_speed_gt_0.5_recall_percent'] = recall[h]
    return clean(result)


class NumericDifferences:
    """Describe floating differences without inventing a tolerance/selection gate."""
    def __init__(self):
        self.items = []
        self.compared = 0

    def add(self, label, expected, actual):
        expected = clean(expected)
        self.compared += 1
        require((expected is None) == (actual is None), 'Undefined-value mismatch: ' + label)
        if expected is None:
            return
        finite(actual, label)
        if float(expected) != float(actual):
            self.items.append(dict(label=label, recomputed=expected, recorded=actual,
                                   recorded_minus_recomputed=float(actual) - float(expected)))

    def result(self):
        return dict(compared_scalars=self.compared, nonidentical_scalars=len(self.items),
                    max_absolute_difference=max((abs(x['recorded_minus_recomputed']) for x in self.items), default=0.),
                    largest_differences=sorted(self.items, key=lambda x: abs(x['recorded_minus_recomputed']), reverse=True)[:30],
                    policy='descriptive numeric differences; no new acceptance tolerance or scientific threshold')


def head_lr(update):
    scale = (.1 + .9 * update / 50 if update < 50 else
             .1 + .9 * .5 * (1 + math.cos(math.pi * (update - 50) / 462)))
    return 1e-5 * scale


def readout_lr(update):
    return (1e-3 * (.1 + .9 * update / 25) if update < 25 else
            1e-3 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (update - 25) / 486))))


def support_signature(row):
    return [(h['horizon_seconds'], h['supported'],
             {g: (v['objects'], v['points']) for g, v in h['groups'].items()})
            for h in row['physical_audit']['horizons']]


def training_audit(run, manifest, descriptors):
    rng = np.random.RandomState(11)
    orders = [rng.permutation(512).tolist() for _ in range(4)]
    order_hash = hashlib.sha256(json.dumps(orders, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(order_hash == manifest['sample_orders_sha256'], 'Four-pass order hash differs')
    expected = np.asarray(orders).reshape(-1).tolist()
    logs = {a: jsonl(run / 'runs' / a / 'training.jsonl') for a in ('K', 'B')}
    flattened, remembered = {}, {}
    physical_residuals, initial_ce_differences = [], []
    for arm in ('K', 'B'):
        require(len(logs[arm]) == 512, 'Expected 512 actual optimizer records: ' + arm)
        flattened[arm] = []
        for u, row in enumerate(logs[arm]):
            require((row['update'], row['examples'], row['pass_index'], len(row['samples'])) ==
                    (u + 1, (u + 1) * 4, u // 128, 4), 'Incorrect accum4 update/order endpoint')
            require(row['future_head_lr'] == head_lr(u) and row['readout_lr'] == readout_lr(u),
                    '512-step schedule differs at update ' + str(u + 1))
            for sample in row['samples']:
                index = len(flattened[arm]); ordinal = expected[index]
                require(sample['ordinal'] == ordinal, 'Wrong actual microstep ordinal')
                desc = descriptors[sample['sample_token']]
                require(all(sample[k] == desc['identity'][k] for k in IDENTITY) and
                        desc['identity']['split'] == 'train' and desc['ordinal'] == ordinal and
                        sample['sparse_label_sha256'] == desc['sha256'], 'Training identity/label source differs')
                binding = tuple(sample[k] for k in (*IDENTITY, 'inputs_sha256', 'targets_sha256',
                                                    'sparse_label_sha256', 'input_tree_sha256'))
                if ordinal in remembered:
                    require(remembered[ordinal] == binding, 'Input/target source changed across passes/arms')
                else:
                    remembered[ordinal] = binding
                loss = sample['occupancy_loss']
                require(set(loss['original_twelve']) == ALL_LOSSES and tuple(loss['optimized_keys']) == OPTIMIZED,
                        'Original 12-to-6 loss objective changed')
                require(loss['intermediate_losses_trainable'] and loss['t0_readout_trainable'] and
                        sample['native_logits_unmodified_by_readout'], 'Original occupancy path/gradient scope changed')
                for name, value in loss['original_twelve'].items():
                    finite(value, name)
                total = np.float32(0.)
                for name in loss['optimized_keys']:
                    total = np.float32(total + np.float32(loss['original_twelve'][name]))
                require(float(total) == loss['objective_sum'], 'Float32 six-loss sum differs')
                finite(sample['physical_loss'], 'physical_loss')
                require(float(np.float32(total + np.float32(sample['physical_loss']))) == sample['total_loss'],
                        'Total objective differs from occupancy plus physical')
                means = []
                require([h['horizon_seconds'] for h in sample['physical_audit']['horizons']] == list(HORIZONS),
                        'Physical training horizon order differs')
                for hi, h in enumerate(sample['physical_audit']['horizons']):
                    require(set(h['groups']) == set(PHYSICAL_GROUPS[1:]), 'Physical training group set differs')
                    for gi, name in enumerate(PHYSICAL_GROUPS[1:]):
                        group = h['groups'][name]
                        require(group['objects'] == desc['counts']['object_speed_group_counts_by_horizon'][hi][gi] and
                                group['points'] == desc['counts']['point_speed_group_counts_by_horizon'][hi][gi],
                                'Training physical support differs from original sparse manifest')
                        if group['objects']:
                            finite(group['loss'], 'physical group loss')
                        else:
                            require(group['loss'] is None, 'Empty physical group must have null logged loss')
                    group_losses = [g['loss'] for g in h['groups'].values() if g['objects'] > 0]
                    require(h['supported'] == bool(group_losses), 'Physical group support flag differs')
                    if group_losses:
                        means.append(float(np.mean(np.asarray(group_losses, np.float64))))
                require(sample['physical_audit']['valid_horizons'] == len(means), 'Physical horizon support differs')
                physical_residuals.append(abs((float(np.mean(means)) if means else 0.) - sample['physical_loss']))
                flattened[arm].append(sample)
    require(len(remembered) == 512, 'Original train512 not fully covered')
    paired = (*IDENTITY, 'ordinal', 'inputs_sha256', 'targets_sha256', 'sparse_label_sha256',
              'input_tree_sha256', 'paired_rng_before_sha256', 'paired_rng_after_sha256')
    for i, (k, b) in enumerate(zip(flattened['K'], flattened['B'])):
        require(all(k[key] == b[key] for key in paired), 'K/B input/label/RNG pairing failed at ' + str(i))
        require(support_signature(k) == support_signature(b), 'K/B physical support/normalization differs')
        if i:
            require(k['paired_rng_before_sha256'] == flattened['K'][i-1]['paired_rng_after_sha256'],
                    'RNG stream is discontinuous at ' + str(i))
        if i < 4:
            require(k['native_logits_sha256'] == b['native_logits_sha256'], 'Fresh paired logits differ')
            for name in OPTIMIZED:
                x, y = k['occupancy_loss']['original_twelve'][name], b['occupancy_loss']['original_twelve'][name]
                if 'ce_inter' in name:
                    require(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-7), 'Existing initial CE gate fails')
                    initial_ce_differences.append(abs(x-y))
                else:
                    require(x == y, 'Existing initial exact Lovasz gate fails')
    return dict(updates_per_arm=512, examples_per_arm=2048, paired_microsteps=2048,
                unique_training_anchors=512, four_pass_order_sha256=order_hash,
                all_512_LRs_exact=True, all_4096_selected_and_total_float32_sums_exact=True,
                all_paired_RNG_input_target_label_hashes_equal=True, RNG_chain_continuous=True,
                maximum_initial_CE_absolute_difference=max(initial_ce_differences),
                physical_scalar_reconstruction_max_abs_residual=max(physical_residuals),
                physical_reconstruction_note='double means of logged rounded group scalars; descriptive, not a new gate')


def occupancy_audit(rows, original_o, selection, descriptors, old_rows):
    expected = [r for r in selection['records'] if r['split'] == 'development']
    require(len(rows) == len(original_o) == len(expected) == len(old_rows) == 200, 'Require complete dev200')
    require(len({r['sample_token'] for r in rows}) == 200 and
            len({r['scene_token'] for r in rows}) == 100, 'Duplicate anchor or wrong scene coverage')
    pooled = {a: [np.zeros((5, 2, 2), np.int64), np.zeros((4, 4, 4), np.int64),
                  np.zeros((4, 8, 3), np.int64)] for a in ('O', 'K', 'B')}
    old_by_token = {r['sample_token']: r for r in old_rows}
    require(len(old_by_token) == 200, 'Duplicate old completed anchor')
    for i, (row, ref, wanted) in enumerate(zip(rows, original_o, expected)):
        require(row['ordinal'] == i and all(row[k] == wanted[k] for k in IDENTITY), 'Fixed dev order/identity differs')
        require(all(row[k] == ref[k] for k in ('sample_token', 'scene_token')), 'Original O reference identity differs')
        array_equal(row['hist_by_arm']['O'], ref['hist_by_horizon'], 'Original O per-anchor histogram changed')
        desc = descriptors[row['sample_token']]
        require(all(row[k] == desc['identity'][k] for k in IDENTITY) and
                row['sparse_label_sha256'] == desc['sha256'] and
                row['raw_label_sha256'] == desc['label_source_sha256'], 'Dev sparse/raw provenance differs')
        old = old_by_token[row['sample_token']]
        require(all(row[k] == old[k] for k in IDENTITY) and
                row['inputs_sha256'] == old['inputs_sha256'] and row['targets_sha256'] == old['targets_sha256'] and
                row['sparse_label_sha256'] == old['label_sha256'], 'Old D and new run input/GT/label source differs')
        require(set(row['hist_by_arm']) == set(row['metrics_by_arm']) == {'O', 'K', 'B'}, 'Missing native arm')
        own = {}
        for arm in ('O', 'K', 'B'):
            detail = row['metrics_by_arm'][arm]
            require(detail['schema'] == 'common-occupancy-change-metrics-v1' and
                    all(detail['identity'][k] == row[k] for k in ('sample_token', 'scene_token')) and
                    detail['shape_hxyz'] == [5, 512, 512, 40] and
                    [h['horizon_index'] for h in detail['horizons']] == list(range(5)) and
                    [h['horizon_index'] for h in detail['transitions']] == list(range(1, 5)),
                    'Original evaluation identity/horizon order/geometry changed')
            occ = integer_array(row['hist_by_arm'][arm], (5, 2, 2), 'native occupancy')
            array_equal(occ, [h['occupancy']['confusion'] for h in detail['horizons']], 'Native/CPU hist mismatch')
            trans = integer_array([t['confusion'] for t in detail['transitions']], (4, 4, 4), 'transition')
            groups = integer_array([[[h['motion_positive_attribution']['groups'][g][k] for k in ('GT', 'TP', 'FN')]
                                     for g in GROUPS] for h in detail['horizons'][1:]], (4, 8, 3), 'attribution')
            array_equal(groups[..., 0], groups[..., 1] + groups[..., 2], 'GT != TP+FN in attribution')
            array_equal(groups[:, :, 0].sum(1), occ[1:, 1, :].sum(1), 'Attribution loses foreground GT')
            array_equal(groups[:, :, 1].sum(1), occ[1:, 1, 1], 'Attribution loses foreground TP')
            for hi, horizon in enumerate(detail['horizons']):
                negative = horizon['global_negative']
                require([negative[k] for k in ('GT', 'FP', 'TN')] ==
                        [int(occ[hi, 0].sum()), int(occ[hi, 0, 1]), int(occ[hi, 0, 0])],
                        'Recorded negative counts differ from original occupancy histogram')
            for hi, transition in enumerate(detail['transitions']):
                domain = transition['domain']
                require(int(trans[hi].sum()) == domain['both_valid'] and
                        sum(domain[k] for k in ('both_valid', 't0_valid_h_ignored', 't0_ignored_h_valid', 'both_ignored'))
                        == domain['grid_voxels'], 'Recorded transition counts/domain do not conserve voxels')
            t0 = detail['t0_boundary']
            require(t0['valid_voxels'] == int(occ[0].sum()) and
                    t0['predicted_foreground_on_valid'] == int(occ[0, :, 1].sum()), 'Own native t0 counts differ')
            own[arm] = (occ, trans, groups)
            for target, value in zip(pooled[arm], own[arm]):
                target += value
        for arm in ('K', 'B'):
            for k in (0, 1):
                array_equal(own[arm][k].sum(-1), own['O'][k].sum(-1), 'Common GT class support differs')
            array_equal(own[arm][2][..., 0], own['O'][2][..., 0], 'Motion-group GT domain differs')
            detail, ref_detail = row['metrics_by_arm'][arm], row['metrics_by_arm']['O']
            require(detail['t0_boundary']['gt_valid_mask_sha256'] == ref_detail['t0_boundary']['gt_valid_mask_sha256'] and
                    detail['extent_xyz_m'] == ref_detail['extent_xyz_m'] and
                    [h['actual_dt_seconds'] for h in detail['horizons']] == [h['actual_dt_seconds'] for h in ref_detail['horizons']] and
                    [t['domain'] for t in detail['transitions']] == [t['domain'] for t in ref_detail['transitions']],
                    'Common valid mask/domain differs')
    counts = {a: dict(zip(('occupancy_5x2x2', 'transition_4x4x4', 'groups_4x8x3_GT_TP_FN'), values))
              for a, values in pooled.items()}
    values = {a: metric_values(*items) for a, items in pooled.items()}
    return dict(samples=200, scenes=100, original_O_per_anchor_hist_exact=True,
                old_D_input_target_sparse_source_exact=True, common_GT_domains_exact=True,
                predicted_t0_equality_not_required=True, pooled_counts=clean(counts), values=values)


def physical_key(row):
    return row['sample_token'], row['instance_token'], row['horizon_seconds']


def index_objects(rows, arms, anchor_scenes):
    grouped = {a: {} for a in arms}
    for row in rows:
        arm = row['arm']
        require(arm in grouped and row['sample_token'] in anchor_scenes and
                row['scene_token'] == anchor_scenes[row['sample_token']], 'Physical arm/anchor/scene mismatch')
        require(row['horizon_seconds'] in HORIZONS and row['group'] in PHYSICAL_GROUPS[1:] and
                type(row['source_points']) is int and row['source_points'] > 0 and
                math.isfinite(row['dt_seconds']) and row['dt_seconds'] > 0, 'Invalid physical support')
        for name in EPE_KEYS:
            finite(row[name], name)
            require(row[name] >= 0, 'Negative EPE')
        key = physical_key(row)
        require(key not in grouped[arm], 'Duplicate physical object identity')
        grouped[arm][key] = row
    return grouped


def descriptions(values):
    x = np.asarray(values, np.float64)
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p90=float(np.quantile(x, .9)) if len(x) else None)


def physical_audit(rows, old_objects, dev, descriptors):
    anchor_scenes = {r['sample_token']: r['scene_token'] for r in dev}
    grouped = index_objects(rows, ('D', 'K', 'B'), anchor_scenes)
    legacy = index_objects([r for r in old_objects if r['arm'] == 'D'], ('D',), anchor_scenes)['D']
    keys = sorted(grouped['D'])
    require(keys and all(set(g) == set(keys) for g in grouped.values()) and set(legacy) == set(keys),
            'Original D/K/B/legacy-D physical identity supports differ')
    support_fields = ('scene_token', 'group', 'source_points', 'dt_seconds', 'zero_epe_3d_m', 'zero_epe_xy_m')
    difference_rows, by_anchor = [], {}
    for key in keys:
        d, old = grouped['D'][key], legacy[key]
        require(all(d[name] == old[name] for name in support_fields), 'Legacy D physical source/support differs')
        for arm in ('K', 'B'):
            require(all(grouped[arm][key][name] == d[name] for name in support_fields), 'New physical support differs')
        by_anchor.setdefault(d['sample_token'], []).append(d)
        delta = {name: d[name] - old[name] for name in ('epe_3d_m', 'epe_xy_m')}
        difference_rows.append(dict(sample_token=key[0], instance_token=key[1], horizon_seconds=key[2],
                                    group=d['group'], source_points=d['source_points'], new_minus_old=delta))
    for row in dev:
        token = row['sample_token']; desc = descriptors[token]['counts']
        for hi, horizon in enumerate(HORIZONS):
            selected = [r for r in by_anchor.get(token, []) if r['horizon_seconds'] == horizon]
            require(len(selected) == desc['valid_objects_by_horizon'][hi] and
                    sum(r['source_points'] for r in selected) == desc['valid_points_by_horizon'][hi],
                    'Physical object/point totals differ from original sparse manifest')
            for gi, group in enumerate(PHYSICAL_GROUPS[1:]):
                parts = [r for r in selected if r['group'] == group]
                require(len(parts) == desc['object_speed_group_counts_by_horizon'][hi][gi] and
                        sum(r['source_points'] for r in parts) == desc['point_speed_group_counts_by_horizon'][hi][gi],
                        'Physical speed-group support differs from original sparse manifest')
        require(all(row['physical_object_rows_by_arm'][arm] == len(by_anchor.get(token, []))
                    for arm in ('D', 'K', 'B')), 'Per-anchor physical row counts disagree')
    pooled = []
    for horizon in HORIZONS:
        for group in PHYSICAL_GROUPS:
            selected = [k for k in keys if k[2] == horizon and (group == 'all' or grouped['D'][k]['group'] == group)]
            item = dict(horizon_seconds=horizon, group=group, object_anchor_pairs=len(selected),
                        source_point_occurrences=sum(grouped['D'][k]['source_points'] for k in selected), metrics={})
            for metric in ('epe_3d_m', 'epe_xy_m'):
                vals = {arm: descriptions([grouped[arm][k][metric] for k in selected]) for arm in ('D', 'K', 'B')}
                vals['zero'] = descriptions([grouped['D'][k]['zero_' + metric] for k in selected])
                vals['legacy_D'] = descriptions([legacy[k][metric] for k in selected])
                item['metrics'][metric] = vals
            pooled.append(item)
    deltas = {name: np.asarray([r['new_minus_old'][name] for r in difference_rows], np.float64)
              for name in ('epe_3d_m', 'epe_xy_m')}
    return dict(rows_per_arm=len(keys), all_identity_source_and_support_exact=True,
                original_sparse_manifest_support_exact=True, physical=pooled,
                unchanged_D_numeric_difference={name: dict(nonidentical_rows=int(np.count_nonzero(x)),
                    mean_signed_difference=float(x.mean()), max_absolute_difference=float(np.abs(x).max()),
                    exact_equal=bool(np.all(x == 0))) for name, x in deltas.items()},
                numeric_difference_policy='reported exactly; no newly invented EPE tolerance',
                O_has_no_native_flow=True, mean_definition='mean point EPE per object, then equal anchor-instance weight',
                point_level_errors_recomputed=False), difference_rows, grouped


def verify_summaries(producer, occupancy, physical, grouped, dev_objects, args, protocol):
    diff = NumericDifferences()
    for arm in ('O', 'K', 'B'):
        calculated = occupancy['values'][arm]
        diff.add(f'producer.scores.{arm}.future_macro_GMO_percent', calculated['future_GMO_IoU_percent'],
                 producer['scores'][arm]['future_macro_GMO_percent'])
        array_equal(occupancy['pooled_counts'][arm]['occupancy_5x2x2'],
                    [h['occupancy']['confusion'] for h in producer['scores'][arm]['horizons']], 'Producer pooled occupancy differs')
        array_equal(occupancy['pooled_counts'][arm]['transition_4x4x4'],
                    [h['confusion'] for h in producer['scores'][arm]['transitions']], 'Producer pooled transition differs')
    # Preserve file order for the producer's per-object means; no helper execution.
    for arm in ('D', 'K', 'B'):
        require(len(producer['physical'][arm]['groups']) == 16 and
                {(x['horizon_seconds'], x['group']) for x in producer['physical'][arm]['groups']} ==
                {(h, g) for h in HORIZONS for g in PHYSICAL_GROUPS}, 'Producer physical grouping differs')
        for item in producer['physical'][arm]['groups']:
            subset = [r for r in dev_objects if r['arm'] == arm and r['horizon_seconds'] == item['horizon_seconds']
                      and (item['group'] == 'all' or r['group'] == item['group'])]
            require(item['objects'] == len(subset) and item['source_points'] == sum(r['source_points'] for r in subset),
                    'Producer physical support total differs')
            for key in EPE_KEYS:
                expected = descriptions([r[key] for r in subset])
                require(expected['count'] == item['metrics'][key]['count'], 'Producer physical metric count differs')
                for stat in ('mean', 'median', 'p90'):
                    diff.add(f'producer.physical.{arm}.{item["horizon_seconds"]}.{item["group"]}.{key}.{stat}',
                             expected[stat], item['metrics'][key][stat])
    optional = {}
    if args.common_summary:
        result = read(args.common_summary)
        require(result['schema'] == 'future-state-common-pooled-summary-v1' and
                result['records_sha256'] == sha(args.run / 'development_records.jsonl'), 'Wrong optional common summary input')
        require(result['analysis_source_sha256'] == protocol['analysis_sources_sha256']['summarize_future_state_common_v1.py'],
                'Optional common summary source is not the frozen source')
        for arm in ('O', 'K', 'B'):
            for name, values in occupancy['pooled_counts'][arm].items():
                array_equal(values, result['pooled_counts'][arm][name], 'Optional common summary integer counts differ')
            require(set(result['values'][arm]) == set(occupancy['values'][arm]), 'Optional common metric set differs')
            for name, expected in occupancy['values'][arm].items():
                diff.add(f'common.{arm}.{name}', expected, result['values'][arm][name])
        optional['common'] = {'path': str(args.common_summary), 'sha256': sha(args.common_summary), 'CI_recomputed': False}
    if args.physical_summary:
        result = read(args.physical_summary)
        require(result['schema'] == 'future-state-physical-pooled-summary-v1' and
                result['analysis_source_sha256'] == protocol['analysis_sources_sha256']['summarize_future_state_physical_v1.py'],
                'Optional physical summary schema/source differs')
        require(set(result['input_sha256'].values()) == {sha(args.run / 'development_records.jsonl'),
                                                       sha(args.run / 'development_objects.jsonl')},
                'Wrong optional physical summary record bindings')
        observed = {(x['horizon_seconds'], x['group']): x for x in result['physical']}
        require(set(observed) == {(h, g) for h in HORIZONS for g in PHYSICAL_GROUPS}, 'Optional physical grouping differs')
        for item in physical['physical']:
            actual = observed[item['horizon_seconds'], item['group']]
            for key in ('object_anchor_pairs', 'source_point_occurrences'):
                require(actual[key] == item[key], 'Optional physical support total differs')
            for key in ('epe_3d_m', 'epe_xy_m'):
                for arm in ('D', 'K', 'B', 'zero'):
                    for stat in ('count', 'mean', 'median', 'p90'):
                        diff.add(f'physical.{item["horizon_seconds"]}.{item["group"]}.{key}.{arm}.{stat}',
                                 item['metrics'][key][arm][stat], actual['metrics'][key]['values'][arm][stat])
        optional['physical'] = {'path': str(args.physical_summary), 'sha256': sha(args.physical_summary), 'CI_recomputed': False}
    return {'floating_comparison': diff.result(), 'optional_summaries': optional}


def run(args, current_done, old_done):
    started = time.monotonic()
    current_hash, old_hash = sha(args.run / 'complete.json'), sha(args.d_run / 'complete.json')
    if args.expected_complete_sha:
        require(current_hash == args.expected_complete_sha, 'Calling task terminal complete SHA differs')
    ledger = verify_small_ledger(args.run, current_done)
    old_ledger = verify_small_ledger(args.d_run, old_done)
    protocol, manifest, producer = read(args.protocol), read(args.run / 'manifest.json'), read(args.run / 'summary.json')
    require(protocol['schema'] == 'future-state-motion-training-v1' and protocol['status'] == 'FROZEN' and
            sha(args.protocol) == manifest['sources']['protocol_sha256'], 'Wrong frozen formal protocol')
    require(manifest['training'] == protocol['training'] and manifest['fresh_initialization'] and
            manifest['optimizer_restored'] is False, 'Formal initialization/recipe differs')
    require(old_hash == protocol['connected_complete_sha256'] == manifest['sources']['connected_complete_sha256'],
            'Wrong legacy completed D run')
    require(old_done['final_checkpoints']['D']['sha256'] == protocol['D_checkpoint_sha256'] ==
            manifest['sources']['D_checkpoint_sha256'], 'Wrong actual D checkpoint binding')
    require(sha(args.o_reference) == protocol['o_development_records_sha256'] and
            sha(args.selection) == protocol['selection_sha256'] and
            sha(args.sparse_manifest) == protocol['labels']['manifest_sha256'], 'Reference/selection/label source differs')
    sparse = read(args.sparse_manifest)
    descriptors = {r['identity']['sample_token']: r for r in sparse['records']}
    require(len(descriptors) == len(sparse['records']) == 712, 'Original train512/dev200 label set differs')
    train = training_audit(args.run, manifest, descriptors)
    # No record below is read before both terminal complete receipts have passed.
    dev = jsonl(args.run / 'development_records.jsonl')
    objects = jsonl(args.run / 'development_objects.jsonl')
    occupancy = occupancy_audit(dev, jsonl(args.o_reference), read(args.selection), descriptors,
                                 jsonl(args.d_run / 'development_records.jsonl'))
    physical, d_differences, grouped = physical_audit(objects,
        jsonl(args.d_run / 'development_objects.jsonl'), dev, descriptors)
    summary_check = verify_summaries(producer, occupancy, physical, grouped, objects, args, protocol)
    require(sha(args.run / 'complete.json') == current_hash and sha(args.d_run / 'complete.json') == old_hash,
            'Terminal receipt changed during audit')
    # A final small-file hash pass detects concurrent edits during this audit.
    verify_small_ledger(args.run, current_done)
    result = dict(schema=SCHEMA, status='STRUCTURAL_PASS_STATISTICS_RECOMPUTED',
        current_complete_sha256=current_hash, legacy_D_complete_sha256=old_hash,
        auditor_source_sha256=sha(__file__), protocol_sha256=sha(args.protocol),
        audit_scope='independent NumPy small-artifact/statistics audit; no producer/helper imports, GPU, model or checkpoint load',
        ledger=ledger, legacy_ledger=old_ledger, training=train, occupancy=occupancy, physical=physical,
        summary_check=summary_check, elapsed_seconds=time.monotonic()-started,
        limitations=['No rerun of inference or pointwise EPE; this aggregates receipt-bound per-object errors.',
                     'Tensor state and actual optimizer contents are assigned to the independent checkpoint auditor.',
                     'No bootstrap confidence intervals recomputed; optional summary counts and point estimates only.',
                     'Numeric differences are reported without a new tolerance or model-selection gate.',
                     'O has no native flow; D is the independently trained physical reference.'])
    return clean(result), d_differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'o-reference', 'd-run', 'selection', 'sparse-manifest', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--common-summary', type=Path)
    parser.add_argument('--physical-summary', type=Path)
    parser.add_argument('--expected-complete-sha')
    args = parser.parse_args()
    # Guard before making output directories or opening any nonterminal record.
    current = terminal(args.run, 'future-state-motion-training-v1', 'COMPLETE_FUTURE_STATE_MOTION_TRAINING')
    old = terminal(args.d_run, 'connected-motion-training-v2', 'COMPLETE_CONNECTED_MOTION_TRAINING')
    require(not args.out.exists(), 'Do not overwrite an earlier audit')
    args.out.mkdir(parents=True)
    try:
        result, differences = run(args, current, old)
        (args.out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        with (args.out / 'D_replay_differences.jsonl').open('w') as stream:
            for row in differences:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        floats = result['summary_check']['floating_comparison']
        report = ('# K/B 正式终点独立小件与统计审核\n\n'
            '订单、512 步 LR、原 12 选 6 loss、paired RNG/input/label、dev200 原 O 逐样本混淆、'
            '旧 D 身份/来源/支持检查通过。整数池化与逐对象 EPE 统计已独立重算。\n\n'
            f'汇总浮点比较：{floats["compared_scalars"]} 项，{floats["nonidentical_scalars"]} 项非逐值相同，'
            f'最大绝对差 {floats["max_absolute_difference"]}。差异完整摘要见 audit.json；不据此新设科学阈值。\n\n'
            '旧 D 逐对象重放误差见 D_replay_differences.jsonl；O 没有原生 flow。'
            '本脚本未加载模型/检查点、未访问 GPU、未重跑性能或 bootstrap；真实参数和优化器交由独立 tensor 审核。\n')
        (args.out / 'report.md').write_text(report)
        files = {name: sha(args.out / name) for name in ('audit.json', 'D_replay_differences.jsonl', 'report.md')}
        (args.out / 'complete.json').write_text(json.dumps(dict(schema=SCHEMA, status=result['status'],
            source_sha256=sha(__file__), training_complete_sha256=result['current_complete_sha256'],
            files_sha256=files), indent=2) + '\n')
        print(json.dumps(dict(status=result['status'], out=str(args.out), summary_numeric_differences=floats['nonidentical_scalars'])))
    except Exception as exc:
        (args.out / 'failed.json').write_text(json.dumps(dict(schema=SCHEMA, status='AUDIT_FAILED',
            error=repr(exc), source_sha256=sha(__file__)), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
