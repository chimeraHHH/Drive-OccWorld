"""Paired original-occupancy and rigid-box displacement comparison.

Pool native confusion counts before IoU; average object-point EPE within each
anchor-instance before equal object-instance aggregation. Bootstrap whole scenes,
preserving every anchor and object. This never defines a native flow for O.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('P', 'J', 'D')
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
HORIZONS = (.5, 1., 1.5, 2.)
EPE = ('epe_xy_m', 'epe_3d_m')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def interval(point, draws, unit):
    valid = np.isfinite(draws)
    return dict(difference=point, unit=unit,
                lower95=np.quantile(draws[valid], .025) if valid.any() else None,
                upper95=np.quantile(draws[valid], .975) if valid.any() else None,
                defined_draws=int(valid.sum()), undefined_draws=int((~valid).sum()))


def summary(occupancy, objects, repetitions=10000, seed=11):
    require(bool(occupancy) and repetitions > 0, 'No anchors or invalid repetitions')
    anchors = {r['sample_token']: r for r in occupancy}
    require(len(anchors) == len(occupancy), 'Duplicate occupancy anchor')
    scenes = sorted({r['scene_token'] for r in occupancy})
    scene_index = {s: i for i, s in enumerate(scenes)}
    hist = {a: np.zeros((len(scenes), 5, 2, 2), dtype=np.int64) for a in ('O', 'J', 'D')}
    for r in occupancy:
        require(set(r['hist_by_arm']) == set(hist), 'Require common O/J/D occupancy')
        gt = None; t0 = None
        for a in hist:
            m = np.asarray(r['hist_by_arm'][a])
            require(m.shape == (5, 2, 2) and m.dtype.kind in 'iu' and (m >= 0).all(), 'Invalid native histogram')
            require(gt is None or np.array_equal(m.sum(-1), gt), 'GT denominators differ')
            require(t0 is None or np.array_equal(m[0], t0), 't0 confusion differs')
            gt, t0 = m.sum(-1), m[0]
            hist[a][scene_index[r['scene_token']]] += m
    grouped = {a: {} for a in ARMS}
    for r in objects:
        a = r['arm']; require(a in ARMS, 'Unknown physical prediction arm')
        require(r['sample_token'] in anchors, 'Physical row outside common anchors')
        require(r['scene_token'] == anchors[r['sample_token']]['scene_token'], 'Object scene differs')
        require(r['horizon_seconds'] in HORIZONS and r['group'] in GROUPS[1:], 'Invalid physical horizon/group')
        require(isinstance(r['source_points'], int) and r['source_points'] > 0, 'Invalid object support')
        require(np.isfinite(r['dt_seconds']) and r['dt_seconds'] > 0, 'Invalid actual dt')
        for key in (*EPE, 'zero_epe_xy_m', 'zero_epe_3d_m'):
            require(np.isfinite(r[key]) and r[key] >= 0, 'Invalid EPE')
        key = (r['sample_token'], r['instance_token'], r['horizon_seconds'])
        require(key not in grouped[a], 'Duplicate physical identity')
        grouped[a][key] = r
    require(bool(grouped['P']) and all(set(grouped[a]) == set(grouped['P']) for a in ARMS), 'Physical support differs')
    keys = sorted(grouped['P'])
    equality = ('scene_token', 'source_points', 'dt_seconds', 'group', 'zero_epe_xy_m', 'zero_epe_3d_m')
    for k in keys:
        for a in ARMS:
            require(all(grouped[a][k][v] == grouped['P'][k][v] for v in equality), 'Physical label/support mismatch')
    # Shared scene draws for every arm, horizon and both metric families.
    draws = np.random.RandomState(seed).randint(len(scenes), size=(repetitions, len(scenes)))
    weights = np.zeros((repetitions, len(scenes)), dtype=np.int64)
    np.add.at(weights, (np.arange(repetitions)[:, None], draws), 1)
    occupancy_points, occupancy_boot = {}, {}
    for a, values in hist.items():
        totals = values.sum(0)
        sampled = (weights @ values.reshape(len(scenes), -1)).reshape(repetitions, 5, 2, 2)
        def metrics(x):
            iou = ratio(x[..., 1, 1], x[..., 1, :].sum(-1)+x[..., :, 1].sum(-1)-x[..., 1, 1])*100
            return dict(future_GMO_IoU_percent=iou[..., 1:].mean(-1),
                        t0_GMO_IoU_percent=iou[..., 0],
                        **{'h'+str(i)+'_GMO_IoU_percent': iou[..., i] for i in range(1, 5)},
                        future_FP=x[..., 1:, 0, 1].sum(-1), future_FN=x[..., 1:, 1, 0].sum(-1))
        occupancy_points[a], occupancy_boot[a] = metrics(totals), metrics(sampled)
    occ_comparisons = {}
    for a, b in (('J', 'O'), ('D', 'O'), ('J', 'D')):
        occ_comparisons[a+'-minus-'+b] = {
            k: interval(occupancy_points[a][k]-occupancy_points[b][k],
                        occupancy_boot[a][k]-occupancy_boot[b][k],
                        'percentage_points' if k.endswith('_percent') else 'voxels')
            for k in occupancy_points[a]}
    motion = []
    for horizon in HORIZONS:
        for group in GROUPS:
            selected = [k for k in keys if k[2] == horizon and (group == 'all' or grouped['P'][k]['group'] == group)]
            ids = np.asarray([scene_index[grouped['P'][k]['scene_token']] for k in selected], dtype=int)
            counts = np.bincount(ids, minlength=len(scenes))
            sampled_counts = weights @ counts
            item = dict(horizon_seconds=horizon, group=group, object_anchor_pairs=len(selected),
                        source_point_occurrences=sum(grouped['P'][k]['source_points'] for k in selected),
                        scenes_with_support=int((counts > 0).sum()), metrics={})
            for metric in EPE:
                values = {a: np.asarray([grouped[a][k][metric] for k in selected], dtype=float) for a in ARMS}
                values['zero'] = np.asarray([grouped['P'][k]['zero_'+metric] for k in selected], dtype=float)
                point, boot, descriptions = {}, {}, {}
                for a, v in values.items():
                    sums = np.bincount(ids, weights=v, minlength=len(scenes))
                    point[a] = float(v.mean()) if len(v) else np.nan
                    boot[a] = ratio(weights @ sums, sampled_counts)
                    descriptions[a] = dict(count=len(v), mean=point[a],
                        median=float(np.median(v)) if len(v) else None,
                        p90=float(np.quantile(v, .9)) if len(v) else None)
                pairs = [('J', 'P'), ('D', 'P'), ('J', 'D')]+[(a, 'zero') for a in ARMS]
                item['metrics'][metric] = dict(values=descriptions, comparisons={
                    a+'-minus-'+b: interval(point[a]-point[b], boot[a]-boot[b], 'metres') for a, b in pairs})
            motion.append(item)
    return clean(dict(schema='connected-motion-pooled-comparison-v1', anchors=len(anchors), scenes=len(scenes),
        scene_tokens=scenes, physical_rows_per_arm=len(keys), original_occupancy=occupancy_points,
        occupancy_comparisons=occ_comparisons, physical=motion,
        bootstrap=dict(repetitions=repetitions, seed=seed, unit='scene', keep_all_anchors_objects=True,
            paired=True, interval='percentile95', multiplicity_adjusted=False,
            weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()),
        contract=dict(occupancy='original per-horizon pooled GMO IoU averaged over four futures',
            physical='point EPE mean inside object, then equal anchor-instance weight; not voxel weighted',
            physical_labels='fixed t0 source-point rigid-box proxy, not directly observed dense scene flow',
            P='unchanged pretrained source-motion head, not O',
            O_has_no_native_flow=True, EPE_lower_is_better=True,
            zero_denominator='null, never zero error', median_p90='point estimates; CI above applies to means',
            single_training_seed=11, historical_development_exposure=True, model_selected=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('occupancy-records', 'motion-objects', 'out'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    output = Path(a.out); require(not output.exists(), 'Do not overwrite a prior result')
    paths = [Path(a.occupancy_records), Path(a.motion_objects)]
    rows = [[json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in paths]
    require(len(rows[0]) == 200 and len({r['scene_token'] for r in rows[0]}) == 100, 'CLI requires complete dev200/100 scenes')
    result = summary(*rows)
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(anchors=result['anchors'], scenes=result['scenes'],
                         original_occupancy=result['original_occupancy'], out=str(output))))


if __name__ == '__main__':
    main()
