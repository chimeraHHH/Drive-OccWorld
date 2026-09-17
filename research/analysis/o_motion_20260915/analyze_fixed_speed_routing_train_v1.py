"""CPU-only fixed speed-threshold comparator on existing train512 predictions.

Reports every old speed-bin boundary; fits/selects nothing and reads no dev data.
Cached query support is GT-defined. This is a physical routing diagnostic, not
full-grid inference, model promotion, or occupancy performance.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT_SHA = '5b8d36a60bba62f81b3e8d494f5b8a591bc540a3cda68fe311d434fbeda1139e'
REFERENCE_SHA = '38bb878bffc98c67715adb84f4a1a2983e5425c1e2a07d0fea1509576117b518'
THRESHOLDS = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
GROUPS = ('all', 'stationary', 'ambiguous', 'moving')
HORIZONS = (.5, 1., 1.5, 2.)
BASELINES = ('D', 'CRN_CV', 'fixed_blend', 'zero')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def select_D(owner, speed):
    """Only the two inference-side arrays enter this fixed decision rule."""
    require(owner.ndim == speed.ndim == 1 and owner.shape == speed.shape and
            np.isfinite(speed).all() and np.all(speed >= 0), 'Input layout/range')
    return (owner[:, None] < 0) & (speed[:, None] >= THRESHOLDS[None, :])


def object_sums(obj, values, count):
    if not len(obj):
        require(count == 0 and len(values) == 0, 'Empty source must have no original objects')
        return np.zeros((0,)+values.shape[1:], dtype=np.float64)
    flat = values.reshape(len(obj), -1)
    return np.stack([np.bincount(obj, weights=flat[:, i], minlength=count)
                     for i in range(flat.shape[1])], axis=1).reshape((count,)+values.shape[1:])


def score_sample(v):
    """Keep all original objects; callers apply future validity only to scoring."""
    obj = v['object_index']; k = len(v['instance_tokens'])
    counts = np.bincount(obj, minlength=k)
    require(np.all(counts > 0), 'Original object must have source points')
    selected = select_D(v['owner'], v['observable_features'][:, 0])
    uncovered = v['owner'] < 0
    require(np.array_equal(selected[:, 0], uncovered) and not selected[:, -1].any(), 'Endpoint policies differ')
    selected_count = object_sums(obj, selected.astype(float), k)
    selected_mass = selected_count/counts[:, None]
    result = []
    for hi, horizon in enumerate(HORIZONS):
        target = v['target_displacement_m'][hi].astype(np.float64)
        d = v['D_displacement_m'][hi].astype(np.float64)
        cv = horizon*v['CV_velocity_mps']
        blend = np.where(uncovered[:, None], d, cv)
        base = np.stack([np.stack([np.linalg.norm((p-target)[:, :2], axis=1),
                                  np.linalg.norm(p-target, axis=1)], axis=1)
                         for p in (d, cv, blend, np.zeros_like(d))], axis=1)
        baseline = object_sums(obj, base, k)/counts[:, None, None]
        policies = []; gains = []; costs = []
        for ti in range(len(THRESHOLDS)):
            field = np.where(selected[:, ti, None], d, cv)
            error = np.stack([np.linalg.norm((field-target)[:, :2], axis=1),
                              np.linalg.norm(field-target, axis=1)], axis=1)
            benefit = np.where(selected[:, ti, None], base[:, 1]-base[:, 0], 0.)
            require(np.allclose(error, base[:, 1]-benefit, rtol=1e-14, atol=1e-14), 'Direct routed EPE disagrees with benefit accounting')
            policies.append(object_sums(obj, error, k)/counts[:, None])
            gains.append(object_sums(obj, np.maximum(benefit, 0.), k)/counts[:, None])
            costs.append(object_sums(obj, np.maximum(-benefit, 0.), k)/counts[:, None])
        epe = np.stack(policies, axis=1)
        require(np.array_equal(epe[:, 0], baseline[:, 2]) and np.array_equal(epe[:, -1], baseline[:, 1]),
                'All-D/none-D routes do not equal old blend/CV')
        result.append(dict(epe=epe, gain=np.stack(gains, axis=1), cost=np.stack(costs, axis=1), baseline=baseline,
                           selected_count=selected_count, selected_mass=selected_mass,
                           uncovered_count=object_sums(obj, uncovered.astype(float), k),
                           uncovered_mass=object_sums(obj, uncovered.astype(float), k)/counts,
                           source_count=counts))
    return result


def synthetic_qa():
    # Unequal object sizes expose accidental point-equal or selected-point weighting.
    obj = np.array([0, 0, 0, 1]); owner = np.array([-1, 2, -1, -1]); speed = np.array([.5, 99., .1, 10.])
    mask = select_D(owner, speed)
    require(mask[:, 2].tolist() == [True, False, False, True] and
            mask[:, 6].tolist() == [False, False, False, True], 'Threshold ties/coverage')
    require(not mask[:, -1].any() and np.array_equal(mask[:, 0], owner < 0), 'Endpoints')
    target = np.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=np.float32)
    d = np.array([[1., 0., 0.], [99., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=np.float32)
    v = dict(object_index=obj, instance_tokens=np.array(['a', 'b']), owner=owner,
             observable_features=speed[:, None], target_displacement_m=np.stack([target]*4),
             D_displacement_m=np.stack([d]*4), CV_velocity_mps=np.array([[0., 0., 0.], [2., 0., 0.], [0., 0., 0.], [0., 0., 0.]]))
    r = score_sample(v)[0]
    require(np.array_equal(r['selected_count'][:, 2], [1., 1.]) and
            np.allclose(r['selected_mass'][:, 2], [1./3., 1.]), 'Complete original object denominator')
    require(np.allclose(r['epe'][:, 2, 0], [0., 2.]) and
            np.allclose(r['gain'][:, 2, 0], [1./3., 0.]) and
            np.allclose(r['cost'][:, 2, 0], [0., 2.]), 'Motion gain and static harm both retained')
    empty = {key: value[:, :0] if key in ('target_displacement_m','D_displacement_m') else value[:0]
             for key, value in v.items()}
    for row in score_sample(empty):
        require(row['epe'].shape == (0,8,2) and row['baseline'].shape == (0,4,2) and
                row['selected_mass'].shape == (0,8), 'Empty-source anchor kept with zero contributions')
    return ['covered CV unchanged and threshold ties accepted', 'zero/infinite thresholds reproduce blend/CV',
            'unequal original object denominators preserved', 'direct vector EPE equals signed gain/cost accounting',
            'legitimate empty-source anchor remains in sample ledger without invented points']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'selection', 'sparse-labels', 'train-index', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=HERE)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--qa-only', action='store_true')
    a = parser.parse_args(); started = time.monotonic()
    require(not a.out.exists(), 'Never overwrite an analysis')
    rules = json.loads(a.rules.read_text())
    require(rules['status'] == 'FROZEN_BEFORE_NEW_POLICY_RESULTS' and rules['source_sha256'] == sha(__file__) and
            rules['thresholds_mps'] == [0., .1, .5, 1., 2., 5., 10., None] and
            rules['reference_analysis_sha256'] == REFERENCE_SHA, 'Fixed routing rules differ')
    qa = synthetic_qa()
    if a.qa_only:
        a.out.write_text(json.dumps(dict(status='PASS_SYNTHETIC_ROUTING_QA_ONLY', checks=qa, source_sha256=sha(__file__),
                                        rules_sha256=sha(a.rules), real_data_read=False), indent=2)+'\n')
        print(a.out); return
    parent_path = HERE/'analyze_motion_evidence_train_v1.py'
    require(sha(parent_path) == PARENT_SHA, 'Original authenticated reader changed')
    spec = importlib.util.spec_from_file_location('original_motion_reader', parent_path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    prior_path = HERE/'motion_evidence_train_analysis_v1.json'
    require(sha(prior_path) == REFERENCE_SHA, 'Original physical reference changed')
    prior = json.loads(prior_path.read_text())
    rows, labels, scenes, sources = parent.authenticate(a)
    counts = np.zeros((4, 3, len(scenes)), dtype=np.int64)
    totals = {name: np.zeros((4, 3, len(scenes))+shape) for name, shape in
              [('epe', (8, 2)), ('gain', (8, 2)), ('cost', (8, 2)), ('baseline', (4, 2)),
               ('selected_count', (8,)), ('selected_mass', (8,)), ('uncovered_count', ()),
               ('uncovered_mass', ()), ('source_count', ())]}
    source_points = 0; valid_points = np.zeros(4, dtype=np.int64)
    for row, label in zip(rows, labels):
        v = parent.load_sample(a, row, label)
        scored = score_sample(v); si = scenes.index(row['identity']['scene_token'])
        source_points += len(v['object_index']); valid_points += v['valid'].sum(axis=1)
        for hi in range(4):
            for gi in range(3):
                take = v['object_future_valid'][hi] & (v['object_speed_group'][hi] == gi)
                counts[hi, gi, si] += int(take.sum())
                for name, out in totals.items():
                    out[hi, gi, si] += scored[hi][name][take].sum(axis=0)
    require(source_points == 1740053 and counts.sum(axis=(1,2)).tolist() == [11254,10967,10636,10332] and
            valid_points.tolist() == [1709467,1675202,1639983,1608197], 'Original population incomplete')
    output = []
    for hi, horizon in enumerate(HORIZONS):
        for group in GROUPS:
            chosen = [0,1,2] if group == 'all' else [GROUPS.index(group)-1]
            count = int(counts[hi, chosen].sum())
            t = {name: value[hi, chosen].sum(axis=(0,1)) for name, value in totals.items()}
            ref = next(r for r in prior['physical'] if r['horizon_seconds'] == horizon and r['group'] == group)
            require(count == ref['original_objects'] and int(t['source_count']) == ref['original_points'] and
                    int(t['uncovered_count']) == ref['uncovered_points'], 'Original group support changed')
            for ai, arm in enumerate(BASELINES):
                for di, dimension in enumerate(('xy','xyz')):
                    require(np.isclose(t['baseline'][ai,di]/count, ref['metrics'][dimension][arm]['full_original_object_mean_epe_m'],
                                       rtol=1e-12, atol=1e-12), 'Recomputed original baseline differs')
            for ti, threshold in enumerate(THRESHOLDS):
                metrics = {}
                for di, dimension in enumerate(('xy','xyz')):
                    epe, gain, cost = (t[key][ti,di]/count for key in ('epe','gain','cost'))
                    cv, blend = t['baseline'][1,di]/count, t['baseline'][2,di]/count
                    require(np.isclose(epe-cv,cost-gain,rtol=1e-12,atol=1e-12), 'Full denominator cost/gain identity')
                    metrics[dimension] = dict(epe_m=float(epe), minus_CV_m=float(epe-cv), minus_full_blend_m=float(epe-blend),
                        positive_gain_m=float(gain), negative_cost_m=float(cost), CV_epe_m=float(cv), full_blend_epe_m=float(blend))
                output.append(dict(horizon_seconds=horizon, group=group, threshold_mps=float(threshold) if np.isfinite(threshold) else None,
                    infinite_threshold=not np.isfinite(threshold), original_objects=count, original_points=int(t['source_count']),
                    uncovered_points=int(t['uncovered_count']), selected_D_points=int(t['selected_count'][ti]),
                    selected_original_object_weight=float(t['selected_mass'][ti]),
                    routing_partition=dict(CV_retained=dict(points=int(t['source_count']-t['uncovered_count']),
                        original_object_weight=float(count-t['uncovered_mass'])),
                        D_selected=dict(points=int(t['selected_count'][ti]), original_object_weight=float(t['selected_mass'][ti])),
                        zero_fallback=dict(points=int(t['uncovered_count']-t['selected_count'][ti]),
                            original_object_weight=float(t['uncovered_mass']-t['selected_mass'][ti]))),
                    selected_fraction_of_uncovered_weight=float(t['selected_mass'][ti]/t['uncovered_mass']) if t['uncovered_mass'] else None,
                    metrics=metrics))
    require('torch' not in sys.modules, 'CPU NumPy-only diagnostic')
    scene_path = a.out.with_suffix('.scene_sums.npz')
    with scene_path.open('xb') as stream:
        np.savez_compressed(stream, counts=counts, scene_tokens=np.asarray(scenes), **totals)
    result = dict(status='COMPLETE_CPU_FIXED_SPEED_ROUTING_TRAIN_DIAGNOSTIC', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__), rules_sha256=sha(a.rules), rules=rules, sources=sources, samples=512, scenes=256,
        source_points=source_points, object_horizon_rows=int(counts.sum()), policies=output,
        scene_sums=dict(file=scene_path.name,sha256=sha(scene_path),axes='horizon,stationary/ambiguous/moving,scene,policy/arm,xy/xyz'),
        verification=dict(all_original_inputs_and_labels_authenticated=True,original_baselines_reproduced=True,
                          direct_vector_and_gain_cost_agree=True,threshold_selected=False,development_read=False,
                          model_forward=False,training=False,optimizer_updates=0,occupancy_result=False),
        synthetic_checks=qa, seconds=time.monotonic()-started)
    with a.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),seconds=result['seconds'],rows=len(output))))


if __name__ == '__main__':
    main()
