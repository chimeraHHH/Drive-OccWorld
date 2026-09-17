"""Independent NumPy recomputation of saved dev vectors, without model inference.

Uses explicit per-object point selection instead of importing the production
scorer. Authenticity is relative to the frozen completed artifact and index;
this does not independently re-run the model or reconstruct raw annotations.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EXPECTED_COMPLETE = 'ec4ebc3ade8616b44c8a3ce216c819b5b7c546d534e853c0722bd4d62cfd8a7c'
H = np.asarray([.5, 1., 1.5, 2.])
T = np.asarray([0., .1, .5, 1., 2., 5., 10., np.inf])
GROUPS = ('stationary', 'ambiguous', 'moving')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(v):
    a = np.ascontiguousarray(v)
    h = hashlib.sha256(json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), 'Never overwrite an audit'
    start = time.monotonic()
    assert sha(a.run/'complete.json') == EXPECTED_COMPLETE
    done = json.loads((a.run/'complete.json').read_text())
    for name, digest in done['files_sha256'].items():
        assert sha(a.run/name) == digest, name
    summary = json.loads((a.run/'summary.json').read_text())
    index = json.loads((a.run/'index.json').read_text())['records']
    refs = {}
    for line in (a.run/'objects.jsonl').read_text().splitlines():
        r = json.loads(line)
        key = (r['sample_token'], r['instance_token'], r['horizon_seconds'])
        assert key not in refs
        refs[key] = r
    assert len(index) == 200 and len(refs) == 16074
    assert {p.relative_to(a.run).as_posix() for p in (a.run/'samples').glob('*.npz')} == {r['file'] for r in index}
    with np.load(a.run/'scene_sums.npz', allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    scenes = saved['scene_tokens'].tolist()
    assert len(scenes) == len(set(scenes)) == 100
    totals = {k: np.zeros_like(v) for k, v in saved.items() if k != 'scene_tokens'}
    maxima = {}
    checks = 0

    def compare(tag, x, y, atol=1e-10):
        nonlocal checks
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape, (tag, x.shape, y.shape)
        assert np.isfinite(x).all() and np.isfinite(y).all(), tag
        err = float(np.max(np.abs(x-y))) if x.size else 0.
        maxima[tag] = max(maxima.get(tag, 0.), err)
        checks += int(x.size)
        assert err <= atol, (tag, err, atol)

    seen = set()
    source_points = 0
    valid_points = np.zeros(4, dtype=np.int64)
    samples = set()
    label_arrays_checked = 0
    for ordinal, row in enumerate(index):
        assert row['ordinal'] == ordinal
        p = a.run/row['file']
        assert sha(p) == row['sha256'], p
        with np.load(p, allow_pickle=False) as z:
            v = {k: z[k].copy() for k in z.files}
        for name, desc in row['original_label_arrays'].items():
            assert v[name].dtype.str == desc['dtype']
            assert list(v[name].shape) == desc['shape']
            assert array_sha(v[name]) == desc['sha256'], (ordinal, name)
            label_arrays_checked += 1
        token = str(v['sample_token'].item())
        scene = str(v['scene_token'].item())
        assert token == row['identity']['sample_token'] and scene == row['identity']['scene_token']
        assert token not in samples
        samples.add(token)
        si = scenes.index(scene)
        d = v['D_displacement_m'].astype(np.float64)
        cv_vel = v['CV_velocity_mps'].astype(np.float64)
        owner = v['owner']
        assert np.isfinite(d).all() and np.isfinite(cv_vel).all()
        assert np.all(cv_vel[owner < 0] == 0), 'Uncovered CV must be zero'
        speed = np.sqrt(np.sum((np.sum(d*H[:, None, None], axis=0)/np.sum(H*H))[:, :2]**2, axis=1))
        compare('LSQ_speed', speed, v['LSQ_speed_xy_mps'])
        gates = (speed[:, None] >= T) & (owner[:, None] < 0)
        assert np.array_equal(gates, v['selected_D_mask'])
        assert len(owner) == row['source_points'] and int((owner >= 0).sum()) == row['covered_points']
        source_points += len(owner)
        valid_points += v['valid'].sum(axis=1)
        for k, instance in enumerate(v['instance_tokens'].tolist()):
            mask = v['object_index'] == k
            n = int(mask.sum())
            assert n > 0
            uncovered = owner[mask] < 0
            selected = gates[mask]
            for hi, horizon in enumerate(H.tolist()):
                if not v['object_future_valid'][hi, k]:
                    continue
                assert v['valid'][hi, mask].all()
                gi = int(v['object_speed_group'][hi, k])
                key = (token, instance, horizon)
                assert key in refs and key not in seen
                seen.add(key)
                r = refs[key]
                assert (r['scene_token'], r['group'], r['source_points'], r['covered_points']) == (scene, GROUPS[gi], n, int((~uncovered).sum()))
                y = v['target_displacement_m'][hi, mask].astype(np.float64)
                dd = d[hi, mask]
                cc = horizon*cv_vel[mask]
                blend = np.where(uncovered[:, None], dd, cc)
                def err(field):
                    delta = field-y
                    return np.stack((np.sqrt(np.sum(delta[:, :2]**2, axis=1)), np.sqrt(np.sum(delta**2, axis=1))), axis=1)
                de, ce = err(dd), err(cc)
                baselines = np.stack([de.mean(0), ce.mean(0), err(blend).mean(0), err(np.zeros_like(dd)).mean(0)])
                errors, gains, costs = [], [], []
                for ti in range(8):
                    routed = cc.copy()
                    routed[selected[:, ti]] = dd[selected[:, ti]]
                    errors.append(err(routed).mean(0))
                    improvement = (ce-de)*selected[:, ti, None]
                    gains.append(np.maximum(improvement, 0).mean(0))
                    costs.append(np.maximum(-improvement, 0).mean(0))
                obj = dict(epe=np.asarray(errors), gain=np.asarray(gains), cost=np.asarray(costs), baseline=baselines,
                           selected_count=selected.sum(0), selected_mass=selected.mean(0),
                           uncovered_count=uncovered.sum(), uncovered_mass=uncovered.mean(), source_count=n)
                for tag, savedkey in [('epe', 'routed_epe_xy_xyz_m'), ('gain', 'positive_gain_xy_xyz_m'), ('cost', 'negative_cost_xy_xyz_m'),
                                      ('selected_count', 'selected_D_points'), ('selected_mass', 'selected_original_object_weight')]:
                    compare('object_'+tag, obj[tag], r[savedkey])
                for ai, arm in enumerate(('D', 'CRN_CV', 'CV_covered_D_uncovered')):
                    for di, dim in enumerate(('xy', '3d')):
                        compare('object_'+arm+'_'+dim, baselines[ai, di], r['reevaluated_GPU_baselines'][arm]['epe_'+dim+'_m'])
                        compare('old_GPU_reference_'+arm+'_'+dim, baselines[ai, di], r['GPU_reference_arms'][arm]['epe_'+dim+'_m'], 1e-10 if ai == 1 else 1e-4)
                totals['counts'][hi, gi, si] += 1
                for name, val in obj.items():
                    totals[name][hi, gi, si] += val
        if (ordinal+1) % 50 == 0:
            print(json.dumps({'samples_audited': ordinal+1, 'objects_audited': len(seen)}), flush=True)
    assert seen == set(refs) and source_points == summary['source_points']
    assert valid_points.tolist() == summary['original_valid_points_by_horizon']
    for name, value in totals.items():
        compare('scene_'+name, value, saved[name], 1e-8)
    assert len(summary['policies']) == 128
    for r in summary['policies']:
        hi = H.tolist().index(r['horizon_seconds'])
        ti = 7 if r['threshold_mps'] is None else T.tolist().index(r['threshold_mps'])
        gi = [0,1,2] if r['group'] == 'all' else [GROUPS.index(r['group'])]
        n = int(totals['counts'][hi, gi].sum())
        t = {key: val[hi, gi].sum(axis=(0,1)) for key, val in totals.items() if key != 'counts'}
        assert n == r['original_objects'] and int(t['source_count']) == r['original_points']
        assert int(t['uncovered_count']) == r['uncovered_points'] and int(t['selected_count'][ti]) == r['selected_D_points']
        compare('policy_selected_weight', t['selected_mass'][ti], r['selected_original_object_weight'])
        for di, dim in enumerate(('xy', 'xyz')):
            e, gain, cost = [t[key][ti, di]/n for key in ('epe','gain','cost')]
            cv, blend = t['baseline'][1,di]/n, t['baseline'][2,di]/n
            vals = dict(epe_m=e, positive_gain_m=gain, negative_cost_m=cost, CV_epe_m=cv, full_blend_epe_m=blend,
                        minus_CV_m=e-cv, minus_full_blend_m=e-blend)
            for name, val in vals.items():
                compare('policy_'+name, val, r['metrics'][dim][name])
    report = dict(status='PASS_INDEPENDENT_SAVED_VECTOR_RECOMPUTATION', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256=sha(__file__), complete_sha256=EXPECTED_COMPLETE, samples=len(samples), scenes=len(scenes),
                  source_points=source_points, object_horizon_rows=len(seen), policy_rows=128, original_label_arrays_sha_checked=label_arrays_checked,
                  scalar_comparisons=checks, max_absolute_difference_by_check=maxima, seconds=time.monotonic()-start,
                  method='Independent explicit per-object NumPy vector EPE and LSQ/gates; production scorer not imported',
                  limitations=['Recomputed from authenticated saved vectors; no independent model inference',
                               'Raw original dataset/annotation generation not independently repeated',
                               'No dense full-grid rerun or occupancy evaluation; no new policy selected'],
                  model_forward=False, optimizer_updates=0)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
