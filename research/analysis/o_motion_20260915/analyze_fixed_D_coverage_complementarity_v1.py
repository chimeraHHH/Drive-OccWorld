"""CPU-only analysis of the complete, fixed D/CV coverage diagnostic.

No model, Torch, fitting, threshold selection or new evaluation support.
Full zero EPE is in producer rows; zero subset means come from the frozen
CRN-CV reference. Subset contributions always divide the ORIGINAL point count.
CLI: --run COMPLETE_RUN --protocol FROZEN_JSON --cv-reference CV_V2 --out NEW.json
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = 'fixed-D-coverage-complementarity-analysis-v1'
PRODUCER_SCHEMA = 'fixed-D-coverage-complementarity-v1'
PRODUCER_SHA = '1a730762d74dda2fcf367d19c80a159e212a9be8884a4cb33e605e9341006211'
PROTOCOL_SHA = '8a0c3ba31a308d5acdf99019bcdac3ca4487d76078f745b3475c14666fd45431'
REFERENCE_COMPLETE_SHA = 'edfdf0b38145ad0b1d5fb69370ba8e8d91fda973f33540c1f5149917f58077f6'
REFERENCE_SUMMARY_SHA = 'ba67afc75b02ef481b0846914207fc01c6d04a4d0c675d3a7c00d1129b81141e'
D_STATE_SHA = '4abd5bf8aab3fff045d4ea9f4c5d7bc3579f7265f795b4aef87eb5774afadb61'
D_CHECKPOINT_SHA = '7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
PRODUCER_ARMS = ('D', 'CRN_CV', 'CV_covered_D_uncovered')
ARMS = ('D', 'CRN_CV', 'fixed_blend', 'zero')
RULES = dict(
    schema=SCHEMA, horizons_seconds=[.5, 1., 1.5, 2.],
    groups=['all', 'moving', 'ambiguous', 'stationary'], arms=list(ARMS),
    fixed_blend_producer_name='CV_covered_D_uncovered',
    full_support=dict(anchors=200, scenes=100, object_horizon_rows=16074),
    full_mean='mean point Euclidean error within each original anchor-instance, then equal anchor-instance mean',
    conditional_mean='mean subset-point error within each nonempty anchor-instance subset, then equal nonempty anchor-instance mean',
    contribution='subset point-error sum / original object point count, then mean over ALL original anchor-instances',
    empty_subset=dict(conditional_mean=None, error_sum=0., full_denominator_contribution=0.),
    zero_full_source='producer row zero_epe_xy_m / zero_epe_3d_m',
    zero_subset_source='frozen CRN-CV v2 coverage_object_records: zero subset mean times subset count; no raw point-norm recomputation',
    bootstrap=dict(comparison='fixed_blend-minus-CRN_CV', unit='scene', paired=True,
                   repetitions=10000, seed=11, interval='percentile95',
                   preserve_all_natural_objects_within_scene=True,
                   equal_scene_means=False, multiplicity_adjusted=False,
                   training_seed_uncertainty_included=False),
    validation=dict(reference_rtol=1e-12, reference_atol_m=1e-10),
    restrictions=dict(no_model_or_threshold_selection=True, no_candidate_or_occupancy_claim=True,
                      no_O_flow=True, historical_development_exposure=True,
                      rigid_box_proxy_not_measured_scene_flow=True))
RULES_PATH = HERE / 'analyze_fixed_D_coverage_complementarity_rules_v1.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ledger(root, entries, names=None):
    if names is not None:
        require(set(entries) == set(names), 'Completion file set differs')
    for name, digest in entries.items():
        require(Path(name).name == name and sha(root / name) == digest,
                'Bound artifact differs: ' + name)


def key(row):
    return (row['sample_token'], row['instance_token'], row['horizon_seconds'])


def index(rows):
    result = {key(r): r for r in rows}
    require(len(result) == len(rows) == 16074, 'Require all 16074 unique object-horizon rows')
    return result


def near(actual, expected, message):
    if actual is None or expected is None:
        require(actual is expected, 'Null denominator differs: ' + message)
    else:
        require(math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                'Numeric identity differs: ' + message)


def nonnegative(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value >= 0, 'Invalid error/count value')
    return float(value)


def authenticate(run, protocol_path, reference):
    require(read(RULES_PATH) == RULES, 'Prospective analysis rules changed')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Wrong frozen diagnostic protocol')
    protocol = read(protocol_path)
    require(protocol['schema'] == PRODUCER_SCHEMA and protocol['status'] == 'FROZEN' and
            protocol['sources_sha256']['fixed_D_coverage_complementarity_v1.py'] == PRODUCER_SHA,
            'Producer source/protocol mismatch')
    require(not (run / 'failed.json').exists(), 'Failed diagnostic is not analyzable')
    done = read(run / 'complete.json')
    require(done['schema'] == PRODUCER_SCHEMA and done['status'] == 'COMPLETE_FIXED_D_COVERAGE_DIAGNOSTIC' and
            (done['samples'], done['scenes'], done['object_horizon_rows'], done['optimizer_updates']) == (200, 100, 16074, 0) and
            done['script_sha256'] == PRODUCER_SHA and done['protocol_sha256'] == PROTOCOL_SHA,
            'Require completed original fixed-D endpoint')
    ledger(run, done['files_sha256'], ('manifest.json', 'objects.jsonl', 'summary.json'))
    manifest, summary = read(run / 'manifest.json'), read(run / 'summary.json')
    require(manifest['schema'] == summary['schema'] == PRODUCER_SCHEMA and
            manifest['protocol_sha256'] == PROTOCOL_SHA and manifest['sources_sha256'] == protocol['sources_sha256'] and
            manifest['recipe'] == protocol['recipe'] and manifest['resources'] == protocol['resources'] and
            manifest['optimizer_updates'] == 0, 'Manifest differs from frozen protocol')
    for field in ('selection_sha256', 'runtime_contract_sha256', 'connected_protocol_sha256',
                  'reference_complete_sha256', 'reference_summary_sha256'):
        require(manifest[field] == protocol[field], 'Manifest source differs: ' + field)
    require(manifest['geometry_cache']['complete_sha256'] == protocol['geometry_complete_sha256'],
            'Geometry coverage source changed')
    d = manifest['D_loader']
    require(d['arm'] == 'D' and d['top_complete_sha256'] == protocol['d_complete_sha256'] and
            d['protocol_sha256'] == protocol['connected_protocol_sha256'] and
            d['checkpoint_sha256'] == D_CHECKPOINT_SHA and d['actual_motion_state_sha256'] == D_STATE_SHA and
            (d['fixed_final_update'], d['fixed_examples'], d['fixed_development_samples']) == (512, 2048, 200) and
            d['actual_optimizer_parameter_steps_all512'] and d['optimizer_updates_in_this_evaluation'] == 0 and
            not d['optimizer_restored'], 'Wrong fixed D load receipt')
    require(summary['status'] == done['status'] and summary['arms'] == list(PRODUCER_ARMS) and
            (summary['samples'], summary['scenes'], summary['object_horizon_rows']) == (200, 100, 16074) and
            summary['reference_parity_passed'] and summary['D_state_unchanged'] and summary['D_gradients_all_None'] and
            summary['optimizer_updates'] == 0, 'Incomplete/failing producer summary')
    require(sha(reference / 'complete.json') == REFERENCE_COMPLETE_SHA and
            sha(reference / 'summary.json') == REFERENCE_SUMMARY_SHA, 'Wrong original CV reference')
    refdone = read(reference / 'complete.json'); ledger(reference, refdone['files_sha256'])
    ref = read(reference / 'summary.json')
    require(refdone['status'] == 'COMPLETE_READ_ONLY_DIAGNOSTIC' and
            (ref['samples'], ref['scenes'], ref['object_horizon_rows_per_arm']) == (200, 100, 16074), 'Incomplete CV reference')
    cov = index(ref['coverage_object_records'])
    original = {a: index([r for r in ref['physical_object_records'] if r['arm'] == a]) for a in ('D', 'CRN_CV')}
    data = index([json.loads(line) for line in (run / 'objects.jsonl').read_text().splitlines() if line.strip()])
    require(set(data) == set(cov) == set(original['D']) == set(original['CRN_CV']), 'Original complete support differs')
    anchors = ref['per_anchor_coverage']; actual = summary['per_anchor']
    require(len(anchors) == len(actual) == 200, 'All anchors required')
    for i, (a, b) in enumerate(zip(actual, anchors)):
        require(a['ordinal'] == b['ordinal'] == i and a['identity'] == b['identity'] and
                a['source_points'] == b['source_points'] and a['covered_points'] == b['covered_points'],
                'Original anchor identity/coverage differs')
    token_scene = {r['identity']['sample_token']: r['identity']['scene_token'] for r in anchors}
    require(len(token_scene) == 200 and len(set(token_scene.values())) == 100 and
            {r['sample_token'] for r in data.values()} == set(token_scene), 'Original anchor/scene set differs')
    sources = dict(diagnostic_complete_sha256=sha(run / 'complete.json'), diagnostic_files_sha256=done['files_sha256'],
                   protocol_sha256=PROTOCOL_SHA, producer_sha256=PRODUCER_SHA,
                   reference_complete_sha256=REFERENCE_COMPLETE_SHA, reference_summary_sha256=REFERENCE_SUMMARY_SHA,
                   rules_sha256=sha(RULES_PATH), analysis_sha256=sha(__file__))
    return data, cov, original, token_scene, sources


def prepare(data, coverage, original, token_scene):
    """Validate original full support and normalize four arms, without filtering."""
    result = []
    for k in sorted(data):
        row, ref = data[k], coverage[k]
        require(row['scene_token'] == token_scene[row['sample_token']] and
                row['horizon_seconds'] in RULES['horizons_seconds'] and
                row['group'] in RULES['groups'][1:], 'Unexpected identity/horizon/group')
        for name in ('scene_token', 'dt_seconds', 'group', 'source_points', 'covered_points'):
            require(row[name] == ref[name], 'Original coverage support changed: ' + name)
        n = row['source_points']; nc = row['covered_points']
        require(type(n) is int and n > 0 and type(nc) is int and 0 <= nc <= n and
                set(row['arms']) == set(PRODUCER_ARMS) and set(row['parts']) == {'covered', 'uncovered'}, 'Invalid row layout')
        normalized = dict(key=k, scene=row['scene_token'], horizon=k[2], group=row['group'], n=n, nc=nc, values={})
        for dim, refdim in (('xy', 'xy_m'), ('3d', 'xyz_m')):
            normalized['values'][dim] = {}
            for arm in ARMS:
                source_arm = 'CV_covered_D_uncovered' if arm == 'fixed_blend' else arm
                full = nonnegative(row['zero_epe_' + dim + '_m'] if arm == 'zero' else row['arms'][source_arm]['epe_' + dim + '_m'])
                if arm in ('D', 'CRN_CV', 'zero'):
                    prior = original['D'] if arm == 'zero' else original[arm]
                    near(full, prior[k][('zero_' if arm == 'zero' else '') + 'epe_' + dim + '_m'], 'original full EPE')
                parts = {}
                for part, count in (('covered', nc), ('uncovered', n - nc)):
                    require(row['parts'][part]['source_points'] == ref[part]['source_points'] == count, 'Partition count changed')
                    if arm == 'zero':
                        conditional = ref[part]['zero'][refdim]
                        require((conditional is None) == (count == 0), 'Reference empty subset differs')
                        total = nonnegative(conditional) * count if count else 0.
                        contribution = total / n
                    else:
                        r = row['parts'][part][source_arm]
                        total = nonnegative(r[dim + '_error_sum_m'])
                        contribution = nonnegative(r[dim + '_contribution_m'])
                        conditional = r[dim + '_subset_mean_m']
                        near(contribution, total / n, 'FULL denominator contribution')
                        near(conditional, total / count if count else None, 'conditional mean')
                        if not count: require(total == contribution == 0., 'Empty subset must contribute zero')
                        if arm == 'CRN_CV': near(conditional, ref[part]['CRN_CV'][refdim], 'original CV subset mean')
                    parts[part] = dict(count=count, conditional_mean_m=conditional, contribution_m=contribution)
                near(full, sum(p['contribution_m'] for p in parts.values()), 'partition reconstructs full EPE')
                normalized['values'][dim][arm] = dict(full_mean_m=full, **parts)
            v = normalized['values'][dim]
            for part, arm in (('covered', 'CRN_CV'), ('uncovered', 'D')):
                near(v['fixed_blend'][part]['contribution_m'], v[arm][part]['contribution_m'], 'prespecified blend branch')
                near(v['fixed_blend'][part]['conditional_mean_m'], v[arm][part]['conditional_mean_m'], 'prespecified blend subset')
            near(v['CRN_CV']['uncovered']['conditional_mean_m'], v['zero']['uncovered']['conditional_mean_m'], 'uncovered CV equals zero')
        result.append(normalized)
    return result


def aggregate(rows, scenes):
    lookup = {s: i for i, s in enumerate(scenes)}
    draws = np.random.RandomState(11).randint(len(scenes), size=(10000, len(scenes)))
    weights = np.stack([np.bincount(r, minlength=len(scenes)) for r in draws]).astype(np.int64)
    bins = []
    for h in RULES['horizons_seconds']:
        for group in RULES['groups']:
            chosen = [r for r in rows if r['horizon'] == h and (group == 'all' or r['group'] == group)]
            require(chosen, 'Missing original horizon/group')
            ids = np.asarray([lookup[r['scene']] for r in chosen], dtype=np.int64)
            counts = np.bincount(ids, minlength=len(scenes)); sample_counts = weights @ counts
            item = dict(horizon_seconds=h, group=group, original_object_count=len(chosen),
                        original_point_count=sum(r['n'] for r in chosen), scenes_with_support=int((counts > 0).sum()),
                        covered_point_count=sum(r['nc'] for r in chosen),
                        uncovered_point_count=sum(r['n'] - r['nc'] for r in chosen), metrics={})
            for dim in ('xy', '3d'):
                values = {}; arrays = {}
                for arm in ARMS:
                    arrays[arm] = np.asarray([r['values'][dim][arm]['full_mean_m'] for r in chosen], dtype=np.float64)
                    desc = dict(full_object_mean_epe_m=float(arrays[arm].mean()), parts={})
                    for part in ('covered', 'uncovered'):
                        parts = [r['values'][dim][arm][part] for r in chosen]
                        subset = [p['conditional_mean_m'] for p in parts if p['count'] > 0]
                        desc['parts'][part] = dict(nonempty_object_count=len(subset),
                            point_count=sum(p['count'] for p in parts),
                            conditional_equal_nonempty_object_mean_epe_m=float(np.mean(subset)) if subset else None,
                            FULL_original_object_denominator_contribution_mean_m=float(np.mean([p['contribution_m'] for p in parts])))
                    near(desc['full_object_mean_epe_m'], sum(p['FULL_original_object_denominator_contribution_mean_m'] for p in desc['parts'].values()), 'aggregate decomposition')
                    values[arm] = desc
                difference = arrays['fixed_blend'] - arrays['CRN_CV']
                scene_difference_sums = np.bincount(ids, weights=difference, minlength=len(scenes))
                valid = sample_counts > 0
                samples = (weights[valid] @ scene_difference_sums) / sample_counts[valid]
                ci = np.quantile(samples, [.025, .975]) if len(samples) else [None, None]
                item['metrics'][dim] = dict(arms=values, fixed_blend_minus_CRN_CV=dict(
                    difference_m=float(difference.mean()), lower95_m=None if ci[0] is None else float(ci[0]),
                    upper95_m=None if ci[1] is None else float(ci[1]),
                    defined_draws=int(valid.sum()), undefined_draws=int((~valid).sum())))
            bins.append(item)
    return bins, hashlib.sha256(weights.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'protocol', 'cv-reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'New output required; never overwrite a previous analysis')
    data, cov, original, token_scene, sources = authenticate(args.run, args.protocol, args.cv_reference)
    normalized = prepare(data, cov, original, token_scene)
    scenes = sorted(set(token_scene.values())); bins, weight_sha = aggregate(normalized, scenes)
    require('torch' not in sys.modules, 'CPU-only analysis must not import Torch')
    result = dict(schema=SCHEMA, status='COMPLETE_CPU_RECORD_ANALYSIS',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  sources=sources, rules=RULES, samples=200, scenes=100, object_horizon_rows=16074,
                  scene_tokens=scenes, bootstrap_weights_sha256=weight_sha, bins=bins,
                  runtime=dict(python=sys.version, numpy=np.__version__, Torch_imported=False,
                               model_forward=False, optimizer_updates=0),
                  verification=dict(original_full_support_and_reference_rows_verified=True,
                                    original_D_and_CV_full_means_reproduced=True,
                                    all_partition_counts_means_sums_reconstructed=True,
                                    fixed_blend_uses_only_preassigned_owner=True,
                                    checkpoint_tensors_reloaded_by_this_analysis=False))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(out=str(args.out), sha256=sha(args.out), bins=len(bins), samples=200)))


if __name__ == '__main__':
    main()
