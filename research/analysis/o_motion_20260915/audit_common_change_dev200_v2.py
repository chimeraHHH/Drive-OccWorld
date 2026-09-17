"""Independent local audit of real O/A/Z common-output evaluation receipts.

Only stdlib/NumPy; no evaluator/aggregator import, model execution, threshold
selection, or checkpoint loading. All statistics start from integer counts.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

N = Path(__file__).resolve().parent
P = N.parent / 'm0_improvement_20260915'
ROOT = N.parents[1]
PROTOCOL = '0752e433d9f37dbd88406099bee4ca4d3af5298b70dafb82d4cf308ec2df30b3'
PRODUCER = 'b73a616c16a1765aeb71e2922c8ba512972b38566f74363a391b0f2e57fc2059'
ARMS = ('O', 'A', 'Z')
GROUPS = ('speed_le_0.1', 'speed_gt_0.1_le_0.5', 'speed_gt_0.5_le_5',
          'speed_gt_5', 'annotated_future_only', 't0_missing_with_history',
          'unknown_no_current_box', 'overlap_current_boxes')
IDENTITY = ('sample_token', 'scene_token', 'split', 'official_index')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def identity(row):
    return {k: row[k] for k in IDENTITY}


def counts(value, shape):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in 'iu'
            and np.all(result >= 0), 'Invalid integer confusion/counts')
    return result.astype(np.int64)


def rates(x):
    """Vectorized pooled estimands; final axis is concatenated sufficient stats."""
    h = x[..., :20].reshape(x.shape[:-1] + (5, 2, 2))
    t = x[..., 20:84].reshape(x.shape[:-1] + (4, 4, 4))
    g = x[..., 84:].reshape(x.shape[:-1] + (4, 8, 3))
    diagonal = np.diagonal(t, axis1=-2, axis2=-1)
    transition_iou = 100 * diagonal / (t.sum(-1) + t.sum(-2) - diagonal)
    moving = g[..., 2:4, :].sum(-2)
    iou = 100 * h[..., 1, 1] / (h[..., 1, 1] + h[..., 0, 1] + h[..., 1, 0])
    result = dict(
        GMO_IoU_percent=iou, future_GMO_mean_percent=iou[..., 1:].mean(-1),
        transition_IoU_percent=transition_iou,
        transition_future_mean_percent=transition_iou.mean(-2),
        moving_gt_point5_recall_percent=100 * moving[..., 1] / moving[..., 0],
        moving_gt_point5_future_mean_recall_percent=(100 * moving[..., 1] / moving[..., 0]).mean(-1),
        global_FP=h[..., 1:, 0, 1], global_FN=h[..., 1:, 1, 0],
        future_global_FP=h[..., 1:, 0, 1].sum(-1),
        future_global_FN=h[..., 1:, 1, 0].sum(-1))
    require(all(np.all(np.isfinite(v)) for v in result.values()), 'Undefined requested pooled metric')
    return result


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit():
    folder = N / 'server_results/training/common_change_dev200_v2'
    training = N / 'server_results/training/supported_fusion_train_v2'
    cache = P / 'server_results/cache/campaign_cache_v1'
    labels = N / 'motion_targets_v1'
    verified = {}

    def bind(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'SHA mismatch: ' + str(path))
        verified[str(path.relative_to(ROOT))] = digest
        return digest

    protocol = read(N / 'supported_fusion_protocol_v2.json')
    bind(N / 'supported_fusion_protocol_v2.json', PROTOCOL)
    require(protocol['status'] == 'FROZEN', 'Protocol not frozen')
    done = read(folder / 'complete.json')
    bind(folder / 'complete.json')
    require(done['status'] == 'COMPLETE_FINAL_DEVELOPMENT_EVALUATION'
            and done['schema'] == 'common-change-evaluation-v1'
            and done['mode'] == 'final_dev200' and done['samples'] == 200
            and done['scenes'] == 100 and done['arms'] == list(ARMS)
            and done['optimizer_updates'] == 0 and done['source_sha256'] == PRODUCER
            and done['protocol_sha256'] == PROTOCOL, 'Completion contract')
    require(not (folder / 'failed.json').exists(), 'Failure receipt present')
    require(set(done['files_sha256']) == {'manifest.json', 'loaded_models.json',
            'records.jsonl', 'summary.json', 'report.md'}, 'Completion files')
    for name, digest in done['files_sha256'].items():
        bind(folder / name, digest)
    manifest = read(folder / 'manifest.json')
    loaded = read(folder / 'loaded_models.json')
    summary = read(folder / 'summary.json')
    state_path = N / 'server_results/jobs/common_change_dev200_v2/state.json'
    state = read(state_path); bind(state_path)
    require(state['status'] == 'EXITED_ZERO' and state['returncode'] == 0, 'Nonzero job exit')
    require(manifest['sources_sha256']['common_change_evaluation_v2.py'] == PRODUCER
            and manifest['protocol_sha256'] == PROTOCOL and manifest['arms'] == list(ARMS)
            and manifest['seed'] == 11 and not manifest['training']
            and not manifest['oracle_inference'] and manifest['optimizer_updates'] == 0
            and manifest['no_threshold_selection']
            and manifest['target_and_raw_box_load_after_all_model_predictions']
            and manifest['numerical_policy'] == protocol['numerical_policy'], 'Manifest contract')
    for name, digest in manifest['sources_sha256'].items():
        bind(N / name if (N / name).exists() else P / name, digest)
    for name, digest in protocol['sources_sha256'].items():
        require(manifest['sources_sha256'][name] == digest, 'Protocol source mismatch')
    bind(P / 'runtime_source_contract.json', manifest['runtime_source_contract_sha256'])
    require(read(P / 'runtime_source_contract.json')['runtime_source_sha256']
            == manifest['runtime_source_sha256'], 'Runtime contract')
    for path, digest in manifest['runtime_source_sha256'].items():
        suffix = path.split('/RadarFlowOcc-sota-p2/', 1)[1]
        bind(ROOT / 'code/Drive-OccWorld-sota-p2' / suffix, digest)

    train_done = read(training / 'complete.json')
    bind(training / 'complete.json', manifest['fusion_complete_sha256'])
    for f, s in train_done['files_sha256'].items(): bind(training / f, s)
    train_manifest = read(training / 'manifest.json')
    require(train_done['status'] == 'COMPLETE_SUPPORTED_FUSION_TRAINING'
            and train_done['updates'] == 512 and train_done['examples'] == 2048
            and train_done['evaluated_samples'] == 200
            and manifest['fixed_final_gate_receipts'] == train_done['final_checkpoints'], 'Final training binding')
    require(loaded['optimizer_updates'] == 0 and not loaded['optimizer_restored']
            and loaded['O_full_state_sha256'] == train_manifest['frozen_O_state_sha256']
            and loaded['motion_head_state_sha256'] == train_manifest['frozen_motion_state_sha256']
            == protocol['motion']['head_state_sha256'], 'Actual loaded frozen states')
    require(loaded['O_head_state_sha256'] == '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
            and loaded['O_head_digest_format'] == 'memory_experiment.state_digest: tuple shape', 'Historical O digest domain')
    require(loaded['native']['checkpoint_sha256'] == manifest['M0_checkpoint_sha256']
            == '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
            and loaded['native']['config_sha256'] == manifest['config_sha256']
            and not loaded['native']['optimizer_loaded'], 'M0/config loader')
    for arm in ('A', 'Z'):
        armroot = training / 'runs' / arm
        arm_done = read(armroot / 'complete.json')
        bind(armroot / 'complete.json', train_done['arm_complete_sha256'][arm])
        for f, s in arm_done['files_sha256'].items():
            if f != 'final.pth': bind(armroot / f, s)
        actual = loaded['gates'][arm]; receipt = train_done['final_checkpoints'][arm]
        require(actual['checkpoint_sha256'] == arm_done['files_sha256']['final.pth'] == receipt['sha256']
                and actual['actual_gate_state_sha256'] == arm_done['final_gate_sha256'] == receipt['gate_state_sha256']
                and actual['manifest_sha256'] == arm_done['files_sha256']['manifest.json']
                and actual['fixed_final_update'] == 512 and not actual['optimizer_restored']
                and actual['optimizer_updates_in_this_evaluation'] == 0, 'Actual loaded gate chain: ' + arm)
    for split, field in [('train', 'train_cache'), ('development', 'sample_cache')]:
        bind(cache / split / 'index.json', manifest[field]['index_sha256'])
        bind(cache / split / 'complete.json', manifest[field]['complete_sha256'])
    dev = read(cache / 'development/index.json')['records']
    train = read(cache / 'train/index.json')['records']
    require(not ({x['scene_token'] for x in train} & {x['scene_token'] for x in dev}), 'Train/dev overlap')
    require(manifest['anchors'] == [identity(x) for x in dev], 'Exact fixed selection')
    bind(labels / 'manifest.json', manifest['raw_labels_manifest_sha256'])
    bind(labels / 'complete.json', manifest['raw_labels_complete_sha256'])
    label_manifest = read(labels / 'manifest.json'); label_done = read(labels / 'complete.json')
    require(label_done['samples'] == 712 and label_done['status'] == 'COMPLETE'
            and label_done['manifest_sha256'] == manifest['raw_labels_manifest_sha256'], 'Raw labels completion')
    descriptors = {r['identity']['sample_token']: r for r in label_manifest['records']}
    require(manifest['raw_label_descriptors'] == [descriptors[x['sample_token']] for x in dev], 'Exact raw labels')
    oref_path = P / 'server_results/campaign_objective_v1/runs/O/development_records.jsonl'
    bind(oref_path, manifest['O_reference_sha256'])
    old_o = jsonl(oref_path)
    old_training = jsonl(training / 'development_records.jsonl')
    rows = jsonl(folder / 'records.jsonl')
    require(len(rows) == len(old_o) == len(old_training) == len(dev) == 200, 'Full 200 coverage')
    require(len({r['sample_token'] for r in rows}) == 200
            and len({r['scene_token'] for r in rows}) == 100, 'Duplicate/missing identities')
    packed = {arm: [] for arm in ARMS}; domain_totals = None; speed_max_error = 0.
    for i, (r, old, o, d) in enumerate(zip(rows, old_training, old_o, dev)):
        ident = identity(r)
        require(r['ordinal'] == i and ident == identity(d) == identity(old)
                and r['sample_token'] == o['sample_token'] and r['scene_token'] == o['scene_token'], 'Ordered identities')
        require(r['hist_by_arm'] == old['hist_by_arm']
                and r['hist_by_arm']['O'] == o['hist_by_horizon'], 'Per-sample prior O/A/Z hist changed')
        require(r['input_sha256'] == d['files']['inputs']['sha256']
                and r['GT_cache_sha256'] == d['files']['targets']['sha256'], 'Cache bytes identity')
        require(all(r[k] is True for k in ('O_reference_hist_exact', 'native_CPU_hist_exact',
                't0_all_models_exact', 'GT_raw_labels_first_loaded_after_all_predictions')), 'Runtime parity flags')
        desc = descriptors[r['sample_token']]
        require(desc['sha256'] == r['raw_label_sha256'] and desc['identity'] == ident, 'Raw source identity')
        path = labels / desc['file']; bind(path, desc['sha256'])
        raw_bytes = gzip.decompress(path.read_bytes())
        require(hashlib.sha256(raw_bytes).hexdigest() == desc['uncompressed_json_sha256'], 'Raw decoded SHA')
        raw = json.loads(raw_bytes); require(raw['identity'] == ident, 'Raw box identity')
        tracks = {x['instance_token']: x for x in raw['tracks']}
        base = r['metrics_by_arm']['O']
        row_domain = []
        for arm in ARMS:
            m = r['metrics_by_arm'][arm]
            require(m['schema'] == 'common-occupancy-change-metrics-v1' and m['identity'] == ident
                    and m['shape_hxyz'] == [5, 512, 512, 40]
                    and m['extent_xyz_m'] == [-51.2, -51.2, -5., 51.2, 51.2, 3.]
                    and m['t0_boundary'] == base['t0_boundary'] and m['contract'] == base['contract'], 'Metric/boundary contract')
            hs = counts(r['hist_by_arm'][arm], (5, 2, 2))
            require([x['occupancy']['confusion'] for x in m['horizons']] == hs.tolist(), 'Native vs CPU counts')
            require(np.array_equal(hs[0], np.asarray(r['hist_by_arm']['O'][0])), 't0 histogram')
            require(np.array_equal(hs.sum(-1), np.asarray(d['target_counts_0_1_255_by_frame'])[2:, :2]), 'Exact GT row support')
            ts = []; gs = []
            for h, item in enumerate(m['horizons']):
                require(item['horizon_index'] == h and item['nominal_seconds'] == h * .5
                        and item['ignored_voxels'] == d['target_counts_0_1_255_by_frame'][h+2][2], 'Horizon/ignore')
                negative = item['global_negative']
                require(negative == dict(GT=int(hs[h, 0].sum()), FP=int(hs[h, 0, 1]), TN=int(hs[h, 0, 0])), 'Global negative counted once')
                if h == 0: continue
                attr = item['motion_positive_attribution']; base_attr = base['horizons'][h]['motion_positive_attribution']
                require(set(attr['groups']) == set(GROUPS), 'All positive groups retained')
                arr = counts([[attr['groups'][g][k] for k in ('GT', 'TP', 'FN')] for g in GROUPS], (8, 3))
                require(np.array_equal(arr[:, 0], arr[:, 1:].sum(-1))
                        and np.array_equal(arr.sum(0), [hs[h, 1].sum(), hs[h, 1, 1], hs[h, 1, 0]])
                        and [attr['groups'][g]['GT'] for g in GROUPS] == [base_attr['groups'][g]['GT'] for g in GROUPS], 'Positive support/count conservation')
                require(attr['unique_box_voxels'] == int(arr[:6, 0].sum())
                        and attr['unknown_voxels'] == arr[6, 0] and attr['overlap_voxels'] == arr[7, 0], 'Attribution coverage')
                ledger = attr['instance_ledger']
                require(len(ledger) == len(tracks) and {e['instance_token'] for e in ledger} == set(tracks), 'All raw union instances retained')
                dt = (raw['frames'][h+2]['timestamp_us'] - raw['frames'][2]['timestamp_us']) / 1e6
                require(item['actual_dt_seconds'] == dt and dt > 0, 'Actual time difference')
                independent_group = {g: np.zeros(3, dtype=np.int64) for g in GROUPS[:6]}
                for e, e0 in zip(ledger, base_attr['instance_ledger']):
                    require({k:v for k,v in e.items() if k not in ('TP','FN','recall')}
                            == {k:v for k,v in e0.items() if k not in ('TP','FN','recall')}, 'Model-dependent GT assignment')
                    t = tracks[e['instance_token']]; mask = t['valid_mask']; speed = None
                    if not mask[h+2]: status = 'missing_current_frame'
                    elif not mask[2]: status = 't0_missing_with_history' if any(mask[:3]) else 'annotated_future_only'
                    else:
                        delta = np.array(t['global_centers_m'][h+2][:2]) - t['global_centers_m'][2][:2]
                        speed = float(np.linalg.norm(delta) / dt)
                        status = GROUPS[0 if speed <= .1 else 1 if speed <= .5 else 2 if speed <= 5 else 3]
                    require(e['status'] == status and e['current_frame_present'] == mask[h+2]
                            and e['t0_frame_present'] == mask[2] and e['TP'] + e['FN'] == e['unique_GT'], 'Raw speed/missing policy')
                    if speed is None: require(e['global_xy_endpoint_speed_mps'] is None, 'Unknown speed filled')
                    else:
                        error = abs(speed - e['global_xy_endpoint_speed_mps']); speed_max_error = max(speed_max_error, error)
                        require(error <= 1e-12 * max(1., speed), 'Raw speed numeric mismatch')
                    if status == 'missing_current_frame': require(e['unique_GT'] == 0, 'Missing geometry assigned')
                    else: independent_group[status] += [e['unique_GT'], e['TP'], e['FN']]
                require(np.array_equal(np.array(list(independent_group.values())), arr[:6]), 'Instance/group conservation')
                gs.append(arr)
                tr = m['transitions'][h-1]; tm = counts(tr['confusion'], (4, 4)); dom = tr['domain']
                require(tr['horizon_index'] == h and tr['nominal_seconds'] == .5*h and tr['actual_dt_seconds'] == dt
                        and tr['row_names'] == tr['column_names'] == ['00','01','10','11'], 'Transition encoding')
                require(dom == base['transitions'][h-1]['domain']
                        and np.array_equal(tm.sum(-1), np.asarray(base['transitions'][h-1]['confusion']).sum(-1)), 'Transition GT domain differs')
                require(int(tm.sum()) == dom['both_valid'] == tr['valid_voxels']
                        and dom['both_valid'] + dom['t0_valid_h_ignored'] + dom['t0_ignored_h_valid'] + dom['both_ignored'] == dom['grid_voxels'], 'Transition joint-domain conservation')
                # These actual 200 records have no 255. Exact endpoint marginals
                # therefore equal original full-domain binary histograms.
                require(dom['t0_valid_h_ignored'] == dom['t0_ignored_h_valid'] == dom['both_ignored'] == 0, 'Unexpected actual ignored-domain scope')
                future = np.zeros((2,2), np.int64); current = future.copy()
                for gt in range(4):
                    for pr in range(4):
                        future[gt % 2, pr % 2] += tm[gt, pr]
                        current[gt // 2, pr // 2] += tm[gt, pr]
                require(np.array_equal(future, hs[h]) and np.array_equal(current, hs[0]), 'Transition endpoint hist mismatch')
                ts.append(tm)
                if arm == 'O': row_domain.append(dom)
            packed[arm].append(np.concatenate([hs.ravel(), np.asarray(ts).ravel(), np.asarray(gs).ravel()]))
        if domain_totals is None: domain_totals = [{k:0 for k in d} for d in row_domain]
        for target, source in zip(domain_totals, row_domain):
            for k,v in source.items(): target[k] += v

    scenes = sorted({r['scene_token'] for r in rows})
    ids = np.array([scenes.index(r['scene_token']) for r in rows])
    draw = np.random.RandomState(11).randint(len(scenes), size=(10000, len(scenes)))
    weights = np.zeros((10000, len(scenes)), np.int64)
    np.add.at(weights, (np.arange(10000)[:, None], draw), 1)
    points = {}; samples = {}; pooled = {}; scene_counts = {}
    for arm in ARMS:
        values = np.asarray(packed[arm], dtype=np.int64)
        grouped = np.zeros((len(scenes), values.shape[-1]), np.int64)
        np.add.at(grouped, ids, values)
        total = grouped.sum(0); points[arm] = rates(total); samples[arm] = rates(weights @ grouped)
        pooled[arm] = dict(occupancy=total[:20].reshape(5,2,2),
            transition=total[20:84].reshape(4,4,4),
            positive_groups=total[84:].reshape(4,8,3))
        scene_counts[arm] = grouped
        require(pooled[arm]['occupancy'].tolist() == [h['occupancy']['confusion'] for h in summary['scores'][arm]['horizons']]
                and pooled[arm]['transition'].tolist() == [t['confusion'] for t in summary['scores'][arm]['transitions']], 'Producer pooled counts')
    contrasts = {}
    for left, right in [('A','O'),('A','Z'),('Z','O')]:
        out = {}
        for key in points[left]:
            distribution = samples[left][key] - samples[right][key]
            interval = np.percentile(distribution, [2.5, 97.5], axis=0)
            out[key] = dict(difference=points[left][key]-points[right][key],
                lower95=interval[0], upper95=interval[1], finite_replicates=10000,
                unit='percentage_points' if 'percent' in key else 'voxel_occurrences')
        contrasts[left+'-minus-'+right] = out
    return serial(dict(schema='common-change-dev200-v2-independent-audit-v1',
        status='PASS_LOCAL_ARTIFACTS_RUNTIME_LOAD_RECEIPTS_AND_INDEPENDENT_COUNTS',
        source_sha256=sha(__file__), python=platform.python_version(), numpy=np.__version__,
        verified_local_files_sha256=verified,
        checks=dict(complete_and_source_chain='PASS', real_job_EXITED_ZERO='PASS',
            actual_loaded_state_vs_final_training_receipts='PASS_RUNTIME_TENSOR_REHASH_RECEIPT',
            all_200_original_O_histograms='EXACT', all_200_A_Z_vs_training_histograms='EXACT',
            all_200_t0_binary_digest_and_native_logits_runtime_assertion='PASS',
            native_vs_CPU_full_histograms='EXACT', group_GT_counts_and_all_foreground_conservation='PASS',
            transition_domains_and_endpoint_histograms='PASS',
            raw_200_label_hashes_and_independent_actual_dt_speed_groups='PASS'),
        samples=200, scenes=100, loaded_models=loaded, pooled_counts=pooled,
        group_order=GROUPS, count_order=['GT','TP','FN'], transition_class_order=['00','01','10','11'],
        scores=points, contrasts=contrasts, transition_domain_totals=domain_totals,
        raw_speed_max_absolute_recompute_error=speed_max_error,
        bootstrap=dict(repetitions=10000,seed=11,unit='scene',scene_order=scenes,
            samples_per_scene=np.bincount(ids,minlength=100),
            method='Percentile; same resampled scene multiplicities for O/A/Z; pool counts then compute ratios; four-future means after per-horizon ratios',
            draw_sha256=hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest()),
        resources=dict(job_seconds=state['seconds'],producer_seconds=done['seconds'],
            peak_allocated_gib=summary['max_peak_allocated_gib']),
        limitations=[
            'Actual checkpoint tensors were rehashed by authenticated completed CUDA evaluator; this local audit checks its receipts, not a second local Torch reload.',
            'Actual fine GT arrays are not locally loaded. Geometry-to-voxel assignment relies on frozen CPU module and authenticated runtime records; raw box speed/missing classification is independently recomputed here.',
            'Previously exposed development200, one training seed11, no multiplicity adjustment or unseen confirmation.',
            'Changes use occupancy states at fixed t0-reference grid locations, not physical flow or ID.',
            'Speed groups use GT unique-box positive voxels and global-XY endpoint speed, not instantaneous speed or object-level detection recall; unassigned/missing groups remain separate.',
            'Transition validity accounting records 255 domain exclusions; physical out-of-ROI motion is not inferable from these inside-ROI counts.',
            'A/Z gates were separately trained and have different transported support; contrast does not identify a unique causal motion mechanism.']))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing overwrite')
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=result['status'], out=str(args.out)), ensure_ascii=False))


if __name__ == '__main__': main()
