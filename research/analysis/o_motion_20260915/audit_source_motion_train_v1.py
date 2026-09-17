"""CPU-only independent ledger/statistics audit of completed motion training.

Does not load torch, checkpoint payloads, cached tensors or predictions.
Reported model EPEs are reaggregated from authenticated object records;
zero-motion EPE and all scored object support are recomputed from NPZ labels.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

Q = Path(__file__).resolve().parent
P = Q.parent/'m0_improvement_20260915'
GROUPS = ('stationary', 'ambiguous', 'moving')
METRICS = ('epe_3d_m', 'epe_xy_m', 'zero_epe_3d_m', 'zero_epe_xy_m')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    check(np.isfinite(a).all(), 'Nonfinite metric')
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None)


def audit_loss(row, support):
    horizons = []
    for h, cell in enumerate(row['audit']['horizons']):
        check(cell['horizon_seconds'] == (h+1)*.5, 'Loss horizon changed')
        losses = []
        for g, name in enumerate(GROUPS):
            rec = cell['groups'][name]
            expected = support[h][g]
            check(rec['objects'] == expected['objects'] and rec['points'] == expected['points'], 'Loss support differs from frozen NPZ')
            if rec['objects']:
                check(rec['loss'] is not None and math.isfinite(rec['loss']) and rec['loss'] >= 0, 'Bad supported group loss')
                losses.append(rec['loss'])
            else:
                check(rec['loss'] is None, 'Empty group must have null loss')
        check(cell['supported'] == bool(losses), 'Horizon support changed')
        if losses:
            horizons.append(float(np.mean(losses)))
    check(row['audit']['valid_horizons'] == len(horizons) and row['audit']['empty_target'] == (not horizons), 'Empty target accounting')
    expected = float(np.mean(horizons)) if horizons else 0.
    observed = row.get('raw_loss', row.get('loss'))
    # Group reductions were FP32; logged group scalars are reaveraged in FP64.
    check(np.isclose(expected, observed, rtol=2e-6, atol=2e-7), 'Logged hierarchy does not reconstruct loss')
    return abs(expected-observed)


def main():
    out = Q/'source_motion_train_v1_audit.json'
    check(not out.exists(), 'Refuse old audit output')
    root = Q/'server_results/training/source_motion_train_v1'
    pre = Q/'server_results/training/source_motion_preflight_v1'
    job = Q/'server_results/jobs/source_motion_train_v1'
    pre_job = Q/'server_results/jobs/source_motion_preflight_v1'
    protocol_path = Q/'source_motion_protocol_v1.json'
    protocol = read(protocol_path)
    manifest, summary, complete = [read(root/f) for f in ('manifest.json', 'summary.json', 'complete.json')]
    pm, ps, pc = [read(pre/f) for f in ('manifest.json', 'summary.json', 'complete.json')]
    receipt = manifest['source_receipt']
    verified = {}
    for folder, comp in ((root, complete), (pre, pc)):
        for name, digest in comp['files_sha256'].items():
            if name == 'final.pth':
                continue
            check(sha(folder/name) == digest, 'Output SHA mismatch: '+name)
            verified[str((folder/name).relative_to(Q))] = digest
        verified[str((folder/'complete.json').relative_to(Q))] = sha(folder/'complete.json')
    check(complete['status'] == 'COMPLETE_SOURCE_MOTION_TRAINING' and complete['mode'] == 'train', 'Formal status')
    check(pc['status'] == 'PASS_SOURCE_MOTION_PREFLIGHT' and pc['evaluated_samples'] == 0, 'Preflight status')
    check(pc['final_checkpoint'] is None and not ps['weights_persisted'], 'Preflight must not save training weights')
    check(complete['updates'] == summary['updates'] == 512 and complete['examples'] == summary['examples'] == 2048, 'Formal budget')
    check(complete['evaluated_samples'] == summary['evaluated_samples'] == 200, 'Full development count')
    check(receipt['protocol_sha256'] == sha(protocol_path) == pm['source_receipt']['protocol_sha256'], 'Protocol mismatch')
    check(manifest['training'] == protocol['training'] == pm['training'], 'Training recipe changed')
    check(receipt['sources_sha256'] == protocol['sources_sha256'] == pm['source_receipt']['sources_sha256'], 'Source receipt mismatch')
    for name, digest in protocol['sources_sha256'].items():
        path = Q/name if (Q/name).is_file() else P/name
        check(sha(path) == digest, 'Bound local source SHA mismatch: '+name)
        verified[str(path.relative_to(Q.parent))] = digest
    states = []
    for folder in (job, pre_job):
        state, launch = read(folder/'state.json'), read(folder/'launch.json')
        check(state['status'] == 'EXITED_ZERO' and state['returncode'] == 0, 'Job failure')
        check(state['command'] == launch['command'] and state['runner_pid'] == launch['runner_pid'], 'Launch/exit identity mismatch')
        for name, digest in protocol['sources_sha256'].items():
            check(launch['script_sha256'][name] == digest, 'Launch source mismatch')
        check(launch['script_sha256']['source_motion_protocol_v1.json'] == sha(protocol_path), 'Launch protocol mismatch')
        for name in ('state.json', 'launch.json'):
            verified[str((folder/name).relative_to(Q))] = sha(folder/name)
        states.append(state)
    check('--preflight' not in states[0]['command'] and '--preflight' in states[1]['command'], 'Wrong execution mode')
    check(manifest['initial_head_sha256'] == pm['initial_head_sha256'] == summary['initial_head_sha256'] == ps['initial_head_sha256'], 'Fresh initial state differs')
    check(manifest['initial_head_sha256'] != ps['final_head_sha256'], 'Formal appears initialized from preflight final')
    check(not manifest['preflight_weights_eligible_for_formal_initialization'] and not complete['resume_supported'], 'Unexpected resume contract')
    check(summary['final_head_sha256'] == complete['final_checkpoint']['head_state_sha256'], 'Final state receipt mismatch')
    check(complete['files_sha256']['final.pth'] == complete['final_checkpoint']['sha256'], 'Checkpoint receipt mismatch')
    for key in ('full_model_or_checkpoint_loaded', 'occupancy_GT_or_predictions_loaded', 'future_labels_in_model_input', 'best_checkpoint_selection'):
        check(manifest[key] is False, 'Inference/selection boundary changed')

    sparse = Q/'sparse_motion_v1'
    sm, sc = read(sparse/'manifest.json'), read(sparse/'complete.json')
    check(sha(sparse/'manifest.json') == receipt['labels']['manifest_sha256'] == sc['manifest_sha256'], 'Sparse manifest chain')
    check(sha(sparse/'complete.json') == receipt['labels']['complete_sha256'], 'Sparse complete chain')
    binding = {r['identity']['sample_token']: r for r in sm['records']}
    cache_rows = {}
    for split in ('train', 'development'):
        cp = P/'server_results/cache/campaign_cache_v1'/split
        index, cc = read(cp/'index.json'), read(cp/'complete.json')
        check(sha(cp/'index.json') == protocol['cache_index_sha256'][split] == cc['index_sha256'], 'Cache index SHA mismatch')
        check(sha(cp/'complete.json') == receipt[split+'_cache']['complete_sha256'], 'Cache complete SHA mismatch')
        check(sha(cp/'index.json') == receipt[split+'_cache']['index_sha256'], 'Cache receipt mismatch')
        cache_rows[split] = index['records']
        for name in ('index.json', 'complete.json'):
            verified[str((cp/name).relative_to(Q.parent))] = sha(cp/name)
    check([r['sample_token'] for r in cache_rows['train']] == [r['identity']['sample_token'] for r in sm['records'][:512]], 'Train order differs')
    check([r['sample_token'] for r in cache_rows['development']] == [r['identity']['sample_token'] for r in sm['records'][512:]], 'Dev order differs')
    check({r['scene_token'] for r in cache_rows['train']}.isdisjoint({r['scene_token'] for r in cache_rows['development']}), 'Scenes overlap')
    support, zero_values = {}, {}
    for descriptor in sm['records']:
        token = descriptor['identity']['sample_token']
        path = sparse/descriptor['file']
        check(sha(path) == descriptor['sha256'], 'Sparse NPZ hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            check(z['sample_token'].item() == token and z['scene_token'].item() == descriptor['identity']['scene_token'], 'Sparse identity')
            n, k = len(z['source_flat_indices']), len(z['instance_tokens'])
            obj = z['object_index']; valid = z['valid']; groups = z['object_speed_group']
            check(np.array_equal(valid, z['object_future_valid'][:,obj]), 'Point/object mask mismatch')
            check(np.array_equal(groups == -1, ~z['object_future_valid']), 'Speed/mask mismatch')
            check(np.array_equal(np.unique(obj), np.arange(k)), 'Sparse compact object IDs')
            support[token] = [[dict(objects=int(np.count_nonzero(groups[h] == g)),
                                   points=int(np.count_nonzero(valid[h] & (groups[h,obj] == g)))) for g in range(3)] for h in range(4)]
            if descriptor['identity']['split'] == 'development':
                target = z['target_displacement_m'].astype(np.float64)
                for h in range(4):
                    for j, instance in enumerate(z['instance_tokens']):
                        chosen = valid[h] & (obj == j)
                        if not chosen.any():
                            continue
                        zero_values[(token, str(instance), (h+1)*.5)] = dict(
                            group=GROUPS[int(groups[h,j])], source_points=int(chosen.sum()),
                            dt_seconds=float(z['dt_future_seconds'][h]),
                            zero_epe_3d_m=float(np.linalg.norm(target[h,chosen],axis=1).mean()),
                            zero_epe_xy_m=float(np.linalg.norm(target[h,chosen,:2],axis=1).mean()))

    train = lines(root/'training.jsonl'); pretrain = lines(pre/'training.jsonl')
    rng = np.random.RandomState(11)
    orders = [rng.permutation(512).tolist() for _ in range(4)]
    check(jhash(orders) == manifest['sample_orders_sha256'] == pm['sample_orders_sha256'], 'Sample-order digest')
    check(len(train) == 512 and len(pretrain) == 4, 'Wrong update record count')
    flat = [x for order in orders for x in order]
    norms, loss_errors, empty_examples, lr_differences = [], [], [], []
    for u, row in enumerate(train):
        check(row['update'] == u+1 and row['examples'] == 4*(u+1) and row['pass_index'] == u//128, 'Update/accumulation numbering')
        lr = .001*(.1+.9*u/25) if u < 25 else .001*(.1+.45*(1+math.cos(math.pi*(u-25)/486)))
        lr_error = abs(row['lr']-lr)
        # Linux/macOS libm cos can differ at the last bits. Keep the formula,
        # index and endpoints fixed; disclose every nonidentical local scalar.
        check(lr_error <= 4*math.ulp(lr), 'LR schedule mismatch beyond4 float64 ULP')
        if lr_error:
            lr_differences.append(dict(zero_based_update=u,recorded=row['lr'],local=lr,
                                       absolute_difference=lr_error,ulp=lr_error/math.ulp(lr)))
        check(len(row['samples']) == 4 and [s['ordinal'] for s in row['samples']] == flat[4*u:4*u+4], 'Actual sample order mismatch')
        check(row['sample_group_sha256'] == jhash([r['sample_token'] for r in row['samples']]), 'Sample-group hash')
        for sample in row['samples']:
            cr = cache_rows['train'][sample['ordinal']]; desc = binding[sample['sample_token']]
            check(sample['sample_token'] == cr['sample_token'] and sample['scene_token'] == cr['scene_token'], 'Sample identity')
            check(sample['labels_sha256'] == desc['sha256'] and sample['inputs_sha256'] == cr['files']['inputs']['sha256'], 'Sample source SHA')
            loss_errors.append(audit_loss(sample, support[sample['sample_token']]))
            if sample['audit']['empty_target']:
                empty_examples.append(dict(update=u+1, sample_token=sample['sample_token'], raw_loss=sample['raw_loss']))
        check(row['loss'] == float(np.mean([r['raw_loss'] for r in row['samples']])), 'Update loss mean')
        norm = row['preclip_grad_norm']; norms.append(norm)
        check(math.isfinite(norm) and norm > 0 and row['all_gradients_finite'], 'Zero/nonfinite global gradient')
        check(row['clip_factor'] == min(1.,10./(norm+1e-6)), 'Clip factor mismatch')
    first_four_match = all({k:v for k,v in train[i].items() if k not in ('seconds','resources')} ==
                           {k:v for k,v in pretrain[i].items() if k not in ('seconds','resources')} for i in range(4))

    dev = lines(root/'development_records.jsonl'); objects = lines(root/'development_objects.jsonl')
    check(len(dev) == 200, 'Missing development record')
    object_keys = [(r['sample_token'],r['instance_token'],r['horizon_seconds']) for r in objects]
    check(len(set(object_keys)) == len(objects) and set(object_keys) == set(zero_values), 'Scored object support differs from valid labels')
    per_sample = Counter(r['sample_token'] for r in objects)
    cache_map = {r['sample_token']:r for r in cache_rows['development']}
    for index, rec in enumerate(dev):
        cr = cache_rows['development'][index]; desc = binding[rec['sample_token']]
        for key in ('sample_token','scene_token','split','official_index'):
            check(rec[key] == cr[key] == desc['identity'][key], 'Dev sample identity/order')
        check(rec['objects'] == per_sample[rec['sample_token']], 'Dev record/object stream count')
        check(rec['inputs_sha256'] == cr['files']['inputs']['sha256'] and rec['labels_sha256'] == desc['sha256'], 'Dev source SHA')
        loss_errors.append(audit_loss(rec,support[rec['sample_token']]))
    for rec, key in zip(objects,object_keys):
        check(rec['scene_token'] == cache_map[rec['sample_token']]['scene_token'], 'Object scene mismatch')
        for field,value in zero_values[key].items():
            check(rec[field] == value, 'Object support/zero metric mismatch: '+field)
        check(all(math.isfinite(rec[key]) and rec[key] >= 0 for key in METRICS), 'Invalid object EPE')
        check(rec['epe_3d_m'] + 1e-12 >= rec['epe_xy_m'], '3D EPE below XY')
    aggregates, bootstrap = [], []
    for h in (.5,1.,1.5,2.):
        for group in ('all', *GROUPS):
            selected = [r for r in objects if r['horizon_seconds'] == h and (group == 'all' or r['group'] == group)]
            agg = dict(horizon_seconds=h,group=group,objects=len(selected),source_points=sum(r['source_points'] for r in selected),
                       metrics={key:describe([r[key] for r in selected]) for key in METRICS})
            aggregates.append(agg)
            # Paired within-object differences, averaged within each supported scene,
            # then scenes weighted equally. Missing-group scenes are not zero-filled.
            by_scene = defaultdict(list)
            for r in selected:
                by_scene[r['scene_token']].append([r['epe_xy_m']-r['zero_epe_xy_m'],r['epe_3d_m']-r['zero_epe_3d_m']])
            scene_tokens = sorted(by_scene)
            values = np.asarray([np.mean(by_scene[t],axis=0) for t in scene_tokens])
            draw = np.random.RandomState(11).randint(0,len(values),size=(10000,len(values)))
            boot = values[draw].mean(axis=1)
            intervals = np.quantile(boot,[.025,.975],axis=0)
            bootstrap.append(dict(horizon_seconds=h,group=group,supported_scenes=len(scene_tokens),
                unsupported_scenes=100-len(scene_tokens),scene_tokens=scene_tokens,
                mean_model_minus_zero_xy_m=float(values[:,0].mean()),ci95_xy_m=intervals[:,0].tolist(),
                mean_model_minus_zero_3d_m=float(values[:,1].mean()),ci95_3d_m=intervals[:,1].tolist(),
                scene_xy_improvement_count=int(np.count_nonzero(values[:,0]<0))))
    check(aggregates == summary['development']['groups'], 'Independent object statistics mismatch')
    check(len({r['scene_token'] for r in dev}) == 100, 'Development scene count')
    result = dict(schema='source-motion-training-independent-audit-v1',status='PASS',
        audited_script_sha256=sha(Path(__file__)),verified_small_files_sha256=verified,
        source_protocol_sha256=sha(protocol_path),sparse_manifest_sha256=sha(sparse/'manifest.json'),
        sparse_complete_sha256=sha(sparse/'complete.json'),
        execution=dict(updates=512,examples=2048,passes=4,accumulation=4,train_seed=11,
            exact_planned_and_actual_orders=True,LR_formula_all_updates_within4_float64_ULP=True,
            LR_cross_platform_last_bit_differences=lr_differences,
            global_preclip_norm_min=min(norms),global_preclip_norm_max=max(norms),
            globally_nonzero_finite_gradient_updates=len(norms),clipped_updates=sum(x>10 for x in norms),
            max_logged_loss_hierarchy_reconstruction_abs_difference=max(loss_errors),
            empty_target_examples_retained=empty_examples,initial_state_sha256=manifest['initial_head_sha256'],
            same_initial_state_as_preflight=True,initial_state_differs_from_preflight_final=True,
            first_four_logged_updates_match_preflight_excluding_timing=first_four_match,
            preflight_saved_no_weights=True,fresh_optimizer_assertion_and_no_checkpoint_restore_in_bound_source=True,
            formal_runner_seconds=states[0]['seconds'],preflight_runner_seconds=states[1]['seconds'],
            peak_allocated_bytes=complete['resources']['peak_allocated_bytes']),
        checkpoint_receipt_only_not_downloaded=complete['final_checkpoint'],
        evaluation=dict(samples=200,scenes=100,valid_anchor_instance_horizon_records=len(objects),
            exact_valid_label_object_key_set=True,all_zero_EPEs_recomputed_exact=True,
            all_four_metrics_mean_median_p90_exact=True,groups=aggregates,
            source_points_and_masks_verified_from_all712_NPZ=True,
            unique_scored_instance_tokens=len({r['instance_token'] for r in objects})),
        exploratory_scene_bootstrap=dict(repetitions=10000,seed=11,interval='percentile2.5/97.5',
            difference='model minus zero, metres; negative favours model',
            estimand='mean across group-supported scenes of within-scene mean paired anchor-instance EPE difference; each object first averages its valid material points',
            absent_group_scenes='excluded, never zero-filled; reported per group/horizon',
            not_same_as_primary_instance_weighted_mean=True,multiplicity_adjusted=False,
            no_training_randomness_claim=True,not_a_selection_or_promotion_gate=True,results=bootstrap),
        limitations=['No checkpoint payload/cache tensor/prediction tensor was loaded. Model EPEs are reaggregated from authenticated object rows, not recomputed from predicted point vectors.',
            'Runtime per-parameter finite-gradient gates are source/receipt supported; nonzero logged norms certify total gradient, not each parameter nonzero.',
            'Only current unique raw-box grid material points with valid future labels are evaluated; overlapping/background/newborn and no-source boxes excluded.',
            'stationary and ambiguous are legacy group names for endpoint-average global xy speed thresholds, not ground-truth physical stationarity/uncertainty.',
            'No occupancy output or fair common O motion metric was measured; zero baseline is neither O nor GT-history CV.',
            'Historical development exposure and one training seed; exploratory scene intervals do not cover training-seed variability.'])
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='PASS',audit_sha256=sha(out),execution=result['execution'],
        evaluated_object_horizons=len(objects),two_second_instance_summary=[r for r in aggregates if r['horizon_seconds']==2],
        two_second_scene_bootstrap=[r for r in bootstrap if r['horizon_seconds']==2]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
