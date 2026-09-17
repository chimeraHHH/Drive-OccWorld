"""Train-only dense learned transport diagnostic; no optimizer or box-label input.

All six arms share frozen O, native input/GT and fixed fine-GMO evaluation.
The motion head sees only measured t0 BEV. Its displacement is used at EVERY
source voxel, including regions without motion supervision. No GT safety mask,
flow clipping, threshold search, trained fusion or development evaluation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

SCHEMA = 'learned-transport-probe-v1'
ARMS = ['O', 'static_splat_O0', 'learned_splat_O0', 'blend_static_half',
        'blend_learned_half', 'blend_reverse_horizons_half']
SOURCE_NAMES = ['learned_transport_probe_v1.py', 'oracle_transport_probe.py',
                'transport_ops.py', 'motion_prediction_head.py',
                'train_source_motion_v1.py', 'native_state_cache.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def source_path(name):
    require(Path(name).name == name, 'Source bindings use basenames')
    for p in (Path(__file__).parent / name,
              Path(__file__).parent.parent / 'm0_improvement_20260915' / name):
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def import_source(name, bindings):
    path = source_path(name + '.py')
    require(sha(path) == bindings[path.name], 'Changed source: ' + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def protocol_template():
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', anchors=16,
        sources_sha256={n: None for n in SOURCE_NAMES},
        training=dict(protocol_sha256=None, complete_sha256=None,
                      checkpoint_sha256=None, head_state_sha256=None),
        oracle_reference_records_sha256=None,
        resources=dict(max_seconds=None, max_allocated_gib=None))


def load_contract(a):
    p = read(a.protocol)
    require(p['schema'] == SCHEMA and p['status'] == 'FROZEN', 'Frozen probe protocol required')
    require(a.anchors in (2, 16) and p['anchors'] == a.anchors, 'Only fixed first2/16 training anchors')
    require(p['resources'] == dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib)
            and a.max_seconds > 0 and a.max_allocated_gib > 0, 'Resource cap mismatch')
    require(set(SOURCE_NAMES) <= set(p['sources_sha256']), 'Missing source bindings')
    for name, digest in p['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Changed source: ' + name)
    training = p['training']
    require(sha(a.training_protocol) == training['protocol_sha256'], 'Training protocol changed')
    tp = read(a.training_protocol)
    helper = import_source('train_source_motion_v1', p['sources_sha256'])
    require(tp['schema'] == helper.SCHEMA and tp['status'] == 'FROZEN' and
            tp['training'] == helper.TRAINING, 'Wrong completed training recipe')
    for name, digest in tp['sources_sha256'].items():
        require(p['sources_sha256'][name] == digest == sha(source_path(name)), 'Training source mismatch')
    run = Path(a.motion_run).resolve()
    require(sha(run / 'complete.json') == training['complete_sha256'], 'Wrong completed training run')
    complete = read(run / 'complete.json')
    require(complete['schema'] == helper.SCHEMA and complete['status'] == 'COMPLETE_SOURCE_MOTION_TRAINING'
            and complete['mode'] == 'train' and complete['updates'] == 512
            and complete['examples'] == 2048 and complete['evaluated_samples'] == 200,
            'Motion head is not fixed final512 with completed evaluation')
    expected_files = {'manifest.json', 'training.jsonl', 'summary.json', 'final.pth',
                      'development_objects.jsonl', 'development_records.jsonl'}
    require(set(complete['files_sha256']) == expected_files, 'Incomplete training artifact chain')
    for name, digest in complete['files_sha256'].items():
        require(sha(run / name) == digest, 'Changed completed training artifact: ' + name)
    checkpoint = run / 'final.pth'
    require(training['checkpoint_sha256'] == complete['files_sha256']['final.pth'] ==
            complete['final_checkpoint']['sha256'], 'Motion final checkpoint binding differs')
    require(training['head_state_sha256'] == complete['final_checkpoint']['head_state_sha256'], 'Motion tensor binding differs')
    manifest, summary = read(run / 'manifest.json'), read(run / 'summary.json')
    require(manifest['source_receipt']['protocol_sha256'] == training['protocol_sha256'] and
            manifest['training'] == tp['training'] and manifest['source_receipt']['labels'] == tp['labels'], 'Training manifest provenance differs')
    require(summary['updates'] == 512 and summary['examples'] == 2048 and summary['evaluated_samples'] == 200
            and summary['final_head_sha256'] == training['head_state_sha256'], 'Training summary incomplete')
    # Bind only the old O confusion records, not the privileged oracle fields.
    require(sha(a.oracle_reference_records) == p['oracle_reference_records_sha256'], 'Oracle reference ledger changed')
    reference = []
    for line in Path(a.oracle_reference_records).read_text().splitlines():
        old = json.loads(line)
        require(set(old) == {'ordinal', 'sample_token', 'scene_token', 'hist_by_arm'} and
                set(old['hist_by_arm']) == {'O'}, 'Reference must be the derived O-only ledger, without oracle fields')
        reference.append(dict(ordinal=old['ordinal'], sample_token=old['sample_token'],
            scene_token=old['scene_token'], hist_by_arm={'O': old['hist_by_arm']['O']}))
        del old
    require(len(reference) == 16 and [r['ordinal'] for r in reference] == list(range(16)), 'Expected old fixed16 reference ledger')
    return p, tp, helper, checkpoint, manifest, reference


def load_motion(checkpoint, manifest, protocol, training_protocol, helper, device):
    import torch
    module = import_source('motion_prediction_head', protocol['sources_sha256'])
    payload = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    require(payload['schema'] == helper.SCHEMA and payload['mode'] == 'train' and
            payload['status'] == 'FIXED_FINAL_512' and payload['update'] == 512 and
            payload['examples'] == 2048 and payload['incomplete_accumulation_examples'] == 0, 'Invalid final payload')
    require(payload['sample_orders'] == helper.planned_orders() and
            payload['sample_orders_sha256'] == helper.json_hash(helper.planned_orders()) == manifest['sample_orders_sha256'], 'Final sample orders changed')
    require(payload['sources'] == manifest['source_receipt'] and
            payload['manifest_sha256'] == sha(checkpoint.parent / 'manifest.json'), 'Final checkpoint source/manifest changed')
    require(payload['sources']['train_cache']['index_sha256'] == training_protocol['cache_index_sha256']['train'], 'Motion trained on other measured cache')
    require(payload['initial_head_sha256'] == manifest['initial_head_sha256'], 'Initial head identity differs')
    head = module.MotionPredictionHead(helper.SHAPE)
    head.load_state_dict(payload['head'], strict=True)
    require(helper.state_digest(head) == payload['head_state_sha256'] == protocol['training']['head_state_sha256'], 'Actual final tensor digest differs')
    require(sum(x.numel() for x in head.parameters()) == manifest['head_parameters'] and
            [n for n, _ in head.named_parameters()] == manifest['trainable_names'], 'Head architecture/scope differs')
    del payload
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head.to(device).eval()


def flow_stats(flow):
    import torch
    # Both componentwise absolute values and vector magnitude; all source voxels.
    magnitude = torch.linalg.vector_norm(flow, dim=1).reshape(-1)
    require(bool(torch.isfinite(magnitude).all()), 'Nonfinite displacement magnitude')
    def describe(x):
        x = x.reshape(-1)
        q = torch.quantile(x, x.new_tensor([.5, .9]))
        return dict(median_m=float(q[0]), p90_m=float(q[1]), max_m=float(x.max()))
    return dict(magnitude=describe(magnitude),
        absolute_xyz={axis: describe(flow[:, i].abs()) for i, axis in enumerate('xyz')},
        all_source_voxels=True, finite=bool(torch.isfinite(flow).all()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'o-checkpoint', 'train-cache', 'motion-run',
                 'training-protocol', 'oracle-reference-records', 'protocol', 'repo', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--anchors', type=int, default=16)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--max-allocated-gib', type=float, required=True)
    a = p.parse_args(argv)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); stopping = []
    signal.signal(signal.SIGTERM, lambda sig, frame: stopping.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stopping.append(sig))
    collected = []
    try:
        protocol, training_protocol, helper, motion_checkpoint, training_manifest, reference = load_contract(a)
        native = import_source('native_state_cache', protocol['sources_sha256'])
        oracle = import_source('oracle_transport_probe', protocol['sources_sha256'])
        require(sha(a.checkpoint) == oracle.M0_SHA and sha(a.o_checkpoint) == oracle.O_SHA and
                sha(a.config) == oracle.CONFIG_SHA, 'Original M0/O/config source changed')
        cache_root, all_rows, cache_receipt = helper.cache_index(a.train_cache, 'train', training_protocol)
        require(cache_receipt['index_sha256'] == oracle.TRAIN_INDEX_SHA, 'Not original native observed-state cache')
        rows = all_rows[:a.anchors]
        for i, record in enumerate(rows):
            require(reference[i]['sample_token'] == record['sample_token'] and
                    reference[i]['scene_token'] == record['scene_token'], 'Old oracle O rows not same ordered samples')
        import torch
        ops = import_source('transport_ops', protocol['sources_sha256'])
        torch.set_num_threads(2); torch.manual_seed(11); np.random.seed(11)
        require(str(a.device).startswith('cuda'), 'Native O forward requires CUDA')
        torch.cuda.set_device(a.device); torch.cuda.manual_seed_all(11)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(a.device)
        def budget():
            require(not stopping, 'Signal received; stop without retry')
            require(time.monotonic() - started < a.max_seconds, 'Wall-clock ceiling reached')
            require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib * 2**30, 'CUDA allocation ceiling exceeded')
        budget()
        model = native.build_native_model(a.config, a.checkpoint, device=a.device, repo=a.repo)
        payload = torch.load(a.o_checkpoint, map_location='cpu', weights_only=False)
        model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True); del payload
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        head = load_motion(motion_checkpoint, training_manifest, protocol, training_protocol, helper, a.device)
        model_sha = native._parameter_digest(model)
        motion_sha = helper.state_digest(head)
        require(not any(m.training for m in model.modules()) and not any(m.training for m in head.modules()), 'All modules must be eval')
        manifest = dict(schema=SCHEMA, protocol_sha256=sha(a.protocol), optimizer_updates=0,
            seed=11, sources_sha256=protocol['sources_sha256'], training=protocol['training'],
            source_cache=cache_receipt, m0_sha256=oracle.M0_SHA, O_sha256=oracle.O_SHA,
            model_parameter_sha256=model_sha, motion_head_sha256=motion_sha,
            oracle_reference_records_sha256=sha(a.oracle_reference_records), arms=ARMS,
            anchors=[{k: r[k] for k in ('sample_token', 'scene_token', 'split', 'official_index')} for r in rows],
            split='train', raw_or_sparse_label_files_read=False, future_GT_used_only_for_native_evaluation=True,
            oracle_reference_is_O_only=True,
            only_reference_O_histogram_and_identity_used=True,
            motion_input='inputs.prev_bev_input[:, -1] only; all dense output voxels used',
            probability_source='O t0 softmax foreground probability, no threshold or mask',
            fusion='0.5 native binary log odds + 0.5 transported log odds',
            probability_clamp=[1e-6, 1-1e-6], transported_value='unnormalized trilinear sum, not coverage average',
            output_frame='fixed t0 LiDAR R; O internal coordinates unchanged',
            nominal_horizon_seconds=[0, .5, 1., 1.5, 2.],
            timing_note='Slots are successive dataset future keyframes; actual dt is not exactly nominal and is not a new head input',
            reverse_intervention=[3, 2, 1, 0], reverse_is_not_matched_training_control=True,
            CUDA_scatter_add_may_be_nondeterministic=True, no_development_or_generalization_claim=True,
            resources=protocol['resources'])
        write(out / 'manifest.json', manifest)
        with torch.no_grad(), (out / 'records.jsonl').open('x') as stream:
            for ordinal, record in enumerate(rows):
                budget(); torch.cuda.synchronize(a.device); tick = time.monotonic()
                sample = native.load_sample(cache_root, record, device=a.device)
                native.validate_inputs(sample['inputs'])
                state = sample['inputs']['prev_bev_input']
                require(tuple(state.shape) == (1, 1, 40000, 256) and state.dtype == torch.float32 and
                        native._tensor_digest(state) == record['reference_bev_tensor_sha256'], 'Changed measured t0 BEV')
                pred = native.replay(model, sample, training=False)[0]
                base = oracle.predictions_to_xyz(pred)
                require(native._tensor_digest(oracle.replace_last_logits(pred, base)) == native._tensor_digest(pred), 'Native/XYZ byte roundtrip failed')
                original_hist = oracle.hist(model, pred, sample)
                require(original_hist == reference[ordinal]['hist_by_arm']['O'], 'O does not exactly reproduce oracle16 original O confusion')
                t0_sha = native._tensor_digest(pred[0])
                flow = head(state[:, -1])
                require(tuple(flow.shape) == (1, 4, 3, *oracle.SHAPE) and bool(torch.isfinite(flow).all()), 'Nonfinite/wrong dense displacement')
                require(native._tensor_digest(state) == record['reference_bev_tensor_sha256'], 'Observed input mutated')
                probability = torch.softmax(base[0:1], dim=1)[:, 1:2]
                fields, transport_stats = {}, {}
                for name in ('static', 'learned'):
                    values, stats = [], []
                    for h in range(4):
                        displacement = torch.zeros_like(flow[:, h]) if name == 'static' else flow[:, h]
                        result = ops.forward_splat_3d(probability, displacement, oracle.EXTENT)
                        require(all(bool(torch.isfinite(value).all()) for value in result.values()), 'Nonfinite splat output/statistic')
                        mass = result['numerator']
                        require(mass.shape == probability.shape and bool(torch.isfinite(mass).all()), 'Transport nonfinite or wrong shape')
                        clamped = mass.clamp(1e-6, 1-1e-6)
                        values.append((torch.log(clamped) - torch.log1p(-clamped))[0, 0])
                        stats.append(dict(horizon_slot=h+1, nominal_horizon_seconds=(h+1)*.5,
                            source_probability_mass=float(probability.sum()), destination_mass=float(mass.sum()),
                            outside_dropped_probability_mass=float(result['dropped_feature_sum'].sum()),
                            source_geometric_weight_mass=float(result['source_weight_mass'].sum()),
                            outside_dropped_geometric_weight_mass=float(result['dropped_weight_mass'].sum()),
                            collision_mass_above_one=float((mass-1).clamp_min(0).sum()),
                            coverage_excess_above_one=float((result['coverage']-1).clamp_min(0).sum()),
                            clipped_low_voxels=int((mass < 1e-6).sum()),
                            clipped_high_voxels=int((mass > 1-1e-6).sum()),
                            displacement=flow_stats(displacement), all_numerics_finite=True))
                        del result, mass, clamped, displacement
                    fields[name] = torch.stack(values); transport_stats[name] = stats
                base_delta = base[:, 1] - base[:, 0]
                deltas = dict(static_splat_O0=fields['static'], learned_splat_O0=fields['learned'],
                    blend_static_half=.5*base_delta[1:]+.5*fields['static'],
                    blend_learned_half=.5*base_delta[1:]+.5*fields['learned'],
                    blend_reverse_horizons_half=.5*base_delta[1:]+.5*fields['learned'].flip(0))
                matrices = {'O': original_hist}
                for name, delta in deltas.items():
                    require(bool(torch.isfinite(delta).all()), 'Nonfinite transported log odds')
                    logits = base.clone(); logits[1:, 1] = logits[1:, 0] + delta
                    changed = oracle.replace_last_logits(pred, logits)
                    require(native._tensor_digest(changed[0]) == t0_sha, 't0 byte parity with O failed')
                    matrices[name] = oracle.hist(model, changed, sample)
                    del changed, logits
                require(list(matrices) == ARMS, 'Arm order changed')
                expected_gt = np.asarray(original_hist, np.int64).sum(-1)
                require(all(np.array_equal(np.asarray(v, np.int64).sum(-1), expected_gt)
                            for v in matrices.values()), 'Arm GT support differs')
                torch.cuda.synchronize(a.device)
                row = dict(ordinal=ordinal, sample_token=record['sample_token'], scene_token=record['scene_token'],
                    split='train', official_index=record['official_index'], hist_by_arm=matrices,
                    transport_stats=transport_stats, t0_all_arms_byte_equal_O=True,
                    original_O_confusion_equal_frozen_oracle_reference=True,
                    input_sha256=record['files']['inputs']['sha256'], target_sha256=record['files']['targets']['sha256'],
                    measured_state_sha256=record['reference_bev_tensor_sha256'],
                    seconds=time.monotonic()-tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); collected.append(row)
                print(json.dumps(dict(event='ANCHOR_COMPLETE', completed=len(collected), planned=len(rows), seconds=row['seconds'])), flush=True)
                del sample, state, pred, base, probability, flow, fields, deltas, base_delta, values
                budget()
        require(native._parameter_digest(model) == model_sha and helper.state_digest(head) == motion_sha, 'Frozen model weights changed')
        scores = {}
        for name in ARMS:
            hist = np.asarray([r['hist_by_arm'][name] for r in collected], np.int64).sum(0)
            denominator = hist[:, 1, :].sum(-1) + hist[:, :, 1].sum(-1) - hist[:, 1, 1]
            iou = hist[:, 1, 1] / np.maximum(denominator, 1) * 100
            scores[name] = dict(hist_by_horizon=hist.tolist(), iou_by_horizon_percent=iou.tolist(),
                                future_mean_iou_percent=float(iou[1:].mean()))
        summary = dict(schema=SCHEMA, status='COMPLETE_TRAIN_ONLY_LEARNED_TRANSPORT_DIAGNOSTIC',
            samples=len(rows), scenes=len({r['scene_token'] for r in rows}), scores=scores,
            optimizer_updates=0, inputs_used_GT_boxes_or_flow=False, development_evaluated=False,
            trained_head_evaluated_on_training_samples=True, no_generalization_or_O_motion_EPE_claim=True,
            temporal_reverse_is_not_matched_training_control=True,
            all_O_reference_histograms_equal=True, all_t0_byte_equal=True, all_weights_unchanged=True,
            seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device))
        budget(); write(out / 'summary.json', summary)
        write(out / 'complete.json', dict(schema=SCHEMA, status=summary['status'], samples=len(rows),
            optimizer_updates=0, files_sha256={name: sha(out / name) for name in ('manifest.json', 'records.jsonl', 'summary.json')}))
    except BaseException as exc:
        write(out / 'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY', error=repr(exc),
            samples_completed=len(collected), optimizer_updates=0, seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
