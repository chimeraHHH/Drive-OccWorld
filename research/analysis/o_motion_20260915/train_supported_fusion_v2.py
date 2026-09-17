"""Matched A/Z gate training on frozen O and frozen source-motion predictions.

A transports O's predicted current foreground with learned displacement;
Z supplies zero displacement at the same transport boundary. Only each arm's
independent 97-parameter gate/AdamW is trained. Original coarse CE+Lovasz are
selected without reimplementing loss formulas. No box/motion labels are read.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import signal
import struct
import sys
import time

import numpy as np

SCHEMA = 'supported-fusion-training-v1'
ARMS = ('A', 'Z')
SOURCES = ('train_supported_fusion_v2.py', 'supported_motion_fusion.py', 'transport_ops.py',
           'train_source_motion_v1.py', 'motion_prediction_head.py', 'learned_transport_probe_v1.py',
           'native_state_cache.py', 'oracle_transport_probe.py', 'objective_supervision_adapters.py')
TRAINING = dict(seed=11, train_samples=512, development_samples=200, passes=4, accumulate=4,
    updates=512, parameters_per_arm=97, lr=1e-3, weight_decay=.01, grad_clip=35.,
    warmup_updates=25, final_lr_ratio=.1, optimizer='AdamW', betas=[.9, .999], eps=1e-8,
    order='continuous_numpy_RandomState11_permutation_each_pass',
    lr_schedule='train_source_motion_v1.schedule_lr; zero_based0=.0001,25=.001,511=.0001',
    loss='unchanged native coarse twelve computed; select_objective O returns CE[1,5]+Lovasz six',
    scope='independent_supported_fusion_gate_only; O and pretrained motion frozen',
    transport='A predicted displacement; Z zero displacement; same predicted-current-foreground support',
    checkpoint='fixed final512 only; no resume, best or initial-development evaluation')


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


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')


def source_path(name):
    require(Path(name).name == name, 'Source must be a basename')
    for path in (Path(__file__).parent / name,
                 Path(__file__).parent.parent / 'm0_improvement_20260915' / name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def import_source(name, bindings):
    path = source_path(name + '.py')
    require(sha(path) == bindings[path.name], 'Changed import: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protocol_template():
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', training=dict(TRAINING),
        sources_sha256={n: None for n in SOURCES},
        numerical_policy=dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False),
        motion=dict(protocol_sha256=None, complete_sha256=None, checkpoint_sha256=None, head_state_sha256=None),
        cache_index_sha256=dict(train=None, development=None), o_development_records_sha256=None,
        resources=dict(preflight=dict(max_seconds=600, max_allocated_gib=32),
                       train=dict(max_seconds=7200, max_allocated_gib=32)))


def load_contract(a):
    p = read(a.protocol)
    require(p['schema'] == SCHEMA and p['status'] == 'FROZEN' and p['training'] == TRAINING, 'Frozen fixed recipe required')
    require(set(SOURCES) <= set(p['sources_sha256']), 'Missing source bindings')
    for name, digest in p['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Source changed: ' + name)
    mode = 'preflight' if a.preflight else 'train'
    require(p['resources'][mode] == dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib)
            and a.max_seconds > 0 and a.max_allocated_gib > 0, 'Resource mismatch')
    require(p['numerical_policy'] == dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False), 'Precision changed')
    require(sha(a.motion_training_protocol) == p['motion']['protocol_sha256'], 'Motion protocol changed')
    tp = read(a.motion_training_protocol)
    helper = import_source('train_source_motion_v1', p['sources_sha256'])
    require(tp['schema'] == helper.SCHEMA and tp['status'] == 'FROZEN' and tp['training'] == helper.TRAINING, 'Motion recipe changed')
    require(tp['cache_index_sha256'] == p['cache_index_sha256'], 'Observed states differ from motion pretraining')
    for name, digest in tp['sources_sha256'].items():
        require(p['sources_sha256'][name] == digest, 'Motion source binding changed')
    root = Path(a.motion_run).resolve()
    require(sha(root / 'complete.json') == p['motion']['complete_sha256'], 'Motion completion changed')
    done = read(root / 'complete.json')
    require(done['schema'] == helper.SCHEMA and done['status'] == 'COMPLETE_SOURCE_MOTION_TRAINING'
            and done['mode'] == 'train' and done['updates'] == 512 and done['examples'] == 2048
            and done['evaluated_samples'] == 200, 'Incomplete motion pretraining')
    expected = {'manifest.json', 'training.jsonl', 'summary.json', 'final.pth',
                'development_objects.jsonl', 'development_records.jsonl'}
    require(set(done['files_sha256']) == expected, 'Motion artifact chain differs')
    for name, digest in done['files_sha256'].items():
        require(sha(root / name) == digest, 'Motion artifact changed: ' + name)
    require(done['final_checkpoint']['sha256'] == p['motion']['checkpoint_sha256'] == done['files_sha256']['final.pth']
            and done['final_checkpoint']['head_state_sha256'] == p['motion']['head_state_sha256'], 'Wrong fixed final motion')
    manifest = read(root / 'manifest.json')
    require(manifest['source_receipt']['protocol_sha256'] == p['motion']['protocol_sha256'] and
            manifest['training'] == tp['training'], 'Motion manifest recipe differs')
    return p, tp, helper, root / 'final.pth', manifest


def exact32(a, b):
    """Value representation equality, including signed zero; all data are FP32."""
    import torch
    return a.dtype == b.dtype == torch.float32 and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def compose_prediction(original, base_xyz, fusion, oracle, disabled=False):
    """Only last decoder/future foreground logit changes, and only at W>0."""
    import torch
    if disabled:
        changed = original.clone()
    else:
        logits = base_xyz.clone()
        foreground = base_xyz[1:, 0] + fusion['logodds'][0, :, 0]
        # Subtracting then adding the original margin can round. Preserve the
        # original class1 bytes directly at every uncovered destination.
        logits[1:, 1] = torch.where(fusion['destination_support'][0, :, 0], foreground, base_xyz[1:, 1])
        changed = oracle.replace_last_logits(original, logits)
    require(exact32(changed[0], original[0]) and exact32(changed[:, :-1], original[:, :-1]), 't0/intermediate decoder bytes changed')
    require(exact32(changed[:, -1, ..., 0], original[:, -1, ..., 0]), 'Background logit changed')
    if not disabled:
        missing = ~fusion['destination_support'][0, :, 0]
        require(exact32(oracle.predictions_to_xyz(changed)[1:, 1][missing], base_xyz[1:, 1][missing]),
                'Uncovered foreground logit bytes changed')
    return changed


def loss_terms(model, prediction, targets, adapter):
    """Use exactly the original functions/tensors; no numerical or GT rewrite."""
    import torch
    losses = model.compute_occ_loss(prediction, targets)
    selected = adapter.select_objective(losses, 'O')
    require(all(bool(torch.isfinite(x).all()) for x in losses.values()), 'Nonfinite original loss; no silent sem/geo omission')
    values = {k: float(v.detach()) for k, v in losses.items()}
    constant = [k for k in selected if k.endswith(('_inter_0', '_inter_1'))]
    active = [k for k in selected if k.endswith('_inter_2')]
    require(len(constant) == 4 and len(active) == 2, 'Decoder loss key contract changed')
    total = sum(selected.values())
    return total, dict(original_twelve=values, optimized_keys=list(selected),
        objective_sum=float(total.detach()), constant_intermediate_sum=sum(values[k] for k in constant),
        last_decoder_sum=sum(values[k] for k in active),
        omitted_semgeo_sum=sum(values[k] for k in losses if k not in selected),
        constant_intermediate_keys=constant, last_decoder_pools_fixed_t0_and_future=True)



def compare_constant_losses(audit_a, audit_z):
    """Audit only: preserve every original loss scalar/gradient unchanged.

    CUDA spatial NLL CE can differ for byte-identical inputs due to its
    atomic reductions. Only these two CE comparisons use the stated tolerance;
    the two constant Lovasz terms still require exact floating equality.
    """
    def ordered32(value):
        bits = struct.unpack('>I', struct.pack('>f', value))[0]
        return 0x80000000 - (bits & 0x7fffffff) if bits & 0x80000000 else 0x80000000 + bits
    keys = audit_a['constant_intermediate_keys']
    require(keys == audit_z['constant_intermediate_keys'], 'Constant key lists differ')
    require(set(keys) == {'loss_voxel_ce_inter_0', 'loss_voxel_ce_inter_1',
                         'loss_voxel_lovasz_inter_0', 'loss_voxel_lovasz_inter_1'}, 'Unexpected constant keys')
    result = {}
    for key in keys:
        left, right = audit_a['original_twelve'][key], audit_z['original_twelve'][key]
        ce = key.startswith('loss_voxel_ce_')
        passed = math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-7) if ce else left == right
        result[key] = dict(A=left, Z=right, Z_minus_A=right-left,
            float32_ULPs=abs(ordered32(left)-ordered32(right)),
            comparison='math.isclose' if ce else 'exact',
            rel_tol=1e-5 if ce else 0., abs_tol=1e-7 if ce else 0., passed=passed)
    return result


def field_stats(result, fused):
    import torch
    rows = []
    source_mass = float(result['source_probability_mass'].sum())
    source_count = float(result['source_foreground_voxels'].sum())
    for h in range(4):
        mass, weight = result['probability_mass'][:, h], result['support_weight'][:, h]
        covered = fused['destination_support'][:, h]
        gate = fused['gate'][:, h].detach()[covered]
        rows.append(dict(horizon_slot=h+1, nominal_seconds=(h+1)*.5,
            source_probability_mass=source_mass, source_foreground_voxels=int(source_count),
            destination_probability_mass=float(mass.sum()), destination_support_weight=float(weight.sum()),
            # Inferred conservation residual; FP32 summation can give tiny negative values.
            outside_probability_mass_residual=source_mass-float(mass.sum()),
            outside_support_weight_residual=source_count-float(weight.sum()),
            collision_probability_excess_above_one=float((mass-1).clamp_min(0).sum()),
            collision_support_excess_above_one=float((weight-1).clamp_min(0).sum()),
            covered_voxels=int(covered.sum()), uncovered_voxels=int((~covered).sum()),
            gate_on_covered=dict(count=gate.numel(), mean=None if not gate.numel() else float(gate.mean()),
                p10=None if not gate.numel() else float(torch.quantile(gate, .1)),
                median=None if not gate.numel() else float(torch.quantile(gate, .5)),
                p90=None if not gate.numel() else float(torch.quantile(gate, .9)),
                fraction_below_point1=None if not gate.numel() else float((gate < .1).float().mean()),
                fraction_above_point9=None if not gate.numel() else float((gate > .9).float().mean()))))
    return rows


def scores(rows, names):
    result = {}
    for name in names:
        matrix = np.asarray([r['hist_by_arm'][name] for r in rows], np.int64).sum(0)
        denominator = matrix[:, 1, :].sum(-1) + matrix[:, :, 1].sum(-1) - matrix[:, 1, 1]
        iou = matrix[:, 1, 1] / np.maximum(denominator, 1) * 100
        result[name] = dict(hist_by_horizon=matrix.tolist(), iou_by_horizon_percent=iou.tolist(),
            future_mean_iou_percent=float(iou[1:].mean()), FP_by_horizon=matrix[:, 0, 1].tolist(),
            FN_by_horizon=matrix[:, 1, 0].tolist())
    return result


def run(a, out, started):
    p, tp, helper, motion_path, motion_manifest = load_contract(a)
    native = import_source('native_state_cache', p['sources_sha256'])
    oracle = import_source('oracle_transport_probe', p['sources_sha256'])
    pretrained = import_source('learned_transport_probe_v1', p['sources_sha256'])
    adapter = import_source('objective_supervision_adapters', p['sources_sha256'])
    train_root, train, train_receipt = helper.cache_index(a.train_cache, 'train', tp)
    dev_root, dev, dev_receipt, reference = None, [], None, []
    if not a.preflight:
        require(a.dev_cache and a.o_development_records, 'Formal run requires dev cache and frozen O reference')
        dev_root, dev, dev_receipt = helper.cache_index(a.dev_cache, 'development', tp)
        require(sha(a.o_development_records) == p['o_development_records_sha256'], 'O development reference changed')
        reference = [json.loads(line) for line in Path(a.o_development_records).read_text().splitlines()]
        require(len(reference) == 200, 'O reference must contain exactly dev200')
        for r, old in zip(dev, reference):
            require(all(r[k] == old[k] for k in ('sample_token', 'scene_token')), 'O reference/cache order differs')
        require(not ({r['scene_token'] for r in train} & {r['scene_token'] for r in dev}), 'Scene overlap')
    require(sha(a.checkpoint) == oracle.M0_SHA and sha(a.o_checkpoint) == oracle.O_SHA and
            sha(a.config) == oracle.CONFIG_SHA, 'Original O/M0/config changed')
    import torch
    # Fusion's ordinary sibling import resolves to this already bound primitive.
    import_source('transport_ops', p['sources_sha256'])
    module = import_source('supported_motion_fusion', p['sources_sha256'])
    require(str(a.device).startswith('cuda'), 'Native frozen O requires CUDA')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False; torch.cuda.reset_peak_memory_stats(a.device)
    stop = []
    signal.signal(signal.SIGTERM, lambda s, f: stop.append(s)); signal.signal(signal.SIGINT, lambda s, f: stop.append(s))
    def budget():
        require(not stop, 'Signal received')
        require(time.monotonic()-started < a.max_seconds, 'Wall-clock ceiling reached')
        require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib*2**30, 'CUDA allocation ceiling exceeded')
    budget()
    model = native.build_native_model(a.config, a.checkpoint, device=a.device, repo=a.repo)
    payload = torch.load(a.o_checkpoint, map_location='cpu', weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True); del payload
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval()
    motion = pretrained.load_motion(motion_path, motion_manifest,
        dict(sources_sha256=p['sources_sha256'], training=p['motion']), tp, helper, a.device)
    require(sha(inspect.getsourcefile(model.compute_occ_loss)) == adapter.DETECTOR_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_occ)) == adapter.HEAD_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_voxel)) == adapter.HEAD_SOURCE_SHA,
            'Loss functions are not original source')
    require(model.future_pred_head.class_weights.tolist() == [1., 5.], 'Original class weights changed')
    require(not any(m.training for m in model.modules()) and not any(m.training for m in motion.modules()), 'Frozen models must stay eval')
    original_model_sha = helper.state_digest(model); motion_sha = helper.state_digest(motion)
    random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
    gates = {'A': module.SupportedMotionFusion().to(a.device).train()}
    gates['Z'] = copy.deepcopy(gates['A'])
    initial_state = {k: v.detach().cpu().clone() for k, v in gates['A'].state_dict().items()}
    initial_sha = helper.state_digest(gates['A'])
    require(initial_sha == helper.state_digest(gates['Z']), 'Initial gates differ')
    require(all(sum(v.numel() for v in gate.parameters()) == 97 for gate in gates.values()), 'Only 97 gate parameters may train')
    require(all(torch.count_nonzero(gate.gate[-1].weight).item() == 0 and
                torch.count_nonzero(gate.gate[-1].bias).item() == 0 for gate in gates.values()), 'Gate must initialize to half')
    optimizers = {arm: torch.optim.AdamW(gates[arm].parameters(), lr=1e-3, betas=(.9,.999), eps=1e-8,
                                       weight_decay=.01, amsgrad=False) for arm in ARMS}
    require(not ({id(x) for x in gates['A'].parameters()} & {id(x) for x in gates['Z'].parameters()}), 'Arms share parameter objects')
    orders = helper.planned_orders(); updates=examples=micro=evaluated=0
    arm_updates={arm:0 for arm in ARMS}
    limit = 4 if a.preflight else 512
    mode = 'preflight' if a.preflight else 'train'
    sources = dict(protocol_sha256=sha(a.protocol), sources_sha256=p['sources_sha256'],
        motion=p['motion'], O_checkpoint_sha256=oracle.O_SHA, M0_checkpoint_sha256=oracle.M0_SHA,
        config_sha256=oracle.CONFIG_SHA, train_cache=train_receipt, development_cache=dev_receipt,
        O_development_records_sha256=None if a.preflight else p['o_development_records_sha256'])
    manifest = dict(schema=SCHEMA, mode=mode, sources=sources, training=TRAINING,
        numerical_policy=p['numerical_policy'], initial_gate_sha256=initial_sha,
        frozen_O_state_sha256=original_model_sha, frozen_motion_state_sha256=motion_sha,
        sample_orders_sha256=helper.json_hash(orders), trainable_names=[n for n,_ in gates['A'].named_parameters()],
        arms=list(ARMS), same_initial_parameters=True, independent_optimizers=True, planned_updates=limit,
        input_source_mask='frozen coarse O current class1 > class0, exact argmax tie class0',
        source_probability='same O t0 softmax class1', boxes_or_motion_labels_read=False,
        final_decoder_futures_only=True, t0_and_uncovered_logits_preserved_bytewise=True,
        original_twelve_losses_computed=True, CE_Lovasz_selected_by_frozen_adapter=True,
        intermediate_two_layer_losses_constant=True, final_layer_t0_pooled_with_futures=True,
        multiple_comparison_adjustment=False, single_training_seed=11,
        GOSPA_or_motion_EPE_claim=False, historical_development_exposure=True)
    write(out/'manifest.json', manifest)
    arm_dirs = {}
    for arm in ARMS:
        arm_dirs[arm] = out/'runs'/arm; arm_dirs[arm].mkdir(parents=True)
        write(arm_dirs[arm]/'manifest.json', dict(manifest, arm=arm, transport='predicted' if arm=='A' else 'zero',
                                               common_manifest_sha256=sha(out/'manifest.json')))
        optimizers[arm].zero_grad(set_to_none=True)
    def checkpoint(arm, filename, status):
        path = arm_dirs[arm]/filename
        payload = dict(schema=SCHEMA, mode=mode, arm=arm, status=status,
            fusion=gates[arm].state_dict(), optimizer=optimizers[arm].state_dict(),
            update=arm_updates[arm], matched_updates_completed=updates,
            examples=examples, incomplete_accumulation_examples=micro,
            sample_orders=orders, sample_orders_sha256=helper.json_hash(orders),
            initial_gate_sha256=initial_sha, final_gate_sha256=helper.state_digest(gates[arm]),
            manifest_sha256=sha(arm_dirs[arm]/'manifest.json'), sources=sources, resume_supported=False)
        tmp=path.with_suffix('.tmp'); torch.save(payload,str(tmp)); tmp.replace(path)
        return dict(file=str(path.relative_to(out)), sha256=sha(path), gate_state_sha256=payload['final_gate_sha256'])
    def shared(record, root):
        with torch.no_grad():
            sample=native.load_sample(root,record,device=a.device)
            native.validate_inputs(sample['inputs'])
            state=sample['inputs']['prev_bev_input']
            require(state.dtype==torch.float32 and tuple(state.shape)==(1,1,40000,256) and
                    native._tensor_digest(state)==record['reference_bev_tensor_sha256'], 'Observed source changed')
            pred=native.replay(model,sample,training=False)[0]
            base=oracle.predictions_to_xyz(pred)
            flow=motion(state[:,-1])
            require(bool(torch.isfinite(flow).all()), 'Nonfinite frozen motion')
            probability=torch.softmax(base[0:1],dim=1)[:,1:2]
            foreground=(base[0:1,1:2] > base[0:1,0:1])
            fields={arm:module.supported_transport(probability,foreground,
                flow if arm=='A' else torch.zeros_like(flow),oracle.EXTENT) for arm in ARMS}
            require(all(not x.requires_grad for field in fields.values() for x in field.values()), 'Transport must be frozen')
            future=(base[1:,1]-base[1:,0])[None,:,None]
            require(native._tensor_digest(state)==record['reference_bev_tensor_sha256'], 'Measured BEV mutated')
            del flow, probability, foreground
        return sample,pred,base,future,fields
    had_gradient={arm:False for arm in ARMS}; batches={arm:[] for arm in ARMS}; dev_rows=[]
    try:
        for pass_index, order in enumerate(orders):
            for ordinal in order:
                budget()
                if micro==0:
                    torch.cuda.synchronize(a.device); tick=time.monotonic()
                    batches={arm:[] for arm in ARMS}
                    for opt in optimizers.values():
                        for group in opt.param_groups: group['lr']=helper.schedule_lr(updates)
                record=train[ordinal]
                sample,pred,base,future,fields=shared(record,train_root)
                audits={}
                for arm in ARMS:
                    # Graphs are fresh and independent; only shared detached fields recur.
                    fused=gates[arm](future,fields[arm]['probability_mass'],fields[arm]['support_weight'])
                    require(all(bool(torch.isfinite(v).all()) for v in fused.values()), 'Nonfinite gate')
                    if updates==0 and micro==0:
                        require(torch.all(fused['gate']==.5).item(), 'Initial gate is not exactly half')
                        disabled=gates[arm](future,fields[arm]['probability_mass'],fields[arm]['support_weight'],disabled=True)
                        require(exact32(disabled['logodds'],future) and exact32(compose_prediction(pred,base,disabled,oracle,disabled=True),pred), 'Disabled fusion not exact O')
                        del disabled
                    changed=compose_prediction(pred,base,fused,oracle)
                    loss,audit=loss_terms(model,changed,sample['targets'],adapter)
                    (loss/4).backward()
                    audits[arm]=audit
                    batches[arm].append(dict(ordinal=ordinal,sample_token=record['sample_token'],scene_token=record['scene_token'],
                        loss=audit, fields=field_stats(fields[arm],fused),
                        inputs_sha256=record['files']['inputs']['sha256'],targets_sha256=record['files']['targets']['sha256']))
                    del changed,loss,fused
                comparison=compare_constant_losses(audits['A'],audits['Z'])
                for arm in ARMS:
                    batches[arm][-1]['constant_loss_comparison']=comparison
                if not all(item['passed'] for item in comparison.values()):
                    write(out/'constant_loss_mismatch.json',dict(ordinal=ordinal,
                        sample_token=record['sample_token'],comparison=comparison))
                    raise ValueError('Fixed intermediate losses exceed their CE tolerance or Lovasz exact gate')
                require(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in motion.parameters()), 'Gradient leaked into frozen models')
                del sample,pred,base,future,fields,audits
                examples+=1;micro+=1
                if micro<4: continue
                norms={}
                for arm in ARMS:
                    require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in gates[arm].parameters()), 'Missing/nonfinite gate gradient')
                    norms[arm]=float(torch.nn.utils.clip_grad_norm_(gates[arm].parameters(),35.,error_if_nonfinite=True))
                    had_gradient[arm] |= norms[arm]>0
                budget()
                for arm in ARMS:
                    optimizers[arm].step();arm_updates[arm]+=1
                    optimizers[arm].zero_grad(set_to_none=True)
                    require(all(bool(torch.isfinite(v).all()) for v in gates[arm].parameters()), 'Nonfinite gate parameter')
                    require(all(bool(torch.isfinite(s[k]).all()) for s in optimizers[arm].state.values() for k in ('exp_avg','exp_avg_sq')), 'Nonfinite optimizer moment')
                updates+=1;micro=0;torch.cuda.synchronize(a.device)
                for arm in ARMS:
                    append(arm_dirs[arm]/'training.jsonl',dict(update=updates,pass_index=pass_index,examples=examples,
                        lr=optimizers[arm].param_groups[0]['lr'],loss=float(np.mean([x['loss']['objective_sum'] for x in batches[arm]])),
                        preclip_grad_norm=norms[arm],clip_factor=min(1.,35./(norms[arm]+1e-6)),
                        samples=batches[arm],sample_group_sha256=helper.json_hash([x['sample_token'] for x in batches[arm]]),
                        shared_update_wall_seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
                print(json.dumps(dict(event='MATCHED_UPDATE',update=updates,examples=examples,
                    losses={arm:float(np.mean([x['loss']['objective_sum'] for x in batches[arm]])) for arm in ARMS},
                    seconds=time.monotonic()-tick)),flush=True)
                budget()
                if updates==limit:break
            if updates==limit:break
        require(updates==limit and all(v==limit for v in arm_updates.values()) and examples==limit*4
                and micro==0 and all(had_gradient.values()), 'Fixed update budget or nonzero gradient gate failed')
        changes={arm:max(float((v.detach().cpu()-initial_state[k]).abs().max())
                         for k,v in gates[arm].state_dict().items()) for arm in ARMS}
        require(all(torch.count_nonzero(gate.gate[-1].weight).item()+torch.count_nonzero(gate.gate[-1].bias).item()>0
                    for gate in gates.values()), 'Zero readout unchanged; weight decay alone is not learning')
        finals={} if a.preflight else {arm:checkpoint(arm,'final.pth','FIXED_FINAL_512') for arm in ARMS}
        final_shas={arm:helper.state_digest(gates[arm]) for arm in ARMS}
        if not a.preflight:
            for gate in gates.values():gate.eval()
            with torch.no_grad():
                for i,record in enumerate(dev):
                    budget();tick=time.monotonic()
                    sample,pred,base,future,fields=shared(record,dev_root)
                    original=oracle.hist(model,pred,sample)
                    require(original==reference[i]['hist_by_horizon'], 'Frozen O dev200 confusion parity failed')
                    histograms={'O':original}; diagnostics={}
                    for arm in ARMS:
                        fused=gates[arm](future,fields[arm]['probability_mass'],fields[arm]['support_weight'])
                        changed=compose_prediction(pred,base,fused,oracle)
                        histograms[arm]=oracle.hist(model,changed,sample)
                        diagnostics[arm]=field_stats(fields[arm],fused)
                        del changed,fused
                    gt_counts=np.asarray(original,np.int64).sum(-1)
                    require(all(np.array_equal(np.asarray(h,np.int64).sum(-1),gt_counts) for h in histograms.values()), 'Evaluation GT support differs')
                    torch.cuda.synchronize(a.device);evaluated+=1
                    row=dict(ordinal=i,sample_token=record['sample_token'],scene_token=record['scene_token'],
                        official_index=record['official_index'],split='development',hist_by_arm=histograms,
                        diagnostics=diagnostics,inputs_sha256=record['files']['inputs']['sha256'],
                        targets_sha256=record['files']['targets']['sha256'],seconds=time.monotonic()-tick,
                        O_frozen_reference_hist_exact=True,t0_all_arms_byte_equal=True)
                    append(out/'development_records.jsonl',row);dev_rows.append(row)
                    for arm in ARMS:
                        append(arm_dirs[arm]/'development_records.jsonl',dict(sample_token=record['sample_token'],
                            scene_token=record['scene_token'],hist_by_horizon=histograms[arm],
                            horizon_seconds=[0.,.5,1.,1.5,2.],diagnostics=diagnostics[arm]))
                    del sample,pred,base,future,fields
                    budget()
            require(evaluated==200, 'Incomplete dev200')
        require(helper.state_digest(model)==original_model_sha and helper.state_digest(motion)==motion_sha,
                'Frozen O/motion weights or buffers changed')
        require(all(helper.state_digest(gates[arm])==final_shas[arm] for arm in ARMS), 'Evaluation changed gate state')
        summary=dict(schema=SCHEMA,mode=mode,updates=updates,examples=examples,evaluated_samples=evaluated,
            scores=None if a.preflight else scores(dev_rows,('O',*ARMS)),
            initial_gate_sha256=initial_sha,final_gate_sha256=final_shas,parameter_max_abs_change=changes,
            at_least_one_nonzero_task_gradient=had_gradient,zero_readout_changed=True,
            all_gradients_finite=True,O_and_motion_parameters_buffers_unchanged=True,
            weights_persisted=not a.preflight,single_training_seed=11,development_only=not a.preflight,
            GOSPA_or_motion_EPE_evaluated=False,motion_accuracy_or_generalization_claim=False,
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device))
        write(out/'summary.json',summary)
        for arm in ARMS:
            files=['manifest.json','training.jsonl']+([] if a.preflight else ['final.pth','development_records.jsonl'])
            write(arm_dirs[arm]/'complete.json',dict(schema=SCHEMA,arm=arm,mode=mode,updates=updates,examples=examples,
                status='PASS_SUPPORTED_FUSION_PREFLIGHT' if a.preflight else 'COMPLETE_SUPPORTED_FUSION_TRAINING',
                files_sha256={n:sha(arm_dirs[arm]/n) for n in files},final_gate_sha256=final_shas[arm]))
        files=['manifest.json','summary.json']+([] if a.preflight else ['development_records.jsonl'])
        budget()
        write(out/'complete.json',dict(schema=SCHEMA,mode=mode,updates=updates,examples=examples,evaluated_samples=evaluated,
            status='PASS_SUPPORTED_FUSION_PREFLIGHT' if a.preflight else 'COMPLETE_SUPPORTED_FUSION_TRAINING',
            files_sha256={n:sha(out/n) for n in files},arm_complete_sha256={arm:sha(arm_dirs[arm]/'complete.json') for arm in ARMS},
            final_checkpoints=finals,elapsed_seconds=time.monotonic()-started))
    except BaseException as exc:
        partial={}
        if not a.preflight:
            for arm in ARMS:
                try:partial[arm]=checkpoint(arm,'interrupted.pth','INCOMPLETE_NOT_RESUMABLE')
                except BaseException as e:partial[arm]=dict(error=repr(e))
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),updates=updates,
            actual_optimizer_updates_by_arm=arm_updates,
            examples=examples,incomplete_accumulation_examples=micro,evaluated_samples=evaluated,
            partial_checkpoints=partial,seconds=time.monotonic()-started))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','config','checkpoint','o-checkpoint','train-cache','motion-run',
                 'motion-training-protocol','repo','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--dev-cache');parser.add_argument('--o-development-records')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv)
    require(not (a.preflight and (a.dev_cache or a.o_development_records)), 'Preflight forbids development input')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:run(a,out,started)
    except BaseException as exc:
        if not (out/'failed.json').exists():
            write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),
                                        phase='initialization',seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
