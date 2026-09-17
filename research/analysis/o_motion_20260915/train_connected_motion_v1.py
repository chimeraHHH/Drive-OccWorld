"""Matched J/D joint motion-and-occupancy training, with O frozen.

J lets the unchanged occupancy objective differentiate through predicted
source displacement; D detaches displacement only at the transport input.
Both motion heads continue the same original rigid-box SmoothL1 supervision.
Only authenticated measured t0 BEV enters motion; boxes enter losses/evaluation.
No resume, best checkpoint, new seed, automatic retry, or dispatch is provided.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

SCHEMA = 'connected-motion-training-v1'
ARMS = ('J', 'D')
SOURCES = ('train_connected_motion_v1.py', 'train_supported_fusion_v2.py',
    'supported_motion_fusion.py', 'transport_ops.py', 'train_source_motion_v1.py',
    'motion_prediction_head.py', 'learned_transport_probe_v1.py', 'native_state_cache.py',
    'oracle_transport_probe.py', 'objective_supervision_adapters.py', 'joint_native_evaluation.py')
NUMERICAL = dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False)
TRAINING = dict(seed=11, train_samples=512, development_samples=200, passes=4,
    accumulate=4, updates=512, motion_parameters=467904, gate_parameters=97,
    motion_lr=1e-4, gate_lr=1e-3, motion_grad_clip=10., gate_grad_clip=35.,
    warmup_updates=25, final_lr_ratio=.1, weight_decay=.01,
    optimizer='AdamW', betas=[.9, .999], eps=1e-8, amsgrad=False,
    physical_loss_coefficient=1., occupancy_loss_coefficient=1., smooth_l1_beta_m=.5,
    lr_schedule='gate=source.schedule_lr(u); motion=gate/10; u=0..511',
    order='continuous_numpy_RandomState11_permutation_each_pass',
    physical_loss='original point_xyz_mean/object_mean/present_group_mean/valid_horizon_mean',
    occupancy_loss='original coarse twelve computed; select_objective O returns original CE+Lovasz six',
    scope='independent motion and gate; separate AdamW moments and clip groups; frozen O',
    transport='J: d; D: d.detach(); identical source O current argmax and probability',
    initialization='same authenticated pretrained motion; same new seed11 gate; no A/Z gate reuse',
    checkpoint='fixed final512; no resume, best or initial-development evaluation')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def source_path(name):
    require(Path(name).name == name, 'Sources require basenames')
    for p in (Path(__file__).parent / name,
              Path(__file__).parent.parent / 'm0_improvement_20260915' / name):
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def import_source(name, bindings):
    p = source_path(name + '.py')
    require(sha(p) == bindings[p.name], 'Source changed: ' + name)
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protocol_template():
    return dict(schema=SCHEMA, status='REVIEW_REQUIRED', training=copy.deepcopy(TRAINING),
        sources_sha256={n: None for n in SOURCES}, numerical_policy=dict(NUMERICAL),
        motion=dict(protocol_sha256=None, complete_sha256=None, checkpoint_sha256=None, head_state_sha256=None),
        cache_index_sha256=dict(train=None, development=None),
        labels=dict(manifest_sha256=None, complete_sha256=None),
        o_development_records_sha256=None,
        engineering=dict(independent_forward_rtol=1e-5, independent_forward_atol=1e-6,
            gradient_rtol=1e-5, gradient_atol=1e-7,
            exact_shared_value_VJP=True, first_fixed_train_anchor_only=True),
        resources=dict(preflight=dict(max_seconds=600, max_allocated_gib=32),
                       train=dict(max_seconds=7200, max_allocated_gib=32)))


def learning_rates(update, helper):
    gate = helper.schedule_lr(update)
    return dict(motion=gate / 10., gate=gate)


def transport_displacement(displacement, arm):
    require(arm in ARMS, 'Unknown connection arm')
    return displacement if arm == 'J' else displacement.detach()


def load_contract(a):
    p = read(a.protocol)
    require(p['schema'] == SCHEMA and p['status'] == 'FROZEN' and p['training'] == TRAINING,
            'Frozen fixed J/D protocol required')
    require(p['numerical_policy'] == NUMERICAL and p['engineering'] == protocol_template()['engineering'],
            'Numerical or engineering policy changed')
    require(set(SOURCES) <= set(p['sources_sha256']), 'Missing source bindings')
    for name, digest in p['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Source changed: ' + name)
    mode = 'preflight' if a.preflight else 'train'
    require(p['resources'][mode] == dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib)
            and a.max_seconds > 0 and a.max_allocated_gib > 0, 'Resource mismatch')
    helper = import_source('train_source_motion_v1', p['sources_sha256'])
    require(sha(a.motion_training_protocol) == p['motion']['protocol_sha256'], 'Motion protocol changed')
    tp = read(a.motion_training_protocol)
    require(tp['schema'] == helper.SCHEMA and tp['status'] == 'FROZEN' and tp['training'] == helper.TRAINING,
            'Wrong motion pretraining')
    require(tp['labels'] == p['labels'] and tp['cache_index_sha256'] == p['cache_index_sha256'],
            'Changed motion labels or measured states')
    for name, digest in tp['sources_sha256'].items():
        require(p['sources_sha256'][name] == digest, 'Motion pretraining source differs')
    root = Path(a.motion_run).resolve(); done = read(root / 'complete.json')
    require(sha(root / 'complete.json') == p['motion']['complete_sha256'], 'Motion completion changed')
    require(done['schema'] == helper.SCHEMA and done['status'] == 'COMPLETE_SOURCE_MOTION_TRAINING'
            and done['mode'] == 'train' and done['updates'] == 512 and done['examples'] == 2048
            and done['evaluated_samples'] == 200, 'Motion pretraining is not complete final512')
    expected = {'manifest.json', 'training.jsonl', 'summary.json', 'final.pth',
                'development_objects.jsonl', 'development_records.jsonl'}
    require(set(done['files_sha256']) == expected, 'Incomplete motion artifact chain')
    # Hash existing training-completion artifacts; preflight never parses old
    # development metrics or reads any development cache/label payload.
    for name, digest in done['files_sha256'].items():
        require(sha(root / name) == digest, 'Motion artifact changed: ' + name)
    require(done['final_checkpoint']['sha256'] == done['files_sha256']['final.pth'] == p['motion']['checkpoint_sha256']
            and done['final_checkpoint']['head_state_sha256'] == p['motion']['head_state_sha256'], 'Wrong pretrained final')
    manifest = read(root / 'manifest.json')
    require(manifest['source_receipt']['protocol_sha256'] == p['motion']['protocol_sha256']
            and manifest['source_receipt']['labels'] == p['labels'] and manifest['training'] == tp['training'],
            'Motion manifest mismatch')
    return p, tp, helper, root / 'final.pth', manifest


def tensor_comparison(a, b, *, rtol, atol):
    import torch
    require(a.shape == b.shape and a.dtype == b.dtype, 'Compared tensor shape/dtype mismatch')
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite compared tensor')
    delta = a.detach() - b.detach()
    return dict(max_abs=float(delta.abs().max()) if delta.numel() else 0.,
        changed_elements=int((a.detach() != b.detach()).sum()), elements=a.numel(),
        rtol=rtol, atol=atol, passed=bool(torch.allclose(a, b, rtol=rtol, atol=atol)))


def gradient_vector(grads, parameters):
    import torch
    require(len(grads) == len(parameters), 'Wrong VJP parameter list')
    require(all(g is None or bool(torch.isfinite(g).all()) for g in grads), 'Nonfinite VJP')
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for g, p in zip(grads, parameters)])


def engineering_probe(shared, label, models, gates, module, helper, fusion, oracle, adapter, model, numerical):
    """One fixed anchor, no .grad write or optimizer step; release all graphs.

    The exact-value proof uses a single computed transport value, J graph and
    its fully detached value for D. A separate independent D scatter is only
    the authorized allclose audit. Actual updates never reuse J's fields in D.
    """
    import torch
    sample, pred, base, future, tokens, probability, foreground = shared
    params = {arm: list(models[arm].parameters()) for arm in ARMS}
    gate_params = {arm: list(gates[arm].parameters()) for arm in ARMS}
    flows = {arm: models[arm](tokens) for arm in ARMS}
    flow_cmp = tensor_comparison(flows['J'], flows['D'],
        rtol=numerical['independent_forward_rtol'], atol=numerical['independent_forward_atol'])
    require(flow_cmp['passed'], 'Independent initial motion forwards disagree')
    fields = module.supported_transport(probability, foreground, flows['J'], oracle.EXTENT)
    exact_fields = {'J': fields, 'D': {k: v.detach() for k, v in fields.items()}}
    initial_logits, vectors, reports, loss_audits = {}, {}, {}, {}
    for arm in ARMS:
        field = exact_fields[arm]
        fused = gates[arm](future, field['probability_mass'], field['support_weight'])
        require(bool(torch.all(fused['gate'] == .5)), 'New gate is not half initialized')
        changed = fusion.compose_prediction(pred, base, fused, oracle)
        initial_logits[arm] = oracle.predictions_to_xyz(changed).detach()
        occ, loss_audits[arm] = fusion.loss_terms(model, changed, sample['targets'], adapter)
        physical, _ = helper.object_group_loss(helper.gather_sparse(flows[arm], label), label)
        all_params = params[arm] + gate_params[arm]
        gp = torch.autograd.grad(physical, all_params, allow_unused=True, retain_graph=True)
        go = torch.autograd.grad(occ, all_params, allow_unused=True, retain_graph=True)
        gt = torch.autograd.grad(physical + occ, all_params, allow_unused=True, retain_graph=False)
        n = len(params[arm]); vp = gradient_vector(gp[:n], params[arm])
        vo = gradient_vector(go[:n], params[arm]); vt = gradient_vector(gt[:n], params[arm])
        vg = gradient_vector(go[n:], gate_params[arm])
        require(all(g is None for g in gp[n:]), 'Physical loss reached gate parameters')
        require(all(g is None for g in go[:n]) if arm == 'D' else float(vo.norm()) > 0.,
                'D occupancy was not fully disconnected or J occupancy motion VJP is zero')
        require(float(vp.norm()) > 0., 'Fixed engineering anchor has no physical gradient; do not replace it')
        sum_cmp = tensor_comparison(vt, vp + vo, rtol=numerical['gradient_rtol'], atol=numerical['gradient_atol'])
        require(sum_cmp['passed'], 'Total VJP does not equal physical plus occupancy')
        vectors[arm] = (vp, vg)
        reports[arm] = dict(physical_loss=float(physical.detach()), occupancy_loss=float(occ.detach()),
            physical_motion_norm=float(vp.norm()), occupancy_motion_norm=float(vo.norm()),
            total_motion_norm=float(vt.norm()), occupancy_gate_norm=float(vg.norm()),
            physical_dot_occupancy=float(torch.dot(vp.double(), vo.double())),
            occupancy_motion_all_None=all(g is None for g in go[:n]), physical_gate_all_None=True,
            total_gradient_sum_comparison=sum_cmp, all_finite=True)
        del physical, occ, changed, fused, gp, go, gt, vp, vo, vt, vg
    require(fusion.exact32(initial_logits['J'], initial_logits['D']), 'Shared-value initial prediction is not byte equal')
    physical_cmp = tensor_comparison(vectors['J'][0], vectors['D'][0],
        rtol=numerical['gradient_rtol'], atol=numerical['gradient_atol'])
    gate_cmp = tensor_comparison(vectors['J'][1], vectors['D'][1],
        rtol=numerical['gradient_rtol'], atol=numerical['gradient_atol'])
    require(physical_cmp['passed'] and gate_cmp['passed'], 'Initial physical/gate VJP mismatch')
    with torch.no_grad():
        independent = module.supported_transport(probability, foreground, flows['D'].detach(), oracle.EXTENT)
        fused_d = gates['D'](future, independent['probability_mass'], independent['support_weight'])
        independent_logits = oracle.predictions_to_xyz(fusion.compose_prediction(pred, base, fused_d, oracle))
        output_cmp = tensor_comparison(initial_logits['J'][1:], independent_logits[1:],
            rtol=numerical['independent_forward_rtol'], atol=numerical['independent_forward_atol'])
        require(output_cmp['passed'], 'Independent initial scatter/forward exceeds authorized tolerance')
    constants = fusion.compare_constant_losses(loss_audits['J'], loss_audits['D'])
    require(all(r['passed'] for r in constants.values()), 'Initial fixed losses disagree')
    require(all(p.grad is None for seq in (*params.values(), *gate_params.values()) for p in seq), 'VJP contaminated .grad')
    return dict(status='PASS_INITIAL_CONNECTION_VJP', optimizer_updates=0,
        shared_transport_value_only_for_exact_VJP=True, actual_training_scatter_is_independent=True,
        shared_value_prediction_byte_equal=True, independent_flow_comparison=flow_cmp,
        independent_future_logit_comparison=output_cmp, physical_gradient_comparison=physical_cmp,
        occupancy_gate_gradient_comparison=gate_cmp, by_arm=reports,
        constant_loss_comparison_J_as_A_D_as_Z=constants, physical_GT_never_used_in_forward=True)


def seed_all(torch):
    random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)


def target_value_digest(target, native):
    """Loss targets are int64 expanded from authenticated uint8, losslessly hash values."""
    import torch
    require(target.dtype == torch.int64 and tuple(target.shape) == (1, 7, 512, 512, 40), 'Native target layout changed')
    require(bool(((target >= 0) & (target <= 255)).all()), 'Targets no longer represent original uint8 values')
    return native._tensor_digest(target.to(torch.uint8))


def run(a, out, started):
    p, tp, helper, motion_path, motion_manifest = load_contract(a)
    native = import_source('native_state_cache', p['sources_sha256'])
    oracle = import_source('oracle_transport_probe', p['sources_sha256'])
    pretrained = import_source('learned_transport_probe_v1', p['sources_sha256'])
    adapter = import_source('objective_supervision_adapters', p['sources_sha256'])
    fusion = import_source('train_supported_fusion_v2', p['sources_sha256'])
    tree = import_source('joint_native_evaluation', p['sources_sha256'])
    train_root, train, train_receipt = helper.cache_index(a.train_cache, 'train', tp)
    label_root, bindings = helper.labels_manifest(a.labels, tp)
    for i, record in enumerate(train):
        helper.check_identity(record, bindings[record['sample_token']])
        require(bindings[record['sample_token']]['ordinal'] == i, 'Training label order differs')
    dev_root, dev, dev_receipt, reference = None, [], None, []
    if not a.preflight:
        require(a.dev_cache and a.o_development_records, 'Formal training needs dev200 and frozen O reference')
        dev_root, dev, dev_receipt = helper.cache_index(a.dev_cache, 'development', tp)
        require(sha(a.o_development_records) == p['o_development_records_sha256'], 'O development reference changed')
        reference = [json.loads(line) for line in Path(a.o_development_records).read_text().splitlines()]
        require(len(reference) == 200, 'Frozen O reference must contain dev200')
        for i, (record, old) in enumerate(zip(dev, reference)):
            helper.check_identity(record, bindings[record['sample_token']])
            require(bindings[record['sample_token']]['ordinal'] == 512+i and
                    all(record[k] == old[k] for k in ('sample_token', 'scene_token')), 'Development identity/order differs')
        require(not ({r['scene_token'] for r in train} & {r['scene_token'] for r in dev}), 'Train/development scene overlap')
    require(sha(a.config) == oracle.CONFIG_SHA and sha(a.checkpoint) == oracle.M0_SHA and
            sha(a.o_checkpoint) == oracle.O_SHA, 'Frozen native O/M0/config changed')
    import torch
    import_source('transport_ops', p['sources_sha256'])
    module = import_source('supported_motion_fusion', p['sources_sha256'])
    require(str(a.device).startswith('cuda'), 'Native O training requires CUDA')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False; torch.cuda.reset_peak_memory_stats(a.device)
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
    fixed_motion = pretrained.load_motion(motion_path, motion_manifest,
        dict(sources_sha256=p['sources_sha256'], training=p['motion']), tp, helper, a.device)
    require(sha(inspect.getsourcefile(model.compute_occ_loss)) == adapter.DETECTOR_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_occ)) == adapter.HEAD_SOURCE_SHA and
            sha(inspect.getsourcefile(model.future_pred_head.loss_voxel)) == adapter.HEAD_SOURCE_SHA,
            'Loss source is not the original frozen implementation')
    require(model.future_pred_head.class_weights.tolist() == [1., 5.], 'Original class weights changed')
    original_model_sha = helper.state_digest(model); initial_motion_sha = helper.state_digest(fixed_motion)
    require(initial_motion_sha == p['motion']['head_state_sha256'], 'Pretrained actual motion state differs')
    models = {arm: copy.deepcopy(fixed_motion).train() for arm in ARMS}
    for head in models.values():
        for parameter in head.parameters(): parameter.requires_grad_(True)
    seed_all(torch)
    gates = {'J': module.SupportedMotionFusion().to(a.device).train()}
    gates['D'] = copy.deepcopy(gates['J'])
    initial_gate_sha = helper.state_digest(gates['J'])
    initial = dict(motion={n: v.detach().cpu().clone() for n, v in models['J'].named_parameters()},
                   gate={n: v.detach().cpu().clone() for n, v in gates['J'].named_parameters()})
    names = dict(motion=[n for n, _ in models['J'].named_parameters()], gate=[n for n, _ in gates['J'].named_parameters()])
    require(all(helper.state_digest(head) == initial_motion_sha for head in models.values()) and
            helper.state_digest(gates['D']) == initial_gate_sha, 'J/D do not have identical initial states')
    require(all(sum(v.numel() for v in h.parameters()) == 467904 for h in models.values()) and
            all(sum(v.numel() for v in g.parameters()) == 97 for g in gates.values()), 'Trainable scope changed')
    objects = [v for arm in ARMS for component in (models[arm], gates[arm]) for v in component.parameters()]
    require(len({id(v) for v in objects}) == len(objects) and all(v.requires_grad for v in objects), 'Shared/frozen trainable parameter objects')
    require(not any(m.training for m in model.modules()) and not any(m.training for m in fixed_motion.modules()) and
            not any(v.requires_grad for v in model.parameters()) and not any(v.requires_grad for v in fixed_motion.parameters()), 'Frozen O/P mode/scope changed')
    optimizers = {arm: {component: torch.optim.AdamW(network.parameters(), lr=TRAINING[component+'_lr'],
        betas=(.9, .999), eps=1e-8, weight_decay=.01, amsgrad=False)
        for component, network in (('motion', models[arm]), ('gate', gates[arm]))} for arm in ARMS}
    require(all(len(opt.state) == 0 for opts in optimizers.values() for opt in opts.values()), 'Optimizer state must start empty')
    orders = helper.planned_orders(); order_sha = helper.json_hash(orders)
    updates = examples = micro = evaluated = 0
    actual = {arm: dict(motion=0, gate=0) for arm in ARMS}
    limit = 4 if a.preflight else 512; mode = 'preflight' if a.preflight else 'train'
    sources = dict(protocol_sha256=sha(a.protocol), sources_sha256=p['sources_sha256'], motion=p['motion'],
        labels=p['labels'], O_checkpoint_sha256=oracle.O_SHA, M0_checkpoint_sha256=oracle.M0_SHA,
        config_sha256=oracle.CONFIG_SHA, train_cache=train_receipt, development_cache=dev_receipt,
        O_development_records_sha256=None if a.preflight else p['o_development_records_sha256'])
    manifest = dict(schema=SCHEMA, mode=mode, sources=sources, training=TRAINING, numerical_policy=NUMERICAL,
        initial_motion_sha256=initial_motion_sha, initial_gate_sha256=initial_gate_sha,
        frozen_O_state_sha256=original_model_sha, frozen_pretrained_motion_state_sha256=initial_motion_sha,
        sample_orders_sha256=order_sha, trainable_names=names, arms=list(ARMS), planned_updates=limit,
        same_initial_parameters=True, independent_optimizers=True, optimizer_components=['motion', 'gate'],
        motion_input='authenticated inputs.prev_bev_input[:, -1] only; no labels or future dt as input',
        source_foreground='O coarse current class1 > class0; tie predicts background',
        source_probability='same frozen O t0 softmax class1; no GT mask/support',
        physical_labels_used_only_by_loss_and_evaluation=True, O_native_flow_claim=False,
        original_twelve_losses_computed=True, CE_Lovasz_selected_by_frozen_adapter=True,
        final_decoder_futures_only=True, t0_and_uncovered_logits_preserved_bytewise=True,
        intermediate_two_layer_losses_constant=True, final_layer_t0_pooled_with_futures=True,
        exact_VJP_shared_value_vs_independent_training_scatter_distinguished=True,
        future_slot_times_nominal_actual_label_dt_reported=True, historical_development_exposure=True,
        single_training_seed=11, multiple_comparison_adjustment=False)
    write(out/'manifest.json', manifest)
    arm_dirs = {arm: out/'runs'/arm for arm in ARMS}
    for arm in ARMS:
        arm_dirs[arm].mkdir(parents=True)
        write(arm_dirs[arm]/'manifest.json', dict(manifest, arm=arm,
            transport_autograd='connected' if arm == 'J' else 'displacement_detached',
            common_manifest_sha256=sha(out/'manifest.json')))
    def checkpoint(arm, filename, status):
        path = arm_dirs[arm]/filename
        payload = dict(schema=SCHEMA, status=status, mode=mode, arm=arm,
            motion=models[arm].state_dict(), fusion=gates[arm].state_dict(),
            optimizers={k: opt.state_dict() for k, opt in optimizers[arm].items()},
            optimizer_updates=actual[arm], update=min(actual[arm].values()), matched_updates_completed=updates,
            examples=examples, incomplete_accumulation_examples=micro,
            sample_orders=orders, sample_orders_sha256=order_sha,
            initial_motion_sha256=initial_motion_sha, initial_gate_sha256=initial_gate_sha,
            final_motion_sha256=helper.state_digest(models[arm]), final_gate_sha256=helper.state_digest(gates[arm]),
            manifest_sha256=sha(arm_dirs[arm]/'manifest.json'), sources=sources, resume_supported=False)
        tmp = path.with_suffix('.tmp'); torch.save(payload, str(tmp)); tmp.replace(path)
        return dict(file=str(path.relative_to(out)), sha256=sha(path),
            motion_state_sha256=payload['final_motion_sha256'], gate_state_sha256=payload['final_gate_sha256'])
    def shared(record, root):
        # Only O is under no_grad. Motion and transport are built by each arm later.
        with torch.no_grad():
            sample = native.load_sample(root, record, device=a.device); native.validate_inputs(sample['inputs'])
            state = sample['inputs']['prev_bev_input']
            require(state.dtype == torch.float32 and tuple(state.shape) == (1, 1, 40000, 256) and
                    native._tensor_digest(state) == record['reference_bev_tensor_sha256'], 'Measured source BEV changed')
            input_sha = tree.tree_digest(sample['inputs']); target_sha = target_value_digest(sample['targets'], native)
            pred = native.replay(model, sample, training=False)[0]
            require(pred.dtype == torch.float32 and bool(torch.isfinite(pred).all()), 'Nonfinite/non-FP32 O predictions')
            base = oracle.predictions_to_xyz(pred)
            probability = torch.softmax(base[0:1], dim=1)[:, 1:2]
            foreground = base[0:1, 1:2] > base[0:1, 0:1]
            future = (base[1:, 1]-base[1:, 0])[None, :, None]
            require(tree.tree_digest(sample['inputs']) == input_sha and
                    target_value_digest(sample['targets'], native) == target_sha, 'O replay changed input or GT values')
        return (sample, pred, base, future, state[:, -1].detach(), probability, foreground), dict(
            input_tree_sha256=input_sha, target_values_sha256=target_sha)
    def unchanged(shared_value, binding):
        sample = shared_value[0]
        require(tree.tree_digest(sample['inputs']) == binding['input_tree_sha256'] and
                target_value_digest(sample['targets'], native) == binding['target_values_sha256'], 'Loss/motion modified shared inputs or GT')
        require(all(v.grad is None for v in model.parameters()) and all(v.grad is None for v in fixed_motion.parameters()), 'Gradient entered frozen O/P')
    def forward_arm(shared_value, label, arm):
        sample, pred, base, future, tokens, probability, foreground = shared_value
        displacement = models[arm](tokens)
        require(bool(torch.isfinite(displacement).all()), 'Nonfinite motion field')
        fields = module.supported_transport(probability, foreground, transport_displacement(displacement, arm), oracle.EXTENT)
        fused = gates[arm](future, fields['probability_mass'], fields['support_weight'])
        require(all(bool(torch.isfinite(v).all()) for v in fused.values()), 'Nonfinite gate output')
        changed = fusion.compose_prediction(pred, base, fused, oracle)
        occupancy, audit = fusion.loss_terms(model, changed, sample['targets'], adapter)
        physical, physical_audit = helper.object_group_loss(helper.gather_sparse(displacement, label), label)
        require(bool(torch.isfinite(physical)), 'Nonfinite physical objective')
        # Exact registered coefficients. Neither term is rescaled to match gradients.
        total = physical + occupancy
        diagnostics = fusion.field_stats({k: v.detach() for k, v in fields.items()}, {k: v.detach() for k, v in fused.items()})
        return total, dict(physical=float(physical.detach()), occupancy=float(occupancy.detach()),
            total=float(total.detach()), loss=audit, physical_audit=physical_audit, fields=diagnostics)
    had_gradient = {arm: dict(motion=False, gate=False) for arm in ARMS}
    dev_rows = []; epe_rows = {arm: [] for arm in ('P', *ARMS)}
    try:
        budget(); first_ordinal = orders[0][0]; first_record = train[first_ordinal]
        shared_value, boundary = shared(first_record, train_root)
        label = helper.load_sparse(label_root, bindings[first_record['sample_token']], first_record)
        engineering = engineering_probe(shared_value, label, models, gates, module, helper, fusion, oracle, adapter, model, p['engineering'])
        unchanged(shared_value, boundary)
        require(all(helper.state_digest(head) == initial_motion_sha for head in models.values()) and
                all(helper.state_digest(gate) == initial_gate_sha for gate in gates.values()), 'VJP changed initial parameters/buffers')
        engineering.update(ordinal=first_ordinal, sample_token=first_record['sample_token'], scene_token=first_record['scene_token'],
            label_sha256=bindings[first_record['sample_token']]['sha256'], boundary=boundary,
            source_inputs_unmodified=True, original_GT_values_unmodified=True, initial_parameters_buffers_unchanged=True)
        write(out/'engineering.json', engineering); del shared_value, boundary, label
        # Engineering consumes no optimizer step and cannot change the formal RNG boundary.
        seed_all(torch)
        for opts in optimizers.values():
            for opt in opts.values(): opt.zero_grad(set_to_none=True)
        budget()
        for pass_index, order in enumerate(orders):
            for ordinal in order:
                budget()
                if micro == 0:
                    torch.cuda.synchronize(a.device); tick = time.monotonic()
                    batches = {arm: [] for arm in ARMS}; lrs = learning_rates(updates, helper)
                    for opts in optimizers.values():
                        for component, opt in opts.items():
                            for group in opt.param_groups: group['lr'] = lrs[component]
                record = train[ordinal]; shared_value, boundary = shared(record, train_root)
                label = helper.load_sparse(label_root, bindings[record['sample_token']], record)
                audits = {}
                for arm in ARMS:
                    # Always independent forwards/scatters. Only d.detach differs in D.
                    total, audit = forward_arm(shared_value, label, arm)
                    (total / 4.).backward(); del total
                    audits[arm] = audit['loss']
                    audit.update(ordinal=ordinal, sample_token=record['sample_token'], scene_token=record['scene_token'],
                        inputs_sha256=record['files']['inputs']['sha256'], targets_sha256=record['files']['targets']['sha256'],
                        label_sha256=bindings[record['sample_token']]['sha256'], boundary=boundary)
                    batches[arm].append(audit)
                comparison = fusion.compare_constant_losses(audits['J'], audits['D'])
                for arm in ARMS: batches[arm][-1]['constant_loss_comparison_J_as_A_D_as_Z'] = comparison
                if not all(item['passed'] for item in comparison.values()):
                    write(out/'constant_loss_mismatch.json', dict(ordinal=ordinal, sample_token=record['sample_token'], comparison=comparison))
                    raise ValueError('Fixed CE tolerance or exact Lovasz gate failed')
                unchanged(shared_value, boundary); del shared_value, boundary, label, audits
                examples += 1; micro += 1
                if micro < 4: continue
                norms = {arm: {} for arm in ARMS}
                for arm in ARMS:
                    for component, network in (('motion', models[arm]), ('gate', gates[arm])):
                        require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in network.parameters()), 'Missing/nonfinite accumulated gradient')
                        clip = TRAINING[component+'_grad_clip']
                        norms[arm][component] = float(torch.nn.utils.clip_grad_norm_(network.parameters(), clip, error_if_nonfinite=True))
                        had_gradient[arm][component] |= norms[arm][component] > 0.
                budget()
                for arm in ARMS:
                    for component, opt in optimizers[arm].items():
                        opt.step(); actual[arm][component] += 1; opt.zero_grad(set_to_none=True)
                        require(all(bool(torch.isfinite(v).all()) for group in opt.param_groups for v in group['params']), 'Nonfinite parameter')
                        require(all(bool(torch.isfinite(s[k]).all()) for s in opt.state.values() for k in ('exp_avg', 'exp_avg_sq')), 'Nonfinite AdamW moment')
                updates += 1; micro = 0; torch.cuda.synchronize(a.device)
                for arm in ARMS:
                    append(arm_dirs[arm]/'training.jsonl', dict(update=updates, pass_index=pass_index, examples=examples,
                        lr=lrs, total_loss=float(np.mean([x['total'] for x in batches[arm]])),
                        physical_loss=float(np.mean([x['physical'] for x in batches[arm]])),
                        occupancy_loss=float(np.mean([x['occupancy'] for x in batches[arm]])),
                        preclip_grad_norm=norms[arm],
                        clip_factor={k: min(1., TRAINING[k+'_grad_clip']/(v+1e-6)) for k, v in norms[arm].items()},
                        samples=batches[arm], sample_group_sha256=helper.json_hash([x['sample_token'] for x in batches[arm]]),
                        shared_update_wall_seconds=time.monotonic()-tick,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)))
                print(json.dumps(dict(event='MATCHED_CONNECTED_UPDATE', update=updates, examples=examples,
                    loss={arm: float(np.mean([x['total'] for x in batches[arm]])) for arm in ARMS},
                    seconds=time.monotonic()-tick)), flush=True)
                budget()
                if updates == limit: break
            if updates == limit: break
        require(updates == limit and examples == limit*4 and micro == 0 and
                all(v == limit for row in actual.values() for v in row.values()) and
                all(v for row in had_gradient.values() for v in row.values()), 'Fixed matched budget/nonzero-gradient gate failed')
        changes = {arm: {component: max(float((v.detach().cpu()-initial[component][name]).abs().max())
            for name, v in network.named_parameters()) for component, network in (('motion', models[arm]), ('gate', gates[arm]))} for arm in ARMS}
        require(all(v > 0. for row in changes.values() for v in row.values()), 'A trainable component did not change')
        require(all(int(torch.count_nonzero(g.gate[-1].weight))+int(torch.count_nonzero(g.gate[-1].bias)) > 0
                    for g in gates.values()), 'Zero gate readout unchanged; decay alone is insufficient')
        final_motion = {arm: helper.state_digest(models[arm]) for arm in ARMS}
        final_gate = {arm: helper.state_digest(gates[arm]) for arm in ARMS}
        finals = {} if a.preflight else {arm: checkpoint(arm, 'final.pth', 'FIXED_FINAL_512') for arm in ARMS}
        if not a.preflight:
            for network in (*models.values(), *gates.values()): network.eval()
            with torch.no_grad():
                for i, record in enumerate(dev):
                    budget(); tick = time.monotonic(); shared_value, boundary = shared(record, dev_root)
                    sample, pred, base, future, tokens, probability, foreground = shared_value
                    label = helper.load_sparse(label_root, bindings[record['sample_token']], record)
                    histograms = {'O': oracle.hist(model, pred, sample)}; diagnostics = {}
                    require(histograms['O'] == reference[i]['hist_by_horizon'], 'Frozen O dev200 exact confusion parity failed')
                    support_keys = []
                    for arm, head in (('P', fixed_motion), *models.items()):
                        displacement = head(tokens)
                        require(bool(torch.isfinite(displacement).all()), 'Nonfinite evaluation displacement')
                        objects = helper.epe_records(helper.gather_sparse(displacement, label), label, record)
                        keys = [(r['instance_token'], r['horizon_seconds'], r['group'], r['source_points'], r['dt_seconds']) for r in objects]
                        if arm == 'P': support_keys = keys
                        else: require(keys == support_keys, 'Physical evaluation support/order differs')
                        for row in objects:
                            row['arm'] = arm; append(out/'development_objects.jsonl', row)
                        epe_rows[arm].extend(objects)
                        if arm != 'P':
                            fields = module.supported_transport(probability, foreground, displacement, oracle.EXTENT)
                            fused = gates[arm](future, fields['probability_mass'], fields['support_weight'])
                            changed = fusion.compose_prediction(pred, base, fused, oracle)
                            histograms[arm] = oracle.hist(model, changed, sample)
                            diagnostics[arm] = fusion.field_stats(fields, fused)
                            del fields, fused, changed
                        del displacement, objects
                    gt_counts = np.asarray(histograms['O'], np.int64).sum(-1)
                    require(all(np.array_equal(np.asarray(h, np.int64).sum(-1), gt_counts) and h[0] == histograms['O'][0]
                                for h in histograms.values()), 'GT support/t0 confusion changed')
                    unchanged(shared_value, boundary); torch.cuda.synchronize(a.device); evaluated += 1
                    row = dict(ordinal=i, sample_token=record['sample_token'], scene_token=record['scene_token'],
                        official_index=record['official_index'], split='development', hist_by_arm=histograms,
                        horizon_seconds=[0., .5, 1., 1.5, 2.], diagnostics=diagnostics,
                        inputs_sha256=record['files']['inputs']['sha256'], targets_sha256=record['files']['targets']['sha256'],
                        label_sha256=bindings[record['sample_token']]['sha256'], boundary=boundary,
                        O_frozen_reference_hist_exact=True, t0_all_arms_byte_equal=True, seconds=time.monotonic()-tick)
                    append(out/'development_records.jsonl', row); dev_rows.append(row)
                    for arm in ARMS:
                        append(arm_dirs[arm]/'development_records.jsonl', dict(ordinal=i, sample_token=record['sample_token'],
                            scene_token=record['scene_token'], hist_by_horizon=histograms[arm],
                            horizon_seconds=[0., .5, 1., 1.5, 2.], diagnostics=diagnostics[arm]))
                    del sample, pred, base, future, tokens, probability, foreground, label, shared_value, boundary
                    budget()
            require(evaluated == 200, 'Incomplete fixed final dev200')
        require(helper.state_digest(model) == original_model_sha and helper.state_digest(fixed_motion) == initial_motion_sha,
                'Frozen O/P parameters or buffers changed')
        require(all(helper.state_digest(models[arm]) == final_motion[arm] and helper.state_digest(gates[arm]) == final_gate[arm]
                    for arm in ARMS), 'Evaluation changed a final state')
        summary = dict(schema=SCHEMA, mode=mode, updates=updates, examples=examples, evaluated_samples=evaluated,
            scores=None if a.preflight else fusion.scores(dev_rows, ('O', *ARMS)),
            physical_epe=None if a.preflight else {arm: helper.epe_summary(rows) for arm, rows in epe_rows.items()},
            initial_motion_sha256=initial_motion_sha, initial_gate_sha256=initial_gate_sha,
            final_motion_sha256=final_motion, final_gate_sha256=final_gate, parameter_max_abs_change=changes,
            at_least_one_nonzero_gradient=had_gradient, zero_gate_readout_changed=True, all_gradients_finite=True,
            actual_optimizer_updates_by_arm=actual, initial_connection_VJP=engineering,
            O_and_pretrained_parameters_buffers_unchanged=True, weights_persisted=not a.preflight,
            physical_GT_is_rigid_box_proxy=True, O_native_flow_claim=False,
            single_training_seed=11, development_only=not a.preflight, historical_development_exposure=True,
            resources=helper.resources(a, started))
        write(out/'summary.json', summary)
        status = 'PASS_CONNECTED_MOTION_PREFLIGHT' if a.preflight else 'COMPLETE_CONNECTED_MOTION_TRAINING'
        for arm in ARMS:
            files = ['manifest.json', 'training.jsonl'] + ([] if a.preflight else ['final.pth', 'development_records.jsonl'])
            write(arm_dirs[arm]/'complete.json', dict(schema=SCHEMA, status=status, mode=mode, arm=arm,
                updates=updates, examples=examples, evaluated_samples=evaluated, optimizer_updates=actual[arm],
                files_sha256={n: sha(arm_dirs[arm]/n) for n in files},
                final_motion_sha256=final_motion[arm], final_gate_sha256=final_gate[arm]))
        files = ['manifest.json', 'engineering.json', 'summary.json'] + ([] if a.preflight else ['development_records.jsonl', 'development_objects.jsonl'])
        budget()
        write(out/'complete.json', dict(schema=SCHEMA, status=status, mode=mode, updates=updates, examples=examples,
            evaluated_samples=evaluated, files_sha256={n: sha(out/n) for n in files},
            arm_complete_sha256={arm: sha(arm_dirs[arm]/'complete.json') for arm in ARMS},
            final_checkpoints=finals, actual_optimizer_updates_by_arm=actual,
            elapsed_seconds=time.monotonic()-started))
    except BaseException as exc:
        partial = {}
        if not a.preflight:
            for arm in ARMS:
                try: partial[arm] = checkpoint(arm, 'interrupted.pth', 'INCOMPLETE_NOT_RESUMABLE')
                except BaseException as error: partial[arm] = dict(error=repr(error))
        write(out/'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY', error=repr(exc), updates=updates,
            actual_optimizer_updates_by_arm=actual, examples=examples, incomplete_accumulation_examples=micro,
            evaluated_samples=evaluated, partial_checkpoints=partial, seconds=time.monotonic()-started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'config', 'checkpoint', 'o-checkpoint', 'train-cache', 'labels', 'motion-run',
                 'motion-training-protocol', 'repo', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--dev-cache'); parser.add_argument('--o-development-records')
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--max-allocated-gib', type=float, required=True)
    a = parser.parse_args(argv)
    require(not (a.preflight and (a.dev_cache or a.o_development_records)), 'Preflight forbids development input')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    try: run(a, out, started)
    except BaseException as exc:
        if not (out/'failed.json').exists():
            write(out/'failed.json', dict(schema=SCHEMA, status='FAILED_NO_RETRY', error=repr(exc),
                                        phase='initialization', seconds=time.monotonic()-started))
        raise


def load_completed_arm(run_dir, protocol, helper, device='cuda:0'):
    """Strict fixed-final loader; returns (motion, gate, receipt), no optimizer restore.

    ``protocol`` is the actual frozen JSON Path, not a reconstructed JSON dict.
    The original byte hash is authenticated. Optimizer values are inspected
    field by field, so Python tuple betas versus JSON lists are not conflated.
    """
    import torch
    require(isinstance(protocol, (str, Path)), 'Pass the actual frozen protocol file path')
    protocol = Path(protocol); p = read(protocol)
    require(p['schema'] == SCHEMA and p['status'] == 'FROZEN' and p['training'] == TRAINING and
            p['numerical_policy'] == NUMERICAL and p['engineering'] == protocol_template()['engineering'], 'Wrong final training protocol')
    require(set(SOURCES) <= set(p['sources_sha256']), 'Final protocol missing sources')
    for name, digest in p['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Final source changed: ' + name)
    require(sha(helper.__file__) == p['sources_sha256']['train_source_motion_v1.py'], 'Wrong supplied helper')
    run_dir = Path(run_dir).resolve(); arm = run_dir.name; top = run_dir.parent.parent
    require(arm in ARMS and run_dir.parent.name == 'runs', 'Expected completed root/runs/J or D')
    require(not (top/'failed.json').exists() and not (run_dir/'failed.json').exists(), 'Failed training is not loadable')
    complete = read(top/'complete.json'); common = read(top/'manifest.json')
    expected = {'manifest.json', 'engineering.json', 'summary.json', 'development_records.jsonl', 'development_objects.jsonl'}
    require(complete['schema'] == SCHEMA and complete['status'] == 'COMPLETE_CONNECTED_MOTION_TRAINING' and
            complete['mode'] == 'train' and complete['updates'] == 512 and complete['examples'] == 2048 and
            complete['evaluated_samples'] == 200 and set(complete['files_sha256']) == expected,
            'Top run is not complete fixed512/dev200')
    for name, digest in complete['files_sha256'].items(): require(sha(top/name) == digest, 'Top artifact changed: ' + name)
    require(set(complete['arm_complete_sha256']) == set(complete['final_checkpoints']) == set(ARMS), 'Missing matched final arm')
    require(complete['actual_optimizer_updates_by_arm'] == {a: dict(motion=512, gate=512) for a in ARMS}, 'Unequal component budgets')
    require(common['schema'] == SCHEMA and common['mode'] == 'train' and common['training'] == TRAINING and
            common['numerical_policy'] == NUMERICAL and common['arms'] == list(ARMS) and common['planned_updates'] == 512,
            'Common manifest recipe changed')
    sources = common['sources']
    require(sources['protocol_sha256'] == sha(protocol) and sources['sources_sha256'] == p['sources_sha256'] and
            sources['motion'] == p['motion'] and sources['labels'] == p['labels'] and
            sources['train_cache']['index_sha256'] == p['cache_index_sha256']['train'] and
            sources['development_cache']['index_sha256'] == p['cache_index_sha256']['development'] and
            sources['O_development_records_sha256'] == p['o_development_records_sha256'], 'Final source/data identity changed')
    oracle = import_source('oracle_transport_probe', p['sources_sha256'])
    require(sources['O_checkpoint_sha256'] == oracle.O_SHA and sources['M0_checkpoint_sha256'] == oracle.M0_SHA and
            sources['config_sha256'] == oracle.CONFIG_SHA, 'Native prediction contract changed')
    require(common['initial_motion_sha256'] == common['frozen_pretrained_motion_state_sha256'] == p['motion']['head_state_sha256'], 'Wrong motion initialization')
    orders = helper.planned_orders(); order_sha = helper.json_hash(orders)
    require(common['sample_orders_sha256'] == order_sha, 'Wrong planned training order')
    engineering = read(top/'engineering.json')
    require(engineering['status'] == 'PASS_INITIAL_CONNECTION_VJP' and engineering['optimizer_updates'] == 0 and
            engineering['ordinal'] == orders[0][0] and engineering['shared_value_prediction_byte_equal'] and
            engineering['by_arm']['D']['occupancy_motion_all_None'] and
            engineering['by_arm']['J']['occupancy_motion_norm'] > 0 and
            engineering['source_inputs_unmodified'] and engineering['original_GT_values_unmodified'], 'Initial connection proof missing')
    summary = read(top/'summary.json')
    require(summary['schema'] == SCHEMA and summary['mode'] == 'train' and summary['updates'] == 512 and
            summary['examples'] == 2048 and summary['evaluated_samples'] == 200 and
            summary['O_and_pretrained_parameters_buffers_unchanged'] and summary['all_gradients_finite'], 'Incomplete training summary')
    # Authenticate both arms' small receipt files, including the paired budget,
    # before loading the requested large tensor payload.
    manifests = {}
    for name in ARMS:
        directory = top/'runs'/name; done_path = directory/'complete.json'
        require(sha(done_path) == complete['arm_complete_sha256'][name], 'Arm completion changed')
        done = read(done_path); manifest = read(directory/'manifest.json')
        require(done['schema'] == SCHEMA and done['mode'] == 'train' and done['arm'] == name and
                done['status'] == complete['status'] and done['updates'] == 512 and done['examples'] == 2048 and
                done['evaluated_samples'] == 200 and done['optimizer_updates'] == dict(motion=512, gate=512), 'Arm final budget mismatch')
        require(set(done['files_sha256']) == {'manifest.json', 'training.jsonl', 'final.pth', 'development_records.jsonl'}, 'Arm file chain differs')
        for filename, digest in done['files_sha256'].items(): require(sha(directory/filename) == digest, 'Arm file changed: ' + filename)
        expected_manifest = dict(common, arm=name, transport_autograd='connected' if name == 'J' else 'displacement_detached',
                                 common_manifest_sha256=sha(top/'manifest.json'))
        require(manifest == expected_manifest, 'Per-arm manifest differs from common recipe')
        final = complete['final_checkpoints'][name]
        require(final == dict(file='runs/'+name+'/final.pth', sha256=done['files_sha256']['final.pth'],
                motion_state_sha256=done['final_motion_sha256'], gate_state_sha256=done['final_gate_sha256']), 'Final pointer differs')
        require(summary['final_motion_sha256'][name] == final['motion_state_sha256'] and
                summary['final_gate_sha256'][name] == final['gate_state_sha256'], 'Summary/final tensor identities differ')
        logs = [json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
        require(len(logs) == 512, 'Incomplete update log')
        for u, row in enumerate(logs):
            expected_order = orders[u//128][(u%128)*4:(u%128+1)*4]
            require(row['update'] == u+1 and row['examples'] == (u+1)*4 and row['pass_index'] == u//128 and
                    row['lr'] == learning_rates(u, helper) and [x['ordinal'] for x in row['samples']] == expected_order and
                    row['sample_group_sha256'] == helper.json_hash([x['sample_token'] for x in row['samples']]), 'Update schedule/order mismatch')
        manifests[name] = manifest
    payload = torch.load(str(run_dir/'final.pth'), map_location='cpu', weights_only=False)
    require(payload['schema'] == SCHEMA and payload['status'] == 'FIXED_FINAL_512' and payload['mode'] == 'train' and
            payload['arm'] == arm and payload['update'] == payload['matched_updates_completed'] == 512 and
            payload['optimizer_updates'] == dict(motion=512, gate=512) and payload['examples'] == 2048 and
            payload['incomplete_accumulation_examples'] == 0 and not payload['resume_supported'], 'Wrong final payload endpoint')
    require(payload['sources'] == sources and payload['manifest_sha256'] == sha(run_dir/'manifest.json') and
            payload['sample_orders'] == orders and payload['sample_orders_sha256'] == order_sha and
            payload['initial_motion_sha256'] == common['initial_motion_sha256'] and
            payload['initial_gate_sha256'] == common['initial_gate_sha256'], 'Final payload provenance differs')
    head_module = import_source('motion_prediction_head', p['sources_sha256'])
    import_source('transport_ops', p['sources_sha256'])
    gate_module = import_source('supported_motion_fusion', p['sources_sha256'])
    motion = head_module.MotionPredictionHead(helper.SHAPE); gate = gate_module.SupportedMotionFusion()
    motion.load_state_dict(payload['motion'], strict=True); gate.load_state_dict(payload['fusion'], strict=True)
    actual_motion = helper.state_digest(motion); actual_gate = helper.state_digest(gate)
    final = complete['final_checkpoints'][arm]
    require(actual_motion == payload['final_motion_sha256'] == final['motion_state_sha256'] and
            actual_gate == payload['final_gate_sha256'] == final['gate_state_sha256'], 'Actual final tensor SHA differs')
    require(set(payload['optimizers']) == {'motion', 'gate'}, 'Missing optimizer state')
    for component, network in (('motion', motion), ('gate', gate)):
        require(sum(v.numel() for v in network.parameters()) == TRAINING[component+'_parameters'] and
                [n for n, _ in network.named_parameters()] == common['trainable_names'][component], 'Final architecture/scope changed')
        optimizer = payload['optimizers'][component]; groups = optimizer['param_groups']
        require(len(groups) == 1, 'Unexpected optimizer group count')
        group = groups[0]; states = optimizer['state']
        require(group['lr'] == learning_rates(511, helper)[component] and tuple(group['betas']) == (.9, .999) and
                group['eps'] == 1e-8 and group['weight_decay'] == .01 and group['amsgrad'] is False and
                group.get('maximize', False) is False and group.get('capturable', False) is False and
                group.get('differentiable', False) is False, 'Final AdamW policy changed')
        parameters = list(network.parameters())
        require(group['params'] == list(range(len(parameters))) and set(states) == set(group['params']), 'Delayed/missing parameter optimizer state')
        for index, parameter in enumerate(parameters):
            state = states[index]
            require(float(state['step']) == 512., 'Parameter AdamW step differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                require(state[key].shape == parameter.shape and state[key].dtype == parameter.dtype and
                        bool(torch.isfinite(state[key]).all()), 'Invalid final AdamW moment')
    del payload
    for network in (motion, gate):
        for parameter in network.parameters(): parameter.requires_grad_(False)
        network.to(device).eval()
    receipt = dict(schema=SCHEMA, arm=arm, checkpoint_sha256=final['sha256'],
        actual_motion_state_sha256=actual_motion, actual_gate_state_sha256=actual_gate,
        manifest_sha256=sha(run_dir/'manifest.json'), common_manifest_sha256=sha(top/'manifest.json'),
        top_complete_sha256=sha(top/'complete.json'), arm_complete_sha256=complete['arm_complete_sha256'][arm],
        protocol_sha256=sha(protocol), fixed_final_update=512, fixed_examples=2048, fixed_development_samples=200,
        actual_optimizer_parameter_steps_all512=True, optimizer_restored=False, optimizer_updates_in_this_evaluation=0)
    return motion, gate, receipt


if __name__ == '__main__':
    main()
