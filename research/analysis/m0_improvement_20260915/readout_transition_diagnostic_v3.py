"""Read-only post-hoc C/O feature/readout 2x2 diagnostic on fixed dev200.

No training, loss adapter, parameter swapping, threshold fitting or candidate
selection. Pure Module prehooks copy the three terminal readout inputs. The
unchanged forward_head applies each complete readout to each feature source.
Pilot is exactly two identity-selected anchors and reports engineering only.
Development needs a hash-bound PASS pilot and a separate frozen authorization.
Only small confusion/hash receipts are persisted, never features or logits.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import signal
import textwrap
import time

NAMES = ('CC', 'CO', 'OC', 'OO')  # feature source, complete readout source
HORIZONS = [0., .5, 1., 1.5, 2.]
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
OBJECTIVE_SHA = '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d'
AGGREGATE_SHA = '9e2327d51dd34e3f207d216b0966177531ffdc4e6ecdddddf0f06004ace5e256'
TREE_SHA = '280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578'
PILOT_RULE = 'first_anchor_of_first_two_distinct_scenes_in_frozen_development_order'
EFFECTS = {
    'total_O_minus_C': [-1, 0, 0, 1],
    'readout_on_C_features': [-1, 1, 0, 0],
    'features_under_O_readout': [0, -1, 0, 1],
    'features_under_C_readout': [-1, 0, 1, 0],
    'readout_on_O_features': [0, 0, -1, 1],
    'interaction_on_metric_scale': [1, -1, -1, 1],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush()
    tmp.replace(path)


def module(filename, digest):
    path = Path(__file__).resolve().with_name(filename)
    require(Path(filename).name == filename and sha(path) == digest, 'Frozen source changed: '+filename)
    spec = importlib.util.spec_from_file_location('_readout_diagnostic_'+path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def first_two_scenes(records):
    selected = []; scenes = set()
    for ordinal, record in enumerate(records):
        if record['scene_token'] not in scenes:
            selected.append(ordinal); scenes.add(record['scene_token'])
        if len(selected) == 2:
            break
    require(len(selected) == 2, 'Pilot needs two distinct scenes')
    return selected


def identity(record):
    return {k: record[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')} | {
        'targets_sha256': record['files']['targets']['sha256'],
        'target_counts_0_1_255_by_frame': record['target_counts_0_1_255_by_frame']}


def verify_files(root, files):
    require(files and all(Path(k).name == k for k in files), 'Unsafe or empty file ledger')
    for name, digest in files.items():
        require(sha(Path(root)/name) == digest, 'Bound file differs: '+str(Path(root)/name))


def load_policy(a):
    policy = read(a.diagnostic_protocol)
    require(policy['schema'] == 'readout-transition-diagnostic-protocol-v1' and policy['status'] == 'FROZEN',
            'Require a frozen diagnostic protocol')
    require(sha(a.protocol) == policy['parent_protocol_sha256'] == PARENT_SHA and
            sha(a.objective_protocol) == policy['objective_protocol_sha256'] == OBJECTIVE_SHA,
            'Wrong parent/objective protocols')
    require(policy['combinations'] == list(NAMES) and policy['pilot_selection'] == PILOT_RULE,
            'Only the fixed C/O 2x2 and identity-only pilot are accepted')
    require(policy['posthoc'] is True and policy['training'] is False and policy['model_selection'] is False and
            policy['historical_validation_exposure'] is True, 'Post-hoc scope disclosure differs')
    sources = policy['source_sha256']
    require(sources.get(Path(__file__).name) == sha(__file__) and
            sources.get('objective_supervision_aggregate.py') == AGGREGATE_SHA and
            sources.get('joint_native_evaluation.py') == TREE_SHA, 'Diagnostic helper source differs')
    verify_files(Path(__file__).parent, sources)
    require(sha(a.config) == policy['config_sha256'] and sha(a.checkpoint) == policy['m0_sha256'],
            'Native config/M0 differs')
    dev = Path(a.development_summary); done = read(dev.with_name('complete.json'))
    require(sha(dev) == policy['development_summary']['summary_sha256'] == done['summary_sha256'] and
            sha(dev.with_name('complete.json')) == policy['development_summary']['complete_sha256'],
            'Development summary receipt differs')
    require(done['status'] == 'COMPLETE' and done['schema'] == 'm0-objective-supervision-aggregate-complete-v1',
            'Development five-model comparison is incomplete')
    verify_files(dev.parent, done['files_sha256'])
    development = read(dev)
    require(development['schema'] == 'm0-objective-supervision-aggregate-v1' and
            development['objective_protocol_sha256'] == OBJECTIVE_SHA and development['samples'] == 200 and
            development['scenes'] == 100 and set(development['models']) == {'M0','M0_fp32','native1','F','O'},
            'Need the complete, unchanged five-model development disclosure')
    require(set(policy['model_sources']) == {'C','O'}, 'Only C and O final sources are accepted')
    for name, oldname in [('C','native1'), ('O','O')]:
        old = development['source_receipts'][oldname]
        files = old['reuse_audit']['files_sha256'] if name == 'C' else old['files_sha256']
        require(policy['model_sources'][name] == dict(files_sha256=files,
                    final_head_state_sha256=old['checkpoint']['final_head_state_sha256']),
                'Final model source differs from certified development: '+name)
    resource = policy['resources']
    require(0 < resource['total_max_seconds'] <= 1800 and 0 < resource['max_allocated_gib'] <= 32,
            'Diagnostic exceeds authorized outer ceiling')
    for key in ('pilot', 'development_ceiling'):
        cap = resource[key]
        require(0 < cap['max_seconds'] <= resource['total_max_seconds'] and
                0 < cap['max_allocated_gib'] <= resource['max_allocated_gib'], 'Invalid phase ceiling')
    return policy, development


def phase_gate(a, policy):
    actual = dict(max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib)
    if a.mode == 'pilot':
        require(a.pilot is None and a.authorization is None, 'Pilot cannot consume its own future gate')
        require(actual == policy['resources']['pilot'], 'Pilot CLI resources must equal protocol')
        return None
    require(a.pilot and a.authorization, 'Development needs a completed pilot and frozen authorization')
    root = Path(a.pilot); done = read(root/'complete.json')
    require(done['schema'] == 'readout-transition-complete-v1' and
            done['status'] == 'PASS_READOUT_TRANSITION_ENGINEERING' and
            done['diagnostic_protocol_sha256'] == sha(a.diagnostic_protocol), 'Pilot gate is not a bound PASS')
    require(set(done['files_sha256']) == {'manifest.json','records.jsonl','summary.json','report.md'},
            'Pilot output file coverage differs')
    verify_files(root, done['files_sha256'])
    summary = read(root/'summary.json')
    require(summary['mode'] == 'pilot' and summary['samples'] == 2 and summary['scenes'] == 2 and
            summary['all_engineering_gates_passed'] is True and summary['optimizer_updates'] == 0 and
            summary['diagnostic_protocol_sha256'] == sha(a.diagnostic_protocol), 'Pilot summary differs')
    auth = read(a.authorization)
    require(auth['schema'] == 'readout-transition-development-authorization-v1' and auth['status'] == 'FROZEN' and
            auth['diagnostic_protocol_sha256'] == sha(a.diagnostic_protocol) and
            auth['pilot_complete_sha256'] == sha(root/'complete.json') and
            auth['pilot_summary_sha256'] == sha(root/'summary.json'), 'Authorization evidence differs')
    require(auth['resources'] == actual, 'Development CLI must equal actual frozen authorization')
    ceiling = policy['resources']['development_ceiling']
    require(0 < a.max_seconds <= ceiling['max_seconds'] and
            0 < a.max_allocated_gib <= ceiling['max_allocated_gib'] and
            a.max_seconds + done['seconds'] <= policy['resources']['total_max_seconds'],
            'Pilot plus development exceeds the immutable ceiling')
    return dict(authorization_sha256=sha(a.authorization), pilot_complete_sha256=sha(root/'complete.json'),
                pilot_summary_sha256=sha(root/'summary.json'), pilot_seconds=done['seconds'])


def load_evidence(a, policy, development):
    aggregate = module('objective_supervision_aggregate.py', AGGREGATE_SHA)
    parent, objective, trainer, helper, metric, audit, engineering = aggregate.load_contracts(a)
    for name, digest in {**parent['source_sha256'], **objective['source_sha256']}.items():
        require(policy['source_sha256'].get(name) == digest, 'Missing bound dependency: '+name)
    require(policy['config_sha256'] == parent['config_sha256'] and policy['m0_sha256'] == parent['m0_sha256'],
            'Native initialization binding differs')
    cache, index, records, m0_hist, selection = metric.load_cache(a.cache, parent, a.protocol)
    require(sha(cache/'index.json') == policy['cache_index_sha256'] == objective['cache_index_sha256']['development'] and
            index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True,
            'Wrong complete separated development cache')
    aggregate.check_selection(records, selection)
    require([identity(r) for r in records] == development['ordered_identities_and_GT'], 'Complete GT/order differs')
    a.control_run = str(Path(a.native_runs_root)/'native1')
    control, c_payload, reuse = helper.validate_control(a, parent, objective, {'development': records})
    del c_payload
    chist, crec = metric.load_arm(a.native_runs_root, 'native1', records, m0_hist, parent, a.protocol, sha(cache/'index.json'))
    crec['reuse_audit'] = reuse
    crec['training'] = audit.check_log(Path(a.control_run), helper, parent['training'])
    crec['checkpoint'] = audit.check_payload(Path(a.control_run), control, objective['control_files_sha256']['latest.pth'],
                                           'native1', helper.planned_orders(parent['training']))
    _, block = helper.native_training_block(parent)
    block.pop('full_task_losses_unchanged')
    block.update(native_sum_accumulation_backward_unchanged=True, frozen_control_helper_sha256=trainer.HELPER_SHA)
    ohist, orec = aggregate.load_candidate(a, 'O', parent, objective, helper, metric, audit, records, m0_hist, control, block)
    omanifest = read(Path(a.runs_root)/'O/manifest.json')
    require(omanifest['engineering_evidence'] == engineering, 'O engineering evidence differs')
    expected_adapter = next(r['adapter'] for r in read(Path(engineering['objective_trainmode']['directory'])/'summary.json')['reports'] if r['arm'] == 'O')
    require(orec['objective_adapter'] == expected_adapter, 'O adapter receipt differs from its training proof')
    for name, receipt in [('native1',crec), ('O',orec)]:
        require(json.loads(json.dumps(receipt, allow_nan=False)) == development['source_receipts'][name], 'Actual final source audit differs: '+name)
    for key in ('tensor_shapes','optimizer_groups','sample_orders_sha256'):
        require(crec['checkpoint'][key] == orec['checkpoint'][key], 'C/O common budget/state shape differs')
    return aggregate, parent, objective, metric, cache, records, {'C':chist, 'O':ohist}, {'C':crec,'O':orec}


def runtime_guard(model):
    import torch
    head = model.future_pred_head
    require(type(model).__name__ == 'Drive_OccWorld' and type(head).__name__ == 'WorldHeadV1', 'Wrong native model/head class')
    for name in ('turn_on_plan','turn_on_flow','predict_flow','motion_residual','doppler_posterior','doppler_advection','doppler_flow_loss'):
        require(not getattr(model, name, None), 'Unexpected enabled branch: '+name)
    require(not head.soft_weight and not head.turn_on_flow and not head.sem_norm and not head.obj_motion_norm and
            head.memory_queue_len == 1 and model.memory_queue_len == 1 and
            head.num_classes == 2 and head.num_pred_height == 16 and head.pred_frame_num == 1 and
            len(head.bev_pred_head) == 3, 'Only original native1 three-readout graph is accepted')
    neck = head.prev_render_neck
    require(neck is None or (not getattr(neck,'sem_norm',False) and not getattr(neck,'sem_gt_train',False)),
            'Semantic/GT conditioning is outside the native contract')
    require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
            'Read-only inference requires all eval/frozen')
    require(all(isinstance(branch, torch.nn.Sequential) and len(branch) == 4 and
                not branch._forward_pre_hooks and not branch._forward_hooks for branch in head.bev_pred_head),
            'Unexpected complete readout structure')
    require(not any(p.grad is not None for p in model.parameters()), 'Unexpected model gradients')
    require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is True and
            torch.backends.cudnn.benchmark is False, 'Frozen native1 FP32 numerical policy differs')
    return dict(native1=True, planning=False, flow=False, M3=False, soft_weight=False, sem_norm=False,
                all_modules_eval=True, all_parameters_frozen=True, readout_branches=3,
                matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False)


def code_receipt(model, objective):
    # native1 uses the already-frozen AST-derived no-GT guard. Its synthetic
    # filename is not a source file. Audit the original class method and retain
    # the exact migration/source-AST receipt checked against completed C/O.
    result = {'native1_derived_future_pred': model.observation_memory_receipt}
    runtime = {str(Path(p).resolve()): digest for p,digest in objective['runtime_source_sha256'].items()}
    require(model.future_pred.__func__.__code__.co_filename == '<M0-observation-memory:native1>',
            'Unexpected runtime future_pred implementation')
    for name, method in [('original_future_pred',type(model).future_pred), ('evaluate_occ_records',model.evaluate_occ_records),
                         ('forward_head',model.future_pred_head.forward_head),
                         ('forward_head_layers',model.future_pred_head.forward_head_layers),
                         ('future_head_forward',model.future_pred_head.forward)]:
        function = inspect.unwrap(getattr(method,'__func__',method)); path = Path(inspect.getsourcefile(function)).resolve()
        require(str(path) in runtime and sha(path) == runtime[str(path)], 'Runtime method source not frozen: '+name)
        result[name] = dict(file=str(path), sha256=sha(path), qualname=function.__qualname__,
            method_AST_sha256=hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))),
                                                   include_attributes=False).encode()).hexdigest())
    return result


def make_models(a, parent, objective, policy, cache_module, memory):
    import torch
    cache_module._prepare_repo(a.repo)
    result = {}; receipts = {}
    for name, arm, directory in [('C','native1',Path(a.native_runs_root)/'native1'), ('O','O',Path(a.runs_root)/'O')]:
        memory.seed_all(11)
        model, migration, params = memory.prepare_model(a.config, a.checkpoint, mode='native1')
        require(memory.state_digest(model.future_pred_head) == '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60',
                'Wrong original M0 future-head initialization')
        manifest = read(directory/'manifest.json')
        require(migration == manifest['migration'] and list(params) == manifest['trainable_parameter_names'] and
                sum(p.numel() for p in params.values()) == 13274016, 'Native parameter path differs')
        bound = policy['model_sources'][name]
        require(sha(directory/'latest.pth') == bound['files_sha256']['latest.pth'], 'Final checkpoint changed before loading')
        payload = torch.load(str(directory/'latest.pth'), map_location='cpu')
        require(payload['metadata'] == manifest and payload['arm'] == arm and payload['update'] == 512 and payload['pass_index'] == 4,
                'Final checkpoint identity differs on load')
        model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True)
        del payload
        require(memory.state_digest(model.future_pred_head) == bound['final_head_state_sha256'], 'Actual loaded head differs')
        for p in model.parameters(): p.requires_grad_(False)
        model.eval(); guard = runtime_guard(model)
        receipts[name] = dict(checkpoint_sha256=bound['files_sha256']['latest.pth'],
            final_head_state_sha256=memory.state_digest(model.future_pred_head),
            full_model_state_sha256=memory.state_digest(model),
            complete_readout_state_sha256=memory.state_digest(model.future_pred_head.bev_pred_head),
            readout_parameter_names=list(dict(model.future_pred_head.bev_pred_head.named_parameters())),
            readout_buffer_names=list(dict(model.future_pred_head.bev_pred_head.named_buffers())),
            guard=guard, runtime_methods=code_receipt(model, objective), loss_adapter_installed=False,
            optimizer_loaded=False, optimizer_constructed=False)
        result[name] = model
    return result, receipts


def capture_terminal(model, sample, cache_module):
    """Pure Module callbacks; return None and never mutate a live tensor/module."""
    import torch
    captured = [[] for _ in range(3)]; handles = []
    def hook_for(layer):
        def capture(branch, inputs):
            require(len(inputs) == 1 and torch.is_tensor(inputs[0]) and
                    list(inputs[0].shape) == [5,1,40000,256] and inputs[0].dtype == torch.float32,
                    'Unexpected terminal readout input shape/dtype')
            require(bool(torch.isfinite(inputs[0]).all()), 'Nonfinite features')
            captured[layer].append(inputs[0].detach().clone())
            return None
        return capture
    try:
        for layer, branch in enumerate(model.future_pred_head.bev_pred_head):
            handles.append(branch.register_forward_pre_hook(hook_for(layer)))
        output = cache_module.replay(model, sample, training=False)[0]
    finally:
        for handle in handles: handle.remove()
    require(all(len(v) == 1 for v in captured), 'Each terminal readout must be called exactly once')
    require(list(output.shape) == [5,3,1,1,40000,16,2] and output.dtype == torch.float32 and
            bool(torch.isfinite(output).all()), 'Unexpected native complete logits')
    features = torch.stack([v[0] for v in captured], dim=1)
    return features, output


def exact_tensor(a, b, message):
    import torch
    require(a.dtype == b.dtype and a.shape == b.shape and
            bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))), message)


def evaluate(model, logits, sample, record, expected_gt_counts):
    import numpy as np
    rows = model.evaluate_occ_records(logits, sample['targets'], sample['inputs']['img_metas'])
    require(len(rows) == 1, 'Expected one unchanged native evaluation row')
    row = rows[0]
    require(row['sample_token'] == record['sample_token'] and row['scene_token'] == record['scene_token'] and
            row['horizon_seconds'] == HORIZONS, 'Native evaluation identity/horizons differ')
    hist = np.asarray(row['hist_by_horizon'])
    require(hist.shape == (5,2,2) and np.issubdtype(hist.dtype,np.integer) and (hist >= 0).all() and
            np.array_equal(hist.sum(2),expected_gt_counts), 'Full native GT confusion contract differs')
    return hist.astype(np.int64)


def measure_anchor(models, record, ordinal, cache, cache_module, tree, expected, budget):
    import numpy as np
    import torch
    begin = time.monotonic(); sample = cache_module.load_sample(cache, record, device=budget.device)
    before = tree(sample['inputs']); target_sha = cache_module._tensor_digest(sample['targets'])
    gt_counts = np.asarray(record['native_hist'], dtype=np.int64).sum(2)
    hist = {}; receipts = {}; t0_logits = {}; t0_features = None
    for source in ('C','O'):
        budget.check(); runtime_guard(models[source])
        features, original = capture_terminal(models[source], sample, cache_module)
        feature_sha = cache_module._tensor_digest(features)
        if source == 'C': t0_features = features[0].detach().cpu()
        else: exact_tensor(features[0].cpu(), t0_features, 'C/O t0 input features must be bitwise identical')
        original_sha = cache_module._tensor_digest(original)
        for readout in ('C','O'):
            name = source+readout
            logits = models[readout].future_pred_head.forward_head(features)
            require(bool(torch.isfinite(logits).all()), 'Nonfinite cross/diagonal logits')
            if source == readout:
                exact_tensor(logits, original, 'Captured diagonal differs from same-call native 5h x3 logits')
            hist[name] = evaluate(models[readout], logits, sample, record, gt_counts)
            if source == readout:
                require(np.array_equal(hist[name], expected[source][ordinal]), 'Diagonal does not reproduce historical five-horizon full-GT confusion')
            t0_logits[name] = logits[0].detach().cpu()
            receipts[name] = dict(feature_source=source, readout_source=readout,
                logits_sha256=cache_module._tensor_digest(logits), feature_sha256=feature_sha,
                diagonal_same_call_native_all_5h_3layer_logits_bitwise_equal=True if source == readout else None,
                diagonal_historical_all_5h_confusion_equal=True if source == readout else None)
            del logits
            budget.check()
        require(cache_module._tensor_digest(features) == feature_sha, 'Captured features changed during readout')
        require(cache_module._tensor_digest(original) == original_sha, 'Original native logits changed during readout')
        receipts[source+source]['same_call_native_logits_sha256'] = original_sha
        require(tree(sample['inputs']) == before and cache_module._tensor_digest(sample['targets']) == target_sha,
                'Native boundary or target changed during diagnostic')
        del features, original
    exact_tensor(t0_logits['OC'], t0_logits['CC'], 't0 OC must equal CC at all three layers/voxels')
    exact_tensor(t0_logits['CO'], t0_logits['OO'], 't0 CO must equal OO at all three layers/voxels')
    torch.cuda.synchronize(); budget.check()
    return dict(schema='readout-transition-sample-v1', ordinal=ordinal, **identity(record),
        horizon_seconds=HORIZONS, hist_by_combination={n:hist[n].tolist() for n in NAMES},
        inputs_full_typed_sha256=before, target_tensor_sha256=target_sha,
        common_replay_valid_frames=[], original_replay_clones_boundary=True,
        input_and_target_unchanged=True, features_unchanged=True,
        t0_features_C_equal_O_bitwise=True, t0_all_layers_OC_equal_CC_bitwise=True,
        t0_all_layers_CO_equal_OO_bitwise=True, combinations=receipts,
        seconds=time.monotonic()-begin, maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def summarize(rows, aggregate, metric):
    """Two conditional paths plus interaction; never a unique causal allocation."""
    import numpy as np
    hist = {n:np.asarray([r['hist_by_combination'][n] for r in rows], dtype=np.int64) for n in NAMES}
    totals = np.stack([hist[n].sum(0) for n in NAMES]); point = metric.metric_arrays(totals)
    boot, bootstrap = aggregate.paired_bootstrap(hist, [r['scene_token'] for r in rows], NAMES, metric.metric_arrays)
    bootstrap.update(primary_contrasts=0, posthoc=True, unique_causal_attribution=False)
    keys = aggregate.METRIC_KEYS; models = {}; effects = {}
    for i, name in enumerate(NAMES):
        models[name] = dict(hist_by_horizon=totals[i].tolist(),
            gmo_iou_by_horizon=point['gmo_iou_by_horizon'][i].tolist(),
            binary_miou_by_horizon=point['binary_miou_by_horizon'][i].tolist(),
            metrics={k:dict(estimate=float(point[k][i]),ci95=aggregate.interval(boot[k][:,i]),unit='ratio') for k in keys})
    for label, weights in EFFECTS.items():
        w = np.asarray(weights, dtype=np.float64); values = {}
        for key in tuple(keys)+('gmo_iou_by_horizon',):
            estimate = np.tensordot(w,point[key],axes=(0,0))
            draws = np.tensordot(boot[key],w,axes=(1,0))
            ci = np.quantile(draws,[.025,.975],axis=0)
            values[key] = dict(delta_ratio=estimate.tolist(),delta_pp=(100*estimate).tolist(),
                ci95_ratio=ci.tolist(),ci95_pp=(100*ci).tolist())
        effects[label] = dict(coefficients=dict(zip(NAMES,weights)),metrics=values)
    return models, effects, bootstrap


class Budget:
    def __init__(self, a, started):
        self.started = started; self.end = started+a.max_seconds; self.device = a.device
        self.cap = a.max_allocated_gib*1024**3
    def check(self):
        require(time.monotonic() < self.end, 'Absolute diagnostic deadline reached')
        import torch
        require(torch.cuda.max_memory_allocated() <= self.cap, 'Allocated-memory ceiling exceeded')


def render_report(out, result):
    lines = ['# C/O 特征与读出的事后 2×2 诊断', '',
        '组合第一位是未来特征来源，第二位是完整三层 readout 来源。C 为固定 native1 最终权重，O 为固定 CE+Lovász 最终权重；不安装训练损失适配器，不训练、不拟合阈值或 stitching，不提名新模型。', '',
        'CC/OO 的全部五时域三层 logits 与同次原生回放逐位一致；历史开发记录只能验证逐样本五时域完整 GT 混淆，不包含历史全层 logits 缓存。t0 特征 C=O，所有三层全部 voxel 上 OC=CC、CO=OO。完整输入、GT 和模型状态保留。', '']
    if result['mode'] == 'pilot':
        lines += ['这是固定两场景各一 anchor 的工程检查；没有运行 200 样本机制分析，不输出效能成功结论。',
                  f"实际样本循环耗时 {result['resources']['anchor_seconds_total']:.3f}s；CUDA allocated 峰值 {result['resources']['maximum_cuda_allocated_bytes']/1024**3:.3f} GiB。预算只能据实测再冻结；冷启动、来源审计与 CPU bootstrap 开销也计入总上限。"]
    else:
        lines += ['完整固定 dev200 / 100 scenes，每场景两 anchors；单训练 seed11、历史验证集已曝光、不是 full validation 或新盲测。', '',
            '| 组合 | Future macro GMO % |', '|---|---:|']
        for name, model in result['models'].items():
            lines.append(f"| {name} | {100*model['metrics']['future_macro_gmo']['estimate']:.4f} |")
        lines += ['', '| 条件路径或交互 | Δ pp [95% scene CI] |', '|---|---:|']
        for name, effect in result['effects'].items():
            v = effect['metrics']['future_macro_gmo']; ci = v['ci95_pp']
            lines.append(f"| {name} | {v['delta_pp']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines += ['', '两条代数路径：OO−CC=(CO−CC)+(OO−CO)=(OC−CC)+(OO−OC)。交互定义 OO−OC−CO+CC，属于所选非线性指标尺度上的差中差。',
            '跨读出可能受潜空间重参数化与共同适应影响；差异不能唯一归因于信息损失、transition 或 readout 的因果份额，不计算唯一贡献百分比。',
            'GMO 主量为每个未来时域分别池化完整 GT 混淆、计算 class1 IoU，再平均四时域；GMO 为可移动语义类而非真实运动。binary mIoU 与 GMO 分列。10,000 次 seed11 场景配对 bootstrap 同时保留每场景两 anchors，区间未作多重比较校正，不含训练种子或选择不确定性。']
    lines += ['', '只保存小型混淆矩阵、身份、哈希与资源记录；未持久化 features/logits。没有自动训练、候选选择、预算扩张或失败重试。', '']
    (out/'report.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pilot','development'), required=True)
    for name in ('repo','config','checkpoint','protocol','objective-protocol','diagnostic-protocol','cache',
                 'native-runs-root','runs-root','development-summary','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--pilot'); parser.add_argument('--authorization')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--max-allocated-gib', type=float, required=True)
    a = parser.parse_args(argv); started = time.monotonic(); out = Path(a.out).resolve()
    require(0 < a.max_seconds <= 1800 and 0 < a.max_allocated_gib <= 32 and a.device.startswith('cuda:'), 'Invalid bounded CUDA execution')
    require(not out.exists(), 'A new output directory is required; no overwrite/retry')
    out.mkdir(parents=True, exist_ok=False)
    def stop(signum, frame):
        raise TimeoutError('Diagnostic stopped by deadline/signal '+str(signum))
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, a.max_seconds)
    try:
        policy, development = load_policy(a); phase = phase_gate(a, policy)
        aggregate, parent, objective, metric, cache, records, expected, evidence = load_evidence(a, policy, development)
        import torch
        import numpy as np
        torch.cuda.set_device(torch.device(a.device))
        budget = Budget(a, started)
        torch.cuda.set_per_process_memory_fraction(min(1.,budget.cap/torch.cuda.get_device_properties(a.device).total_memory), device=a.device)
        torch.cuda.reset_peak_memory_stats()
        cache_module = module('native_state_cache.py', parent['source_sha256']['native_state_cache.py'])
        memory = module('memory_experiment.py', parent['source_sha256']['memory_experiment.py'])
        tree_module = module('joint_native_evaluation.py', TREE_SHA)
        initialization_started = time.monotonic()
        models, loaded = make_models(a,parent,objective,policy,cache_module,memory)
        initialization_seconds = time.monotonic()-initialization_started
        budget.check()
        selected = first_two_scenes(records) if a.mode == 'pilot' else list(range(200))
        if phase:
            pilot = read(Path(a.pilot)/'summary.json')
            require(pilot['loaded_models'] == loaded and pilot['selected_identities'] == [identity(records[i]) for i in first_two_scenes(records)],
                    'Pilot source/model/selection receipt differs from this run')
        manifest = dict(schema='readout-transition-manifest-v1',mode=a.mode,
            diagnostic_protocol_sha256=sha(a.diagnostic_protocol),parent_protocol_sha256=PARENT_SHA,
            objective_protocol_sha256=OBJECTIVE_SHA,source_sha256=policy['source_sha256'],
            runtime_source_sha256=objective['runtime_source_sha256'],cache_index_sha256=sha(cache/'index.json'),
            cache_complete_sha256=sha(cache/'complete.json'),development_summary=policy['development_summary'],
            final_source_receipts=evidence,loaded_models=loaded,phase_gate=phase,
            selected_ordinals=selected,selected_identities=[identity(records[i]) for i in selected],
            resources=dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),optimizer_updates=0)
        write(out/'manifest.json',manifest); rows = []
        pre_sample_initialization_seconds = time.monotonic()-started
        with torch.no_grad(), (out/'records.jsonl').open('x') as stream:
            for ordinal in selected:
                row = measure_anchor(models,records[ordinal],ordinal,cache,cache_module,tree_module.tree_digest,expected,budget)
                stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); rows.append(row)
                print(json.dumps(dict(event='READOUT_DIAGNOSTIC',completed=len(rows),planned=len(selected),
                                      sample_token=row['sample_token'],seconds=row['seconds'])),flush=True)
        for name, model in models.items():
            runtime_guard(model)
            require(memory.state_digest(model) == loaded[name]['full_model_state_sha256'] and
                    memory.state_digest(model.future_pred_head.bev_pred_head) == loaded[name]['complete_readout_state_sha256'],
                    'Model parameter/buffer state changed')
        verify_files(Path(__file__).parent,policy['source_sha256'])
        for path, digest in objective['runtime_source_sha256'].items():
            require(sha(path) == digest, 'Runtime source changed during diagnostic')
        for name, directory in [('C',Path(a.native_runs_root)/'native1'),('O',Path(a.runs_root)/'O')]:
            verify_files(directory,policy['model_sources'][name]['files_sha256'])
        require(sha(cache/'index.json') == manifest['cache_index_sha256'] and
                sha(cache/'complete.json') == manifest['cache_complete_sha256'] and
                sha(a.diagnostic_protocol) == manifest['diagnostic_protocol_sha256'], 'Contract changed while running')
        budget.check()
        result = dict(schema='readout-transition-summary-v1',mode=a.mode,
            diagnostic_protocol_sha256=sha(a.diagnostic_protocol),samples=len(rows),scenes=len({r['scene_token'] for r in rows}),
            sources=dict(diagnostic_protocol_sha256=sha(a.diagnostic_protocol),
                parent_protocol_sha256=PARENT_SHA,objective_protocol_sha256=OBJECTIVE_SHA,
                script_sha256=sha(__file__),source_sha256=policy['source_sha256'],
                development_summary=policy['development_summary'],cache_index_sha256=manifest['cache_index_sha256']),
            selected_identities=manifest['selected_identities'],loaded_models=loaded,
            all_engineering_gates_passed=True,optimizer_updates=0,model_and_input_state_unchanged=True,
            phase_gate=phase,resources=dict(elapsed_seconds=time.monotonic()-started,
                model_initialization_seconds=initialization_seconds,
                pre_sample_initialization_seconds=pre_sample_initialization_seconds,
                anchor_seconds_total=sum(r['seconds'] for r in rows),anchor_seconds=[r['seconds'] for r in rows],
                max_sample_seconds=max(r['seconds'] for r in rows),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved()),
            semantics=dict(combination_order='feature source / complete readout source',posthoc=True,
                historical_validation_exposure=True,training_seeds=1,full_validation=False,new_candidate_selection=False,
                unique_causal_attribution=False,latent_reparameterization_and_coadaptation_caveat=True,
                future_features_difference_required=False,features_or_logits_persisted=False,
                historical_reference_has_full_layer_logits=False))
        if a.mode == 'development':
            result['models'],result['effects'],result['bootstrap'] = summarize(rows,aggregate,metric)
            for diagonal, oldname in [('CC','native1'),('OO','O')]:
                old = development['models'][oldname]
                require(result['models'][diagonal]['hist_by_horizon'] == [r['confusion_GT_rows_prediction_columns'] for r in old['horizons']],
                        'Development diagonal pooled counts differ')
                for key in aggregate.METRIC_KEYS:
                    require(result['models'][diagonal]['metrics'][key] == old['metrics'][key], 'Diagonal point/scene-CI differs')
        budget.check(); result['resources']['elapsed_seconds'] = time.monotonic()-started
        write(out/'summary.json',result); render_report(out,result)
        status = 'PASS_READOUT_TRANSITION_ENGINEERING' if a.mode == 'pilot' else 'COMPLETE_READOUT_TRANSITION_DIAGNOSTIC'
        write(out/'complete.json',dict(schema='readout-transition-complete-v1',status=status,mode=a.mode,
            diagnostic_protocol_sha256=sha(a.diagnostic_protocol),samples=len(rows),scenes=result['scenes'],optimizer_updates=0,
            seconds=time.monotonic()-started,files_sha256={n:sha(out/n) for n in ('manifest.json','records.jsonl','summary.json','report.md')}))
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_READOUT_TRANSITION_DIAGNOSTIC',mode=a.mode,
              error_type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-started,optimizer_updates=0,automatic_retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == '__main__':
    main()
