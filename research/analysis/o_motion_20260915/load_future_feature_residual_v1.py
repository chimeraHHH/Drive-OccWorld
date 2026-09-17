"""Read-only fixed-final R/R0 and frozen-D loaders; never restore an optimizer.

Source and formal protocol are pinned. Future checkpoint byte hashes come from
completed source-bound receipts, never from a placeholder or an intermediate.
This module was prepared before final results existed; actual tensor validation
occurs only when load_completed_arm is called on a complete formal run.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCHEMA = 'future-feature-residual-training-v1'
STATUS = 'COMPLETE_FUTURE_FEATURE_RESIDUAL_TRAINING'
PROTOCOL_SHA = '45048560948fae975945685e5151db158933b81b94b53b543551213b29a79a56'
TRAINER_SHA = '2226681ee8b6b0b7ffe40ee40540947619217f2f3cdd83d034f101ba6706cbde'
CORE_SHA = 'e427b74698ad4537f135f6dab03df9787f8661abac666164e21e48997caa541d'
COMMON_SHA = '57cf304bafb050fac05b06abe68ce4600a37ca020d9d10d0673ec0ab1edb3bc1'
ARMS = ('R', 'R0')
NAMES = ('projection.weight', 'projection.bias', 'decoder.0.weight',
         'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def source_path(name):
    require(Path(name).name == name and name.endswith('.py'), 'Python source basename required')
    for p in (Path(__file__).parent/name,
              Path(__file__).parent.parent/'m0_improvement_20260915'/name):
        if p.is_file(): return p
    raise FileNotFoundError(name)


def import_source(name, expected):
    path = source_path(name+'.py'); require(sha(path) == expected, 'Source changed: '+name)
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path.resolve(), 'Different module alias: '+name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def finite_tree(value):
    if isinstance(value, float): require(math.isfinite(value), 'Nonfinite serialized number')
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, list):
        for v in value: finite_tree(v)


def file_chain(root, receipt, expected, skip_bytes=()):
    require(set(receipt['files_sha256']) == set(expected), 'Unexpected completed artifact set')
    for name, digest in receipt['files_sha256'].items():
        require(Path(name).name == name and isinstance(digest,str) and len(digest)==64 and
                set(digest) <= set('0123456789abcdef'), 'Invalid completed file binding: '+name)
        if name not in skip_bytes: require(sha(root/name) == digest, 'Completed artifact changed: '+name)


def completed_training_metadata(training_run, protocol_path, helper, verify_checkpoint_bytes=False):
    """Authenticate small files; pth bytes are read only for a real tensor load.

    Metadata-only mode checks exact paired LRs; the real loader additionally
    checks source schedule values in its runtime (avoids claiming identical
    transcendental libm rounding across macOS metadata and Linux training).
    """
    require(isinstance(protocol_path, (str, Path)), 'Actual frozen protocol file path required')
    protocol_path = Path(protocol_path); require(sha(protocol_path) == PROTOCOL_SHA, 'Wrong formal R/R0 protocol')
    p = read(protocol_path); require(p['schema'] == SCHEMA and p['status'] == 'FROZEN', 'Formal protocol required')
    bindings = p['sources_sha256']
    require(bindings['train_future_feature_residual_v1.py'] == TRAINER_SHA and
            bindings['future_feature_residual_v1.py'] == CORE_SHA and
            bindings['common_connected_motion_evaluation_v2.py'] == COMMON_SHA, 'Changed model/loader source')
    for name, digest in bindings.items(): require(sha(source_path(name)) == digest, 'Source changed: '+name)
    require(sha(helper.__file__) == bindings['train_source_motion_v1.py'], 'Wrong supplied helper')
    trainer = import_source('train_future_feature_residual_v1', TRAINER_SHA)
    require(p['training'] == trainer.TRAINING and p['numerical_policy'] == trainer.NUMERICAL, 'Training recipe changed')
    root = Path(training_run).resolve(); complete_sha = sha(root/'complete.json')
    done = read(root/'complete.json'); manifest = read(root/'manifest.json')
    require(not (root/'failed.json').exists(), 'Failed run is not loadable')
    require(done['schema'] == SCHEMA and done['status'] == STATUS and done['mode'] == 'train' and
            done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 200,
            'Only completed formal final512/dev200 is loadable')
    file_chain(root, done, ('manifest.json', 'loaded_models.json', 'summary.json', 'development_records.jsonl'))
    require(set(done['arm_complete_sha256']) == set(done['final_checkpoints']) == set(ARMS), 'Both completed matched arms required')
    require(manifest['schema'] == SCHEMA and manifest['mode'] == 'train' and manifest['training'] == p['training'] and
            manifest['numerical_policy'] == p['numerical_policy'] and manifest['arms'] == list(ARMS) and
            manifest['planned_updates'] == 512 and manifest['trainable_names'] == list(NAMES) and manifest['seed'] == 11,
            'Final manifest scope changed')
    require(manifest['same_initial_parameters'] and manifest['independent_optimizers'] and
            not manifest['raw_or_sparse_labels_used_for_training'], 'Matched frozen-input recipe changed')
    sources = manifest['sources']
    require(sources['protocol_sha256'] == PROTOCOL_SHA and sources['sources_sha256'] == bindings and
            sources['connected_protocol_sha256'] == p['connected_protocol_sha256'] and
            sources['connected_complete_sha256'] == p['connected_complete_sha256'] and
            sources['D_checkpoint_sha256'] == p['D_checkpoint_sha256'] and
            sources['runtime_source_contract_sha256'] == p['runtime_source_contract_sha256'] and
            all(sources[k+'_cache']['index_sha256'] == p['cache_index_sha256'][k] for k in ('train','development')),
            'Final source/data binding changed')
    oracle = import_source('oracle_transport_probe', bindings['oracle_transport_probe.py'])
    require(sources['config_sha256'] == oracle.CONFIG_SHA and sources['M0_checkpoint_sha256'] == oracle.M0_SHA and
            sources['O_checkpoint_sha256'] == oracle.O_SHA, 'Original O/M0/config changed')
    initial_sha = manifest['initial_module_sha256']
    require(manifest['engineering_preflight'] == dict(complete_sha256=p['engineering']['preflight_complete_sha256'],
            initial_module_sha256=initial_sha), 'Actual preflight/fresh initialization binding changed')
    orders = helper.planned_orders(); order_sha = helper.json_hash(orders)
    require(manifest['sample_orders_sha256'] == order_sha, 'Wrong four-pass sample schedule')
    summary = read(root/'summary.json'); finite_tree(summary)
    require(summary['schema'] == SCHEMA and summary['mode'] == 'train' and summary['updates'] == 512 and
            summary['examples'] == 2048 and summary['evaluated_samples'] == 200 and
            summary['initial_module_sha256'] == initial_sha and summary['weights_persisted'] and
            all(summary[k] for k in ('all_gradients_finite','upstream_task_gradient_reached','zero_readout_changed',
                                    'initial_O_parity_pass','frozen_O_D_unchanged')), 'Incomplete final training proof')
    loaded = read(root/'loaded_models.json'); d = loaded['D']
    require(loaded['O_full_state_sha256'] == manifest['frozen_O_state_sha256'] and not loaded['optimizer_restored'] and
            d['actual_motion_state_sha256'] == manifest['frozen_D_motion_state_sha256'] and
            d['top_complete_sha256'] == p['connected_complete_sha256'] and d['checkpoint_sha256'] == p['D_checkpoint_sha256'] and
            d['protocol_sha256'] == p['connected_protocol_sha256'] and d['actual_optimizer_parameter_steps_all512'] and
            d['fixed_final_update'] == 512 and not d['optimizer_restored'] and d['optimizer_updates_in_this_evaluation'] == 0,
            'Actual frozen O/D load receipt changed')
    logs = {}; manifests = {}
    for arm in ARMS:
        folder = root/'runs'/arm; require(not (folder/'failed.json').exists(), 'Failed arm is not loadable')
        ac = read(folder/'complete.json'); am = read(folder/'manifest.json')
        require(sha(folder/'complete.json') == done['arm_complete_sha256'][arm], 'Arm complete changed')
        require(ac['schema'] == SCHEMA and ac['mode'] == 'train' and ac['arm'] == arm and ac['status'] == STATUS and
                ac['updates'] == 512 and ac['examples'] == 2048 and ac['evaluated_samples'] == 200, 'Unequal final arm budget')
        file_chain(folder, ac, ('manifest.json','training.jsonl','final.pth','development_records.jsonl'),
                   skip_bytes=() if verify_checkpoint_bytes else ('final.pth',))
        require(am == dict(manifest, arm=arm, transport='actual_D' if arm == 'R' else 'zero',
                           common_manifest_sha256=sha(root/'manifest.json')), 'Per-arm recipe differs')
        final = done['final_checkpoints'][arm]
        require(final == dict(file='runs/'+arm+'/final.pth', sha256=ac['files_sha256']['final.pth'],
                              module_state_sha256=ac['final_module_sha256']) and
                summary['final_module_sha256'][arm] == final['module_state_sha256'], 'Final pointer/state differs')
        rows = [json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        require(len(rows) == 512, 'Incomplete actual update log'); finite_tree(rows)
        for u, row in enumerate(rows):
            require(row['update'] == u+1 and row['examples'] == 4*(u+1) and row['pass_index'] == u//128 and
                    row['lr'] > 0 and (not verify_checkpoint_bytes or row['lr'] == helper.schedule_lr(u)) and len(row['samples']) == 4 and
                    [x['ordinal'] for x in row['samples']] == orders[u//128][4*(u%128):4*(u%128+1)] and
                    row['sample_group_sha256'] == helper.json_hash([x['sample_token'] for x in row['samples']]),
                    'Actual training schedule/order differs')
            require(set(row['parameter_gradient_norms']) == set(NAMES) and
                    all(v >= 0 for v in row['parameter_gradient_norms'].values()) and row['preclip_grad_norm'] >= 0 and
                    row['clip_factor'] == min(1.,35./(row['preclip_grad_norm']+1e-6)), 'Invalid gradient/clip record')
        require(all(any(row['parameter_gradient_norms'][n] > 0 for row in rows) for n in NAMES), 'Some parameter never received task gradient')
        logs[arm] = rows; manifests[arm] = am
    for r, r0 in zip(logs['R'], logs['R0']):
        require(r['lr'] == r0['lr'] and r['sample_group_sha256'] == r0['sample_group_sha256'], 'Arm order/LR differs')
        for a,b in zip(r['samples'],r0['samples']):
            require(all(a[k] == b[k] for k in ('ordinal','sample_token','scene_token','inputs_sha256','targets_sha256','input_tree_sha256')),
                    'Matched arms consumed different samples')
    dev = [json.loads(x) for x in (root/'development_records.jsonl').read_text().splitlines()]
    require(len(dev) == 200 and len({r['sample_token'] for r in dev}) == 200 and
            len({r['scene_token'] for r in dev}) == 100, 'Incomplete fixed development200')
    for arm in ARMS:
        armdev = [json.loads(x) for x in (root/'runs'/arm/'development_records.jsonl').read_text().splitlines()]
        require(len(armdev) == 200, 'Missing per-arm final evaluation')
        for i,(r,a) in enumerate(zip(dev,armdev)):
            require(r['ordinal'] == a['ordinal'] == i and set(r['hist_by_arm']) == set(r['metrics_by_arm']) == {'O','R','R0'} and
                    all(r[k] == a[k] for k in ('sample_token','scene_token','official_index','split')) and
                    r['hist_by_arm'][arm] == a['hist_by_horizon'] and a['horizon_seconds'] == [0,.5,1,1.5,2] and
                    all(r[k] for k in ('O_reference_hist_exact','native_CPU_hist_exact','t0_all_arms_exact','GT_raw_read_after_predictions')),
                    'Final evaluation identity/parity record differs')
    require(sha(root/'complete.json') == complete_sha, 'Completion changed while validating')
    return dict(root=root, protocol=p, protocol_path=protocol_path, complete=done, manifest=manifest,
                summary=summary, loaded_models=loaded, manifests=manifests, sources=sources, orders=orders,
                order_sha256=order_sha, top_complete_sha256=complete_sha,
                checkpoint_bytes_verified=verify_checkpoint_bytes, actual_checkpoint_tensors_verified=False)


def authenticated_training(training_run, protocol_path):
    """CPU small-artifact interface for a merger; does not open any pth file."""
    require(sha(protocol_path) == PROTOCOL_SHA, 'Wrong formal R/R0 protocol')
    p = read(protocol_path)
    helper = import_source('train_source_motion_v1',p['sources_sha256']['train_source_motion_v1.py'])
    return completed_training_metadata(training_run,protocol_path,helper,verify_checkpoint_bytes=False)


def load_completed_arm(training_run, arm, protocol_path, helper, device='cuda:0'):
    """Return (frozen eval residual module, strict actual final-state receipt)."""
    require(arm in ARMS, 'Expected R or R0')
    meta = completed_training_metadata(training_run, protocol_path, helper, verify_checkpoint_bytes=True)
    import torch
    p = meta['protocol']; root = meta['root']; folder = root/'runs'/arm
    checkpoint = folder/'final.pth'; before = checkpoint.stat()
    payload = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    require(payload['schema'] == SCHEMA and payload['mode'] == 'train' and payload['arm'] == arm and
            payload['status'] == 'FIXED_FINAL_512' and payload['update'] == payload['matched_updates_completed'] == 512 and
            payload['examples'] == 2048 and payload['incomplete_accumulation_examples'] == 0 and not payload['resume_supported'],
            'Only exact fixed-final payload is loadable')
    require(payload['sources'] == meta['sources'] and payload['manifest_sha256'] == sha(folder/'manifest.json') and
            payload['sample_orders'] == meta['orders'] and payload['sample_orders_sha256'] == meta['order_sha256'] and
            payload['initial_module_sha256'] == meta['manifest']['initial_module_sha256'], 'Actual payload provenance differs')
    import_source('transport_ops',p['sources_sha256']['transport_ops.py'])
    core = import_source('future_feature_residual_v1', CORE_SHA)
    # Original trainer initializes on CPU after seed11, then transfers to CUDA.
    # Preserve caller RNG; no optimizer, fitting or CUDA allocation is needed here.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11); module = core.FutureFeatureResidual()
    require(helper.state_digest(module) == payload['initial_module_sha256'], 'Fresh seed11 initial state differs')
    require(sum(v.numel() for v in module.parameters()) == 41137 and
            [n for n,_ in module.named_parameters()] == list(NAMES), 'Residual parameter scope changed')
    state = payload['residual_module']; initial = module.state_dict()
    require(set(state) == set(initial), 'State keys differ')
    for key, expected in initial.items():
        value = state[key]
        require(isinstance(value,torch.Tensor) and value.shape == expected.shape and value.dtype == expected.dtype == torch.float32 and
                bool(torch.isfinite(value).all()), 'Invalid actual state tensor: '+key)
        if key not in NAMES: require(torch.equal(value,expected), 'Nontrainable buffer changed: '+key)
    module.load_state_dict(state,strict=True); actual = helper.state_digest(module)
    final = meta['complete']['final_checkpoints'][arm]
    require(actual == payload['final_module_sha256'] == final['module_state_sha256'], 'Actual final residual tensor SHA differs')
    optimizer = payload['optimizer']; groups = optimizer['param_groups']; states = optimizer['state']
    require(len(groups) == 1, 'Unexpected optimizer group count'); group = groups[0]
    require(group['lr'] == helper.schedule_lr(511) and tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8 and
            group['weight_decay'] == .01 and group['amsgrad'] is False and not group.get('maximize',False) and
            not group.get('capturable',False) and not group.get('differentiable',False), 'Final AdamW policy differs')
    parameters = list(module.parameters())
    require(group['params'] == list(range(len(parameters))) and set(states) == set(group['params']), 'Missing optimizer parameter state')
    for index, parameter in enumerate(parameters):
        s = states[index]
        require(set(s) == {'step','exp_avg','exp_avg_sq'} and isinstance(s['step'],torch.Tensor) and s['step'].numel() == 1 and
                bool(torch.isfinite(s['step']).all()) and float(s['step']) == 512., 'Invalid actual AdamW step')
        for key in ('exp_avg','exp_avg_sq'):
            require(s[key].shape == parameter.shape and s[key].dtype == parameter.dtype and bool(torch.isfinite(s[key]).all()),
                    'Invalid AdamW moment shape/dtype/value')
        require(bool((s['exp_avg_sq'] >= 0).all()), 'Negative AdamW second moment')
    after = checkpoint.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino) == (after.st_size,after.st_mtime_ns,after.st_ino) and
            sha(checkpoint) == final['sha256'] and sha(root/'complete.json') == meta['top_complete_sha256'], 'Artifact changed during load')
    del payload, optimizer, states, state
    for parameter in module.parameters(): parameter.requires_grad_(False)
    module.to(device).eval()
    receipt = dict(schema='future-feature-residual-final-load-v1',arm=arm,checkpoint_sha256=final['sha256'],
        actual_module_state_sha256=actual,initial_module_sha256=meta['manifest']['initial_module_sha256'],
        manifest_sha256=sha(folder/'manifest.json'),common_manifest_sha256=sha(root/'manifest.json'),
        top_complete_sha256=meta['top_complete_sha256'],arm_complete_sha256=meta['complete']['arm_complete_sha256'][arm],
        protocol_sha256=PROTOCOL_SHA,loader_source_sha256=sha(__file__),sources_sha256=p['sources_sha256'],
        sample_orders_sha256=meta['order_sha256'],parameters=41137,fixed_final_update=512,fixed_examples=2048,
        fixed_development_samples=200,actual_optimizer_parameter_steps_all512=True,
        optimizer_restored=False,optimizer_updates_in_this_evaluation=0)
    return module,receipt


def load_frozen_D(connected_training_run, connected_protocol_path, residual_training_run,
                  residual_protocol_path, helper, device='cuda:0'):
    """Delegate actual final D to the existing strict loader and discard its gate."""
    meta = completed_training_metadata(residual_training_run,residual_protocol_path,helper); p = meta['protocol']
    require(sha(connected_protocol_path) == p['connected_protocol_sha256'] and
            sha(Path(connected_training_run)/'complete.json') == p['connected_complete_sha256'], 'Wrong D source run')
    trainer = import_source('train_connected_motion_v2',p['sources_sha256']['train_connected_motion_v2.py'])
    motion,unused_gate,receipt = trainer.load_completed_arm(Path(connected_training_run)/'runs'/'D',Path(connected_protocol_path),helper,device)
    del unused_gate
    require(receipt == meta['loaded_models']['D'] and helper.state_digest(motion) == meta['manifest']['frozen_D_motion_state_sha256'],
            'Actual D differs from R/R0 frozen training source')
    return motion,dict(receipt,residual_training_complete_sha256=meta['top_complete_sha256'],
                       residual_protocol_sha256=PROTOCOL_SHA,old_D_gate_discarded=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--protocol',required=True);parser.add_argument('--device',default='cpu')
    a = parser.parse_args();p = read(a.protocol)
    helper = import_source('train_source_motion_v1',p['sources_sha256']['train_source_motion_v1.py'])
    _,receipt = load_completed_arm(a.training_run,a.arm,a.protocol,helper,a.device)
    print(json.dumps(receipt,indent=2,allow_nan=False))


if __name__ == '__main__': main()
