"""Authenticate fixed Cpl/Fix training records and actual final checkpoint tensors.

No training, selection, recovery or model forward. Metadata inspection is CPU
only. validate_payload receives already loaded final tensors and strict-loaded
modules; it never constructs an optimizer or changes model parameters.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

TRAINER_SHA = '3378bf608b262264eed99124f323b1e2a517faaf706b612256fb8ba2b0156ff3'
ARMS = ('Cpl', 'Fix')


def trainer_module():
    path = Path(__file__).with_name('train_shared_rigid_state_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != TRAINER_SHA:
        raise ValueError('Fixed trainer source changed')
    name = path.stem
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve(): raise ValueError('Wrong trainer module')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = trainer_module()
require, sha, read = trainer.require, trainer.sha, trainer.read


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ledger(root, values, names, skip=()):
    require(set(values) == set(names), 'Artifact ledger differs')
    for name, digest in values.items():
        require(Path(name).name == name and len(digest) == 64, 'Invalid file binding')
        if name not in skip: require(sha(Path(root)/name) == digest, 'Artifact changed: ' + name)


def authenticate(root, expected_complete_sha, protocol_path, selection_path, sparse_manifest_path):
    """Check all 512 paired update records; checkpoint bytes are checked on load."""
    root = Path(root); p = read(protocol_path)
    require(not (root/'failed.json').exists() and sha(root/'complete.json') == expected_complete_sha,
            'Need the named successful terminal training run')
    done = read(root/'complete.json')
    require(p['schema'] == trainer.SCHEMA and p['status'] == 'FROZEN' and p['training'] == trainer.TRAINING,
            'Training recipe changed')
    require(done['schema'] == trainer.SCHEMA and done['status'] == trainer.STATUS and done['mode'] == 'training_only'
            and done['updates'] == 512 and done['examples'] == 2048 and done['evaluated_samples'] == 0
            and done['experiment_complete'] is False and done['development_evaluation_performed'] is False,
            'Require fixed-final512, without previous development selection')
    ledger(root, done['files_sha256'], ('manifest.json', 'summary.json'))
    manifest, summary = read(root/'manifest.json'), read(root/'summary.json')
    require(manifest['schema'] == trainer.SCHEMA and manifest['training'] == trainer.TRAINING and
            manifest['numerical_policy'] == p['numerical_policy'] and manifest['arms'] == list(ARMS) and
            manifest['sources']['protocol_sha256'] == sha(protocol_path) and
            manifest['sources']['sources_sha256'] == p['sources_sha256'] and
            manifest['sources']['geometry_cache_complete_sha256'] == p['geometry_cache_complete_sha256'] and
            manifest['preflight_weights_loaded'] is False and manifest['development_evaluation'] is False and
            manifest['GT_not_passed_to_forward'] is True, 'Training manifest differs')
    require(summary['schema'] == trainer.SCHEMA and summary['status'] == trainer.STATUS and
            summary['updates'] == 512 and summary['examples'] == 2048 and summary['evaluated_samples'] == 0 and
            summary['frozen_non_head_O_unchanged'] and summary['paired_rng_all_examples_exact'] and
            summary['all_logged_losses_gradients_parameters_moments_finite'] and
            summary['preflight_weights_loaded'] is False, 'Incomplete training summary')
    require(set(done['arm_complete_sha256']) == set(done['final_checkpoints']) == set(ARMS), 'Both arms required')
    contract = trainer.pre._contract_loader()
    for name, digest in p['sources_sha256'].items(): require(sha(contract.source_path(name)) == digest, 'Source changed: '+name)
    helper = contract.import_bound('train_source_motion_v1', p['sources_sha256'])
    orders = helper.planned_orders(); flat = [i for order in orders for i in order]
    schedules = dict(head=[trainer.u.head_lr(i) for i in range(512)], new=[helper.schedule_lr(i) for i in range(512)])
    require(manifest['sample_orders'] == orders and manifest['sample_orders_sha256'] == helper.json_hash(orders)
            and manifest['schedules'] == schedules, 'Training order/schedules changed')
    require(sha(selection_path) == p['selection_sha256'] and sha(sparse_manifest_path) == p['labels']['manifest_sha256'],
            'Original sample/physical-label contract changed')
    selected = [r for r in read(selection_path)['records'] if r['split'] == 'train']
    labels = {r['identity']['sample_token']:r for r in read(sparse_manifest_path)['records']}
    require(len(selected) == 512, 'Training selection incomplete')
    names = dict(head=manifest['future_head_parameter_names'], new=manifest['new_parameter_names'])
    require(len(names['head']) == 130 and len(names['new']) == 11 and
            all(len(v) == len(set(v)) for v in names.values()), 'Parameter-name contract differs')
    logs, counts = {}, {}
    for arm in ARMS:
        folder = root/'runs'/arm
        require(sha(folder/'complete.json') == done['arm_complete_sha256'][arm], 'Arm completion changed')
        ac, am = read(folder/'complete.json'), read(folder/'manifest.json')
        require(ac['schema'] == trainer.SCHEMA and ac['status'] == trainer.STATUS and ac['arm'] == arm and
                ac['updates'] == 512 and ac['examples'] == 2048 and ac['evaluated_samples'] == 0, 'Arm incomplete')
        ledger(folder, ac['files_sha256'], ('manifest.json', 'training.jsonl', 'final.pth'), skip=('final.pth',))
        checkpoint = done['final_checkpoints'][arm]
        require(checkpoint == summary['final_checkpoints'][arm] == ac['final_checkpoint'] and
                checkpoint['file'] == 'runs/'+arm+'/final.pth' and checkpoint['sha256'] == ac['files_sha256']['final.pth'],
                'Fixed final checkpoint binding differs')
        require(am['schema'] == trainer.SCHEMA and am['arm'] == arm and
                am['common_manifest_sha256'] == sha(root/'manifest.json') and
                am['coupled_spatial_address'] == (arm == 'Cpl') and am['learned_values_include_pose_residual'] and
                am['physical_state_learned'], 'Arm mechanism differs')
        logs[arm] = jsonl(folder/'training.jsonl'); require(len(logs[arm]) == 512, 'Missing update records')
        counts[arm] = {k:{n:0 for n in values} for k,values in names.items()}
        for update, row in enumerate(logs[arm]):
            require(row['update'] == update+1 and row['examples'] == (update+1)*4 and row['pass_index'] == update//128 and
                    len(row['samples']) == 4 and row['future_head_lr'] == schedules['head'][update] and
                    row['new_lr'] == schedules['new'][update], 'Update/order/LR mismatch')
            for offset, sample in enumerate(row['samples']):
                ordinal = flat[4*update+offset]; expected = selected[ordinal]
                require(sample['ordinal'] == ordinal and all(sample[k] == expected[k] for k in
                        ('sample_token','scene_token','official_index','split')) and
                        sample['sparse_label_sha256'] == labels[sample['sample_token']]['sha256'] and
                        sample['hook_counts'] == [1,1,1], 'Actual sample/label/hook differs')
                for key in ('physical_loss','total_loss'): require(math.isfinite(sample[key]), 'Nonfinite logged objective')
                require(set(sample['occupancy_loss']['optimized_keys']) == {
                    'loss_voxel_'+f+'_inter_'+str(i) for f in ('ce','lovasz') for i in range(3)}, 'O objective keys differ')
            for group in names:
                norms = row['gradients'][group]['stats']['parameter_norms']
                require(set(norms) == set(names[group]), 'Gradient parameter set differs')
                for name, value in norms.items():
                    require(value is None or (math.isfinite(value) and value >= 0), 'Nonfinite gradient norm')
                    counts[arm][group][name] += int(value is not None)
        for group in names:
            actual = summary['actual_optimizer_steps'][arm][group]
            require(actual['actual_non_none_gradient_updates'] == counts[arm][group], 'Optimizer counts disagree with all update logs')
            for name, count in counts[arm][group].items():
                state = actual['named_parameter_states'][name]
                require((state is None and count == 0) or (state is not None and state['step'] == count and 0 < count <= 512),
                        'Actual parameter step count differs')
    paired_keys = ('ordinal','sample_token','scene_token','official_index','split','inputs_sha256','targets_sha256',
                   'sparse_label_sha256','input_tree_sha256','geometry_tree_sha256','state_source',
                   'paired_rng_before_sha256','paired_rng_after_sha256')
    for left, right in zip(logs['Cpl'], logs['Fix']):
        for a,b in zip(left['samples'],right['samples']):
            require(all(a[k] == b[k] for k in paired_keys), 'Paired data or RNG differ')
    receipt = dict(complete_sha256=expected_complete_sha, protocol_sha256=sha(protocol_path),
        manifest_sha256=sha(root/'manifest.json'), summary_sha256=sha(root/'summary.json'),
        final_checkpoints=done['final_checkpoints'], sample_orders_sha256=manifest['sample_orders_sha256'],
        updates=512, examples=2048, logs_and_real_non_none_counts_verified=True,
        final_checkpoint_tensor_validation_pending=True)
    return manifest, summary, counts, receipt


def validate_payload(torch, payload, arm, model, objects, manifest, summary, counts, helper):
    """Check tensor contents, optimizer moments, actual steps and saved RNG."""
    require(payload['schema'] == trainer.SCHEMA and payload['status'] == 'FIXED_FINAL_512_PENDING_FIXED_DEV200'
            and payload['arm'] == arm and payload['update'] == 512 and payload['examples'] == 2048 and
            payload['sources'] == manifest['sources'] and payload['sample_orders'] == manifest['sample_orders'] and
            payload['sample_orders_sha256'] == manifest['sample_orders_sha256'] and payload['schedules'] == manifest['schedules'] and
            payload['schedule_updates_applied'] == 512 and payload['actual_optimizer_steps'] == summary['actual_optimizer_steps'][arm] and
            payload['preflight_weights_loaded'] is False and payload['development_evaluation_performed'] is False,
            'Loaded checkpoint endpoint contract differs')
    require(trainer.u.rng_digest(payload['rng_state']) == payload['rng_state_sha256'] == summary['final_rng_sha256'],
            'Actual saved RNG differs')
    for key in ('initial_future_head_sha256','initial_new_module_sha256','frozen_non_head_state_sha256'):
        require(payload[key] == manifest[key], 'Initial/frozen model provenance differs')
    require(trainer.u.non_head_digest(model) == manifest['frozen_non_head_state_sha256'], 'Original observer changed')
    actual = dict(head=helper.state_digest(model.future_pred_head), new=helper.state_digest(objects))
    require(actual['head'] == payload['final_future_head_sha256'] == summary['final_future_head_sha256'][arm] and
            actual['new'] == payload['final_new_module_sha256'] == summary['final_new_module_sha256'][arm],
            'Actual final parameter tensors differ')
    states = {}
    for group, module in (('head',model.future_pred_head), ('new',objects)):
        named = list(module.named_parameters()); optimizer = payload['optimizers'][group]
        require(payload['optimizer_parameter_names'][group] == [n for n,_ in named] and len(optimizer['param_groups']) == 1,
                'Optimizer parameter mapping differs')
        g = optimizer['param_groups'][0]; ids = g['params']
        require(len(ids) == len(named) == len(set(ids)) and set(optimizer['state']) <= set(ids) and
                g['lr'] == manifest['schedules'][group][-1] and tuple(g['betas']) == (.9,.999) and
                g['eps'] == 1e-8 and g['weight_decay'] == .01 and g['amsgrad'] is False, 'Actual optimizer recipe differs')
        states[group] = {}
        for key,(name,param) in zip(ids,named):
            require(bool(torch.isfinite(param).all()), 'Nonfinite final parameter')
            count = counts[arm][group][name]; item = optimizer['state'].get(key)
            if count == 0:
                require(item is None, 'Unexpected actual Adam state for unused parameter');states[group][name] = None;continue
            require(item is not None and set(item) == {'step','exp_avg','exp_avg_sq'} and item['step'].numel() == 1 and
                    float(item['step'].item()) == count, 'Actual Adam step differs')
            for field in ('exp_avg','exp_avg_sq'):
                tensor = item[field]
                require(tensor.dtype == param.dtype and tuple(tensor.shape) == tuple(param.shape) and
                        bool(torch.isfinite(tensor).all()), 'Actual Adam moment shape/dtype/finite differs')
                if field == 'exp_avg_sq': require(bool((tensor >= 0).all()), 'Negative Adam second moment')
                norm = float(tensor.double().norm())
                reported = summary['actual_optimizer_steps'][arm][group]['named_parameter_states'][name][field+'_norm']
                require(math.isclose(norm,reported,rel_tol=1e-10,abs_tol=1e-12), 'Actual moment norm differs beyond CPU/GPU reduction tolerance')
            states[group][name] = count
    return dict(arm=arm, actual_tensor_digests=actual, optimizer_parameter_steps=states,
                saved_rng_sha256=payload['rng_state_sha256'], actual_checkpoint_tensors_checked=True,
                optimizer_restored=False, optimizer_updates_performed=0)
