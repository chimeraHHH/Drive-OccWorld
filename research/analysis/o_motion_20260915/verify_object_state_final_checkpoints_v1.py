"""Read-only CPU verification of completed fixed512 V/G checkpoint tensors.

No native model construction, forward, optimizer restoration or GPU operation.
The original, byte-pinned O checkpoint supplies the native head tensor schema;
the two new modules are constructed on CPU in the trainer's seed11 order.
Frozen D and non-head O are absent from the final payload: their provenance is
checked against existing receipts, not misrepresented as a new tensor check.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'm0_improvement_20260915'
SOURCE_ROOT = HERE
PROTOCOL_SHA = '93bd80fa938739f40d46a13339a888a4bf1e377385e5cf269ce644f60982ccc9'
TRAINER_SHA = '2841aeaa533b4883e635014fa74d22520d94890ef1939a136f75beb8366ad162'
AUTH_SHA = '4a03aeda837e78b97ff18d27140d08602619ee96502b7f62fa82f591ca08f584'
O_SHA = 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
O_HISTORICAL_HEAD_SHA = '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
SCHEMA = 'object-state-final-checkpoint-cpu-audit-v1'
TRAIN_SCHEMA = 'object-state-forecast-training-v1'
ARMS = ('V', 'G')
COMPONENTS = (('future_pred_head', 'future_head', 13274016),
              ('motion_readout', 'readout', 745392), ('conditioner', 'conditioner', 95872))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def source_path(name):
    require(Path(name).name == name, 'Source must be a basename')
    for path in (SOURCE_ROOT / name, SOURCE_ROOT.parent / 'm0_improvement_20260915' / name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def bound_module(name, digest):
    path = source_path(name + '.py')
    require(sha(path) == digest, 'Frozen source changed: ' + name)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path.resolve(), 'Wrong imported module: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stat_identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def state_digest(state, historical=False):
    """Exact helper.state_digest list-shape domain, or original O tuple domain."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        shape = str(tuple(value.shape)) if historical else json.dumps(list(value.shape))
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(shape.encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def inspect_state(state, expected, parameter_names, torch):
    require(isinstance(state, dict) and set(state) == set(expected), 'State tensor keys differ')
    require(len(parameter_names) == len(set(parameter_names)) and set(parameter_names) <= set(state),
            'Missing/duplicate named parameters')
    rows = []
    for name, reference in expected.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                value.layout == torch.strided and value.shape == reference.shape and
                value.dtype == reference.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                'Invalid CPU FP32 tensor: ' + name)
        raw = value.detach().contiguous().numpy().tobytes()
        rows.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                         elements=value.numel(), parameter=name in parameter_names,
                         finite=True, tensor_bytes_sha256=hashlib.sha256(raw).hexdigest()))
    return dict(state_sha256=state_digest(state), tensor_count=len(rows),
                parameter_tensors=len(parameter_names),
                parameter_elements=sum(state[n].numel() for n in parameter_names), tensors=rows)


def load_cpu(path, expected_sha, torch):
    before = stat_identity(path)
    require(sha(path) == expected_sha, 'Checkpoint file checksum differs: ' + str(path))
    # These are authenticated project-produced pickle checkpoints, never arbitrary inputs.
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    require(stat_identity(path) == before and sha(path) == expected_sha, 'Checkpoint changed during CPU load')
    return payload


def expected_steps(logs, named_groups, selection):
    train = [row for row in selection['records'] if row['split'] == 'train']
    require(len(train) == 512 and len({r['scene_token'] for r in train}) == 256, 'Wrong fixed training selection')
    counts = {group: {name: 0 for name in names} for group, names in named_groups.items()}
    require(len(logs) == 512, 'Need all 512 completed updates')
    for index, row in enumerate(logs):
        require(row['update'] == index + 1 and row['examples'] == 4 * (index + 1) and
                row['pass_index'] == index // 128 and len(row['samples']) == 4, 'Update identity differs')
        for key in ('future_head_lr', 'readout_lr'):
            require(math.isfinite(row[key]) and row[key] > 0, 'Invalid logged learning rate')
        for sample in row['samples']:
            original = train[sample['ordinal']]
            require(all(sample[k] == original[k] for k in ('sample_token', 'scene_token', 'official_index', 'split')),
                    'Training sample differs from fixed selection')
            require(sample['native_logits_unmodified_by_readout'] is True and
                    all(math.isfinite(sample[k]) for k in ('total_loss', 'physical_loss')), 'Invalid native/loss log')
        for group, names in named_groups.items():
            stats = row['gradients'][group]; values = stats['parameter_norms']
            require(set(values) == set(names) and math.isfinite(stats['norm']) and stats['norm'] >= 0,
                    'Gradient parameter scope/nonfinite total differs')
            require(stats['none_parameters'] == sum(v is None for v in values.values()) and
                    stats['nonzero_parameters'] == sum(v is not None and v > 0 for v in values.values()),
                    'Gradient summary disagrees with parameter entries')
            for name, value in values.items():
                require(value is None or (math.isfinite(value) and value >= 0), 'Nonfinite logged parameter gradient')
                counts[group][name] += int(value is not None)  # Zero gradients still cause AdamW updates.
    require(all(value == 512 for value in counts['head'].values()), 'Original whole-head optimizer contract differs')
    return counts


def inspect_optimizer(optimizer, named_tensors, expected, expected_lr, declared, torch):
    require(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1,
            'Wrong AdamW state/group structure')
    group = optimizer['param_groups'][0]; states = optimizer['state']; names = list(named_tensors)
    require(group['params'] == list(range(len(names))) and
            set(states) == {i for i, name in enumerate(names) if expected[name] > 0},
            'Optimizer parameter order/missing state differs from actual gradient-presence logs')
    require(group['lr'] == expected_lr and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and
            group['weight_decay'] == .01 and group['amsgrad'] is False and not group.get('maximize', False) and
            not group.get('capturable', False) and not group.get('differentiable', False), 'Actual AdamW configuration differs')
    rows = []
    for index, name in enumerate(names):
        row = dict(name=name, optimizer_parameter_id=index, expected_step_from_non_None_gradients=expected[name])
        if expected[name] == 0:
            row.update(actual_step=0, state_present=False); rows.append(row); continue
        state = states[index]; tensor = named_tensors[name]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected AdamW state fields: ' + name)
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.ndim == 0 and
                step.dtype == torch.float32 and bool(torch.isfinite(step)) and float(step) == expected[name],
                'Actual per-parameter AdamW step differs: ' + name)
        moments = {}
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided and
                    value.shape == tensor.shape and value.dtype == tensor.dtype and bool(torch.isfinite(value).all()),
                    'Actual AdamW moment shape/dtype/finiteness differs: ' + name + '/' + key)
            if key == 'exp_avg_sq':
                require(bool((value >= 0).all()), 'Negative AdamW second moment: ' + name)
            moments[key] = dict(shape=list(value.shape), dtype=str(value.dtype), finite=True,
                               tensor_bytes_sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
        row.update(actual_step=int(step), state_present=True, step_dtype=str(step.dtype), moments=moments); rows.append(row)
    # Producer records insertion order of opt.state.values(), not sorted parameter IDs.
    steps = [int(state['step']) for state in states.values()]
    actual = dict(parameters_with_state=len(states), steps=steps, all_steps512=all(s == 512 for s in steps),
                  empty_conditioner_graph_skips_parameter_update=True)
    require(actual == declared, 'Actual optimizer payload and serialized producer receipt differ')
    return dict(configuration={k: v for k, v in group.items() if k != 'params'}, actual_step_receipt=actual,
                named_parameter_steps={r['name']: r['actual_step'] for r in rows}, parameters=rows,
                optimizer_restored=False, optimizer_steps_executed=0)


def verify(a):
    global SOURCE_ROOT
    SOURCE_ROOT = Path(a.source_root).resolve()
    require(sha(a.protocol) == PROTOCOL_SHA, 'Only the frozen formal V/G protocol is accepted')
    root = Path(a.training_run); done_sha = sha(root / 'complete.json')
    require(done_sha == a.complete_sha256, 'Actual final completion receipt SHA differs')
    protocol = read(a.protocol)
    require(protocol['sources_sha256']['object_state_forecast_train_v1.py'] == TRAINER_SHA,
            'Wrong saved-payload producer')
    auth = bound_module('summarize_object_state_common_v1', AUTH_SHA)
    _, metadata_receipt = auth.authenticate_training(root, a.protocol, a.selection, a.o_reference,
                                                     a.sparse_manifest, a.raw_manifest)
    manifest = read(root / 'manifest.json'); summary = read(root / 'summary.json')
    done = read(root / 'complete.json'); loaded = read(root / 'loaded_models.json')
    helper = bound_module('train_source_motion_v1', protocol['sources_sha256']['train_source_motion_v1.py'])
    trainer = bound_module('object_state_forecast_train_v1', TRAINER_SHA)
    require(sha(a.preflight_protocol) == protocol['preflight_protocol_sha256'], 'Preflight protocol changed')
    initial_keys = ('initial_future_head_sha256', 'initial_readout_sha256', 'initial_conditioner_sha256')
    engineering = trainer.validate_engineering(a.engineering_preflight, protocol, read(a.preflight_protocol),
                                               *(manifest[k] for k in initial_keys))
    require(manifest['engineering_preflight'] == engineering, 'Formal engineering binding differs')
    require(all(summary[k] == loaded[k] == manifest[k] for k in initial_keys) and
            loaded['optimizer_restored'] is False and loaded['O_full_state_sha256'] == manifest['initial_O_state_sha256'],
            'Initial model source/summary mismatch')
    droot = Path(a.connected_training_run)
    require(sha(droot / 'complete.json') == protocol['connected_complete_sha256'], 'Original D completion changed')
    d_done = read(droot / 'complete.json'); d_final = d_done['final_checkpoints']['D']
    require(sha(droot / 'manifest.json') == d_done['files_sha256']['manifest.json'], 'Original D manifest changed')
    d_manifest = read(droot / 'manifest.json'); d_receipt = loaded['D']
    require(d_receipt['checkpoint_sha256'] == d_final['sha256'] == protocol['D_checkpoint_sha256'] and
            d_receipt['actual_motion_state_sha256'] == d_final['motion_state_sha256'] == manifest['frozen_D_motion_state_sha256'] and
            d_receipt['actual_gate_state_sha256'] == d_final['gate_state_sha256'] and
            d_receipt['top_complete_sha256'] == protocol['connected_complete_sha256'] and
            d_receipt['protocol_sha256'] == protocol['connected_protocol_sha256'] and
            d_receipt['common_manifest_sha256'] == d_done['files_sha256']['manifest.json'] and
            d_receipt['arm_complete_sha256'] == d_done['arm_complete_sha256']['D'] and
            d_manifest['frozen_O_state_sha256'] == manifest['initial_O_state_sha256'], 'Frozen D/O provenance differs')
    d_arm = droot / 'runs' / 'D'
    require(sha(d_arm / 'complete.json') == d_receipt['arm_complete_sha256'] and
            sha(d_arm / 'manifest.json') == d_receipt['manifest_sha256'] == read(d_arm / 'complete.json')['files_sha256']['manifest.json'],
            'Original D arm receipt changed')
    require(d_receipt['fixed_final_update'] == 512 and d_receipt['fixed_examples'] == 2048 and
            d_receipt['fixed_development_samples'] == 200 and d_receipt['optimizer_restored'] is False and
            d_receipt['optimizer_updates_in_this_evaluation'] == 0, 'Wrong frozen D source endpoint')

    # All terminal metadata and checksums above precede any Torch checkpoint unpickle.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    require(not torch.cuda.is_initialized(), 'Run in a fresh CPU-only process')
    torch.set_num_threads(2)
    original = load_cpu(a.o_checkpoint, O_SHA, torch)
    require(original['arm'] == 'O' and original['update'] == 512 and original['metadata']['seed'] == 11,
            'Wrong original O checkpoint endpoint')
    original_head = original['future_pred_head']; original_names = original['metadata']['trainable_parameter_names']
    prefix = 'future_pred_head.'
    require(isinstance(original_names, list) and all(isinstance(n, str) and n.startswith(prefix) and
            n.count(prefix) == 1 for n in original_names) and len(set(original_names)) == len(original_names),
            'Original O parameter names must contain exactly one leading future_pred_head prefix and be unique')
    # Original O names came from model.named_parameters(); this trainer uses the head's own namespace.
    head_names = [name[len(prefix):] for name in original_names]
    require(head_names == manifest['future_head_parameter_names'] == list(original_head) and len(head_names) == 130 and
            original['metadata']['trainable_parameters'] == 13274016, 'Whole native head parameter scope differs')
    o_info = inspect_state(original_head, original_head, head_names, torch)
    require(o_info['parameter_elements'] == 13274016 and
            o_info['state_sha256'] == manifest['initial_future_head_sha256'] and
            state_digest(original_head, historical=True) == O_HISTORICAL_HEAD_SHA, 'Actual original O initial tensor bytes differ')
    core = bound_module('future_state_motion_v1', protocol['sources_sha256']['future_state_motion_v1.py'])
    condition_core = bound_module('object_state_conditioner_v2', protocol['sources_sha256']['object_state_conditioner_v2.py'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        readout = core.FutureStateMotionReadout()
        conditioner = condition_core.ObjectStateConditioner()
    readout_names = [n for n, _ in readout.named_parameters()]
    condition_names = [n for n, _ in conditioner.named_parameters()]
    new_names = ['readout.' + n for n in readout_names] + ['conditioner.' + n for n in condition_names]
    require(len(readout_names) == 10 and len(condition_names) == 31 and new_names == manifest['readout_parameter_names'],
            'New module parameter order/scope differs')
    require(helper.state_digest(readout) == manifest['initial_readout_sha256'] and
            helper.state_digest(conditioner) == manifest['initial_conditioner_sha256'], 'Fresh CPU seed11 initial modules differ')
    references = {'future_pred_head': original_head, 'motion_readout': readout.state_dict(), 'conditioner': conditioner.state_dict()}
    names_by_component = {'future_pred_head': head_names, 'motion_readout': readout_names, 'conditioner': condition_names}
    selection = read(a.selection); orders = helper.planned_orders(); results = {}
    for arm in ARMS:
        folder = root / 'runs' / arm; final = done['final_checkpoints'][arm]; am = read(folder / 'manifest.json')
        require(all(am[k] == manifest[k] for k in initial_keys) and am['future_head_parameters'] == 13274016 and
                am['readout_parameters'] == 745392 and am['conditioner_parameters'] == 95872, 'Arm initialization/capacity differs')
        logs = jsonl(folder / 'training.jsonl')
        expected = expected_steps(logs, {'head': head_names, 'readout': new_names}, selection)
        payload = load_cpu(folder / 'final.pth', final['sha256'], torch)
        expected_keys = {'schema', 'status', 'arm', 'future_pred_head', 'motion_readout', 'conditioner', 'optimizers',
                         'update', 'examples', 'sample_orders', 'sample_orders_sha256', 'manifest_sha256',
                         'common_manifest_sha256', 'sources', 'actual_optimizer_steps', 'resume_supported'}
        expected_keys.update(initial_keys); expected_keys.update('final_' + stem + '_sha256' for _, stem, _ in COMPONENTS)
        require(set(payload) == expected_keys and payload['schema'] == TRAIN_SCHEMA and payload['status'] == 'FIXED_FINAL_512'
                and payload['arm'] == arm and payload['update'] == 512 and payload['examples'] == 2048
                and payload['resume_supported'] is False, 'Wrong actual final payload schema/endpoint')
        require(payload['sources'] == manifest['sources'] and payload['manifest_sha256'] == sha(folder / 'manifest.json') and
                payload['common_manifest_sha256'] == sha(root / 'manifest.json') and payload['sample_orders'] == orders and
                payload['sample_orders_sha256'] == manifest['sample_orders_sha256'] == helper.json_hash(orders) and
                all(payload[k] == manifest[k] for k in initial_keys), 'Actual payload source/order/initial bindings differ')
        components = {}
        for key, stem, count in COMPONENTS:
            info = inspect_state(payload[key], references[key], names_by_component[key], torch)
            require(info['parameter_elements'] == count and info['state_sha256'] == payload['final_' + stem + '_sha256'] ==
                    final[stem + '_state_sha256'], 'Actual final component digest/count differs: ' + key)
            components[key] = info
        require(set(payload['optimizers']) == {'head', 'readout'} and
                payload['actual_optimizer_steps'] == summary['actual_optimizer_steps'][arm], 'Optimizer receipt differs')
        head = {n: payload['future_pred_head'][n] for n in head_names}
        new = {'readout.' + n: payload['motion_readout'][n] for n in readout_names}
        new.update({'conditioner.' + n: payload['conditioner'][n] for n in condition_names})
        optimizers = {}
        for group, named, lr_key in (('head', head, 'future_head_lr'), ('readout', new, 'readout_lr')):
            optimizers[group] = inspect_optimizer(payload['optimizers'][group], named, expected[group], logs[-1][lr_key],
                                                 payload['actual_optimizer_steps'][group], torch)
        results[arm] = dict(checkpoint_sha256=final['sha256'], components=components, optimizers=optimizers,
                            arm_complete_sha256=done['arm_complete_sha256'][arm], initial_bindings_exact=True)
        del payload, head, new
    for group in ('head', 'readout'):
        require(results['V']['optimizers'][group]['named_parameter_steps'] == results['G']['optimizers'][group]['named_parameter_steps'],
                'Actual per-name V/G AdamW steps differ')
    # Detect changed producer artifacts across the complete read, without recomputing any statistics.
    require(sha(root / 'complete.json') == done_sha, 'Completion receipt changed during verification')
    for name, digest in done['files_sha256'].items():
        require(sha(root / name) == digest, 'Completed artifact changed during verification: ' + name)
    for arm in ARMS:
        folder = root / 'runs' / arm
        require(sha(folder / 'complete.json') == done['arm_complete_sha256'][arm], 'Arm endpoint changed')
        for name, digest in read(folder / 'complete.json')['files_sha256'].items():
            require(sha(folder / name) == digest, 'Arm artifact changed: ' + arm + '/' + name)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    return dict(schema=SCHEMA, status='PASS_ACTUAL_CPU_FINAL_TENSORS', source_sha256=sha(__file__),
                source_root=str(SOURCE_ROOT), protocol_sha256=PROTOCOL_SHA,
                complete_sha256=done_sha, metadata=metadata_receipt, arms=results,
                original_O=dict(checkpoint_sha256=O_SHA, historical_head_sha256=O_HISTORICAL_HEAD_SHA, head=o_info),
                fresh_initialization=dict(seed=11, construction_order=['motion_readout', 'conditioner'],
                    same_seed11_tensor_initialization_recreated_on_CPU=True,
                    runtime_reset_and_fresh_optimizer_proof='frozen producer code and authenticated logs/preflight; not derivable from final tensors alone'),
                frozen_components=dict(D_receipt=d_receipt, non_head_O_state_sha256=manifest['frozen_non_head_state_sha256'],
                    D_tensors_present_in_final=False, non_head_O_tensors_present_in_final=False,
                    D_tensors_loaded_by_this_audit=False, non_head_O_tensors_loaded_by_this_audit=False,
                    scope='source/initial/final runtime receipts cross-linked only; this audit cannot independently prove absent frozen tensors unchanged'),
                learning_rates='actual final optimizer LR equals authenticated paired last-update logs; no cross-platform libm schedule re-evaluation',
                torch_version=str(torch.__version__), python_version=sys.version, device='cpu',
                forward_calls=0, optimizer_restored=False, optimizer_updates=0, cuda_initialized=False,
                performance_recomputed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-run', 'complete-sha256', 'o-checkpoint', 'out'):
        parser.add_argument('--' + name, required=True)
    defaults = dict(source_root=HERE, protocol=HERE / 'object_state_forecast_training_protocol_v1.json',
                    selection=PARENT / 'selection_v1.json',
                    o_reference=PARENT / 'server_results/campaign_objective_v1/runs/O/development_records.jsonl',
                    sparse_manifest=HERE / 'sparse_motion_v1/manifest.json', raw_manifest=HERE / 'motion_targets_v1/manifest.json',
                    connected_training_run=HERE / 'server_results/training/connected_motion_train_v2',
                    engineering_preflight=HERE / 'object_state_forecast_preflight_v2',
                    preflight_protocol=HERE / 'object_state_forecast_preflight_protocol_v2.json')
    for name, value in defaults.items():
        parser.add_argument('--' + name.replace('_', '-'), default=str(value))
    a = parser.parse_args(argv); output = Path(a.out).resolve()
    # The only writes are new audit artifacts, never checkpoints or source receipts.
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    try:
        result = verify(a); result['elapsed_seconds'] = time.monotonic() - started
        (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        report = ('V/G 最终 checkpoint CPU 实载核验通过。每臂原 future head、新 motion readout 与 conditioner 的全部张量、'
                  '实际 AdamW moments/逐参数 step、固定终点与来源绑定均核对。step 按真实日志中非 None 梯度计数，零梯度仍计更新。\n\n'
                  '原 O head 独立从固定 checkpoint 校验；D 与 O 非 head 未保存在本轮 final.pth，仅追溯原来源和运行时收据。'
                  '这次 CPU 核验不能独立证明缺失组件的张量未变，也不能仅凭最终权重证明全过程 RNG/重置行为。'
                  '未执行 forward、optimizer、GPU 或成绩重算。\n')
        (output / 'report.md').write_text(report)
        complete = dict(schema=SCHEMA, status=result['status'], files_sha256={n: sha(output / n) for n in ('audit.json', 'report.md')})
        (output / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
        print(json.dumps(dict(status=result['status'], output=str(output), audit_sha256=sha(output / 'audit.json'))))
    except Exception as error:
        (output / 'failed.json').write_text(json.dumps(dict(schema=SCHEMA, status='FAIL', error=repr(error),
            elapsed_seconds=time.monotonic() - started), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
