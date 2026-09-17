"""Fixed 512-update, single-seed F or O continuation from the same frozen M0.

F changes only original supervision resolution; O changes only which original
loss terms contribute gradients. The completed native1 C is audited and reused,
never used as candidate initialization. The frozen native optimizer-to-final-eval
AST is executed unchanged except for its already reviewed AdamW audit insertion.
No initial development run, best-checkpoint selection, retry or resume is offered.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time

PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
HELPER_SHA = '7539c19e8cd184df9d25f86d6bd1561fb2164420a15924642d62cfc74cb502a3'
AST_HELPER_SHA = '3aef116f11bf1ce04d11d501b5e3a2f4a44b24b75cf26bd8ec5b14bd56b5be94'
FULLRES_SHA = '12698cb2bb071dd6daf79ccc64ba8f506d0a773a676dd57296f8410fe747d0f7'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
ARMS = ('F', 'O')
IDENTITY_KEYS = ('sample_token', 'scene_token', 'official_index', 'split')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    data = Path(path).read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Hash mismatch: '+str(path))
    return json.loads(data)


def load_contracts(objective_path, parent_path, verify_runtime=True):
    """Shared read-only source/protocol gate; does not load torch or model data."""
    parent, protocol = read(parent_path, PARENT_SHA), read(objective_path)
    require(protocol['schema'] == 'objective-supervision-training-protocol-v1' and
            protocol['status'] == 'FROZEN_BEFORE_F_O_TRAINING' and protocol['arms'] == list(ARMS),
            'Require the frozen two independent objective/supervision arms')
    require(protocol['parent_protocol_sha256'] == PARENT_SHA, 'Wrong parent protocol')
    for key in ('training', 'numerical_policy', 'config_sha256', 'm0_sha256', 'selection_sha256'):
        require(protocol[key] == parent[key], 'Candidate/C contract differs: '+key)
    hp = parent['training']
    expected = dict(seed=11, train_samples=512, development_samples=200, passes=4,
        accumulate=4, lr=1e-5, weight_decay=.01, grad_clip=35, warmup_updates=50, scope='whole_future_pred_head')
    require(all(hp[k] == v for k, v in expected.items()), 'Unexpected fixed native training budget')
    require(parent['numerical_policy']['future_matmul_tf32'] is False and
            parent['numerical_policy']['future_cudnn_tf32'] is True, 'Wrong future-head precision')
    required = {'objective_supervision_train.py', 'objective_supervision_adapters.py',
        'objective_supervision_preflight.py', 'objective_supervision_aggregate.py',
        'run_objective_supervision_campaign.py', 'full_resolution_supervision_preflight.py',
        'frame_train.py', 'frame_consistent_adapter.py', 'frame_aggregate.py'}
    require(required <= set(protocol['source_sha256']), 'Incomplete objective source bindings')
    require(protocol['source_sha256']['frame_train.py'] == HELPER_SHA and
            protocol['source_sha256']['frame_consistent_adapter.py'] == AST_HELPER_SHA and
            protocol['source_sha256']['full_resolution_supervision_preflight.py'] == FULLRES_SHA,
            'Reviewed control/AST/full-resolution helper revision differs')
    sources = dict(parent['source_sha256'])
    for name, digest in protocol['source_sha256'].items():
        require(name not in sources or sources[name] == digest, 'Conflicting source binding: '+name)
        sources[name] = digest
    pkg = Path(__file__).resolve().parent
    for name, digest in sources.items():
        require(Path(name).name == name and sha(pkg/name) == digest, 'Frozen source changed: '+name)
    require(parent['runtime_source_sha256'].items() <= protocol['runtime_source_sha256'].items(),
            'All original runtime/loss sources must remain bound')
    if verify_runtime:
        for filename, digest in protocol['runtime_source_sha256'].items():
            require(Path(filename).is_absolute() and sha(filename) == digest, 'Runtime source changed: '+filename)
    spec = importlib.util.spec_from_file_location('_frozen_objective_control_helper', pkg/'frame_train.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return parent, protocol, helper


def validate_engineering(protocol, objective_path, parent):
    """Receipt gates for the no-optimizer loss probe and real train-mode probe."""
    evidence = protocol['engineering_evidence']
    require(set(evidence) == {'full_resolution_vjp', 'objective_trainmode'}, 'Both real engineering proofs are required')
    result = {}
    for name, binding in evidence.items():
        path = Path(binding['directory'])
        if not path.is_absolute():
            path = Path(objective_path).resolve().parent/path
        require(not (path/'failed.json').exists(), 'Failed engineering receipt: '+name)
        complete = read(path/'complete.json', binding['complete_sha256'])
        manifest = read(path/'manifest.json', binding['manifest_sha256'])
        summary = read(path/'summary.json', binding['summary_sha256'])
        require(complete['manifest_sha256'] == binding['manifest_sha256'] and
                complete['summary_sha256'] == binding['summary_sha256'] and
                summary['manifest_sha256'] == binding['manifest_sha256'], 'Broken engineering receipt chain: '+name)
        source_sha = (manifest['script_sha256'] if name == 'full_resolution_vjp' else
                      manifest['source_sha256'][binding['source']])
        checkpoint_sha = manifest['m0_checkpoint_sha256'] if name == 'full_resolution_vjp' else manifest['m0_sha256']
        require(binding['source'] in protocol['source_sha256'] and
                source_sha == protocol['source_sha256'][binding['source']], 'Probe source differs: '+name)
        require(manifest['parent_protocol_sha256'] == PARENT_SHA and
                checkpoint_sha == parent['m0_sha256'] and
                manifest['config_sha256'] == parent['config_sha256'] and
                manifest['train_cache_index_sha256'] == protocol['cache_index_sha256']['train'] and
                manifest['seed'] == 11 and manifest['precision'] == parent['numerical_policy'],
                'Engineering source/cache/seed/precision differs: '+name)
        if name == 'full_resolution_vjp':
            require(complete['status'] == summary['status'] == 'PASS_NO_OPTIMIZER_PREFLIGHT' and
                    complete['optimizer_updates'] == summary['optimizer_updates'] == 0 and
                    summary['paired_prediction_digest_and_original_fullGT_histograms_exact'] is True and
                    summary['full_GT_not_downsampled'] is True and
                    summary['unchanged_initial_head_sha256'] == INITIAL_HEAD_SHA,
                    'Original full-resolution gradient proof is not a PASS')
        else:
            validate_trainmode_summary(summary, manifest, complete, protocol)
        result[name] = dict(directory=str(path.resolve()), complete_sha256=binding['complete_sha256'],
            manifest_sha256=binding['manifest_sha256'], summary_sha256=binding['summary_sha256'], status=complete['status'])
    return result


def validate_trainmode_summary(summary, manifest, complete, protocol):
    # The producer interface is shared with objective_supervision_preflight.py.
    require(complete['status'] == summary['status'] == 'PASS_OBJECTIVE_SUPERVISION_ENGINEERING_PREFLIGHT',
            'Real objective train-mode preflight has not passed')
    require(summary['checkpoint_saved'] is False and summary['automatic_formal_training'] is False and
            summary['development_data_read'] is False and complete['performance_claim'] is False,
            'Engineering weights must be discarded without automatic continuation')
    for name, digest in manifest['source_sha256'].items():
        require(protocol['source_sha256'][name] == digest, 'Engineering source differs: '+name)
    require(manifest['arms'] == list(ARMS) and manifest['updates_per_arm'] == complete['updates_per_arm'] == 4 and
            manifest['examples_per_arm'] == complete['examples_per_arm'] == 16 and manifest['accumulation'] == 4,
            'Wrong engineering training exposure')
    reports = summary['reports']
    require([r['arm'] for r in reports] == list(ARMS), 'Both isolated objective arms must pass')
    for row in reports:
        arm = row['arm']
        require(row['arm'] == arm and row['initial_head_sha256'] == INITIAL_HEAD_SHA and
                row['trainable_parameters'] == 13274016 and row['actual_updates'] == 4 and row['examples'] == 16,
                'Wrong real objective preflight initialization/budget: '+arm)
        require(row['initial_prediction_and_fullGT_hist_exact'] is True and row['all_gradients_finite'] is True and
                row['initial_parity'] == summary['native_fp32_initial'],
                'Initial prediction parity or complete train-mode gradient gate missing: '+arm)
        require(row['adapter']['arm'] == arm and row['adapter']['adapter_sha256'] ==
                protocol['source_sha256']['objective_supervision_adapters.py'], 'Wrong engineering adapter')
        require(math.isfinite(row['parameter_max_abs_change']) and row['parameter_max_abs_change'] > 0,
                'No real train-mode parameter change')
        require([r['update'] for r in row['optimizer_updates']] == [1,2,3,4] and
                [r['examples_cumulative'] for r in row['optimizer_updates']] == [4,8,12,16],
                'Incomplete real optimizer-update sequence')


def validate_inputs(a):
    parent, protocol, helper = load_contracts(a.objective_protocol, a.protocol)
    require(sha(a.config) == parent['config_sha256'] and sha(a.checkpoint) == parent['m0_sha256'], 'Wrong initial config/M0')
    cap = protocol['resource_policy']['arms'][a.arm]
    require(0 < a.max_seconds <= cap['max_seconds'] and type(a.max_seconds) is int, 'Deadline exceeds frozen arm cap')
    require(math.isfinite(cap['max_allocated_gib']) and cap['max_allocated_gib'] > 0, 'Invalid allocator cap')
    selection = read(Path(a.protocol).with_name('selection_v1.json'), parent['selection_sha256'])['records']
    from memory_experiment import cached_records
    records = {}
    for role, cache in [('train', a.train_cache), ('development', a.dev_cache)]:
        index = read(Path(cache)/'index.json', protocol['cache_index_sha256'][role])
        rows = cached_records(cache); expected = [r for r in selection if r['split'] == role]
        require([[r[k] for k in IDENTITY_KEYS] for r in rows] == [[r[k] for k in IDENTITY_KEYS] for r in expected],
                'Exact cached sample identity/order differs: '+role)
        require(index['split'] == role and index['config_sha256'] == parent['config_sha256'] and
                index['selection_sha256'] == parent['selection_sha256'] and
                index['extractor_sha256'] == parent['source_sha256']['native_state_cache.py'], 'Cache provenance differs')
        require(index['future_occupancy_is_input'] is False and index['inputs_targets_separate'] is True and
                index['native_model']['radar_contract'] == 'native_loader_B_without_common_source_override',
                'Native input/target contract changed')
        require(all(r['parity']['bitwise_all_five_horizons_three_layers'] and
                    r['parity']['exact_native_confusion'] for r in rows), 'Incomplete original cache parity')
        records[role] = rows
    require(len(records['train']) == 512 and len(records['development']) == 200, 'Wrong sample counts')
    require(not {r['scene_token'] for r in records['train']} &
            {r['scene_token'] for r in records['development']}, 'Training/development scenes overlap')
    evidence = validate_engineering(protocol, a.objective_protocol, parent)
    return parent, protocol, helper, records, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('objective-protocol', 'protocol', 'repo', 'config', 'checkpoint', 'train-cache', 'dev-cache', 'control-run', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    a = parser.parse_args(argv); out = Path(a.out).resolve()
    require(not out.exists() and a.max_seconds > 0, 'Require a new output and positive deadline')
    started = time.monotonic(); context = {}
    def stop(*_):
        raise TimeoutError('Fixed objective-training deadline or termination; no retry/resume')
    old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.alarm(a.max_seconds)
    from memory_experiment import write, event, seed_all, state_digest
    try:
        parent, protocol, helper, records, evidence = validate_inputs(a)
        control_manifest, control_payload, control_receipt = helper.validate_control(a, parent, protocol, records)
        block, block_receipt = helper.native_training_block(parent)
        block_receipt.pop('full_task_losses_unchanged')
        block_receipt.update(native_sum_accumulation_backward_unchanged=True,
            task_loss_intervention=a.arm, frozen_control_helper_sha256=HELPER_SHA)
        sys.path.insert(0, str(Path(a.repo).resolve()))
        import numpy as np
        import torch
        from memory_experiment import prepare_model
        from native_state_cache import load_sample, replay
        from objective_supervision_adapters import install_objective_supervision
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        cap_bytes = int(protocol['resource_policy']['arms'][a.arm]['max_allocated_gib']*2**30)
        require(0 < cap_bytes <= total_memory, 'Frozen allocator cap exceeds visible GPU capacity')
        torch.cuda.set_per_process_memory_fraction(cap_bytes/total_memory, device=device)
        seed_all(11); model, migration, params = prepare_model(a.config, a.checkpoint, 'native1')
        require(migration['mode'] == 'native1' and model.memory_queue_len == 1 and
                model.future_pred_head.memory_queue_len == 1, 'Only original single-slot geometry is allowed')
        objective_receipt = install_objective_supervision(model, arm=a.arm)
        initial = state_digest(model.future_pred_head)
        require(initial == INITIAL_HEAD_SHA == control_manifest['initial_head_sha256'], 'Candidate initialization differs from C')
        require(list(params) == control_manifest['trainable_parameter_names'] and
                sum(p.numel() for p in params.values()) == control_manifest['trainable_parameters'] == 13274016,
                'Trainable scope/order/capacity differs from C')
        native_state, control_state = model.future_pred_head.state_dict(), control_payload['future_pred_head']
        require(set(native_state) == set(control_state), 'Candidate/C state keys differ')
        for name, value in native_state.items():
            require(value.shape == control_state[name].shape and value.dtype == control_state[name].dtype and
                    torch.isfinite(control_state[name]).all().item(), 'Candidate/C shape/dtype or final C finiteness differs')
        control_optimizer_groups = copy.deepcopy(control_payload['optimizer']['param_groups'])
        del control_payload, native_state, control_state
        out.mkdir(parents=True, exist_ok=False); write(out/'control_reuse.json', control_receipt)
        hp = parent['training']; train_rows, dev_rows = records['train'], records['development']
        metadata = dict(arm=a.arm, seed=11, initial_head_sha256=initial,
            protocol_sha256=sha(a.objective_protocol), parent_protocol_sha256=PARENT_SHA,
            m0_sha256=parent['m0_sha256'], migration=migration, objective_adapter=objective_receipt,
            train_index_sha256=protocol['cache_index_sha256']['train'], dev_index_sha256=protocol['cache_index_sha256']['development'],
            source_sha256=protocol['source_sha256'], parent_source_sha256=parent['source_sha256'],
            runtime_source_sha256=protocol['runtime_source_sha256'], numerical_policy=parent['numerical_policy'],
            trainable_parameters=sum(p.numel() for p in params.values()), trainable_parameter_names=list(params),
            examples_per_pass=512, passes=4, augmentation='none; identical deterministic native t0 cache in all arms',
            engineering_evidence=evidence, control_reuse=control_receipt, training_block=block_receipt,
            reused_frozen_precision_reference=control_manifest['frozen_precision_reference'],
            initial_development_repeated=False, initialization_parity_proof='fixed real engineering prediction/hist exact gate',
            objective_description=('full native GT; original CE, sem, geo, Lovasz across three layers' if a.arm == 'F' else
                'original coarse GT; only original CE[1,5] and Lovasz across three layers contribute gradients'),
            loss_log_definition=('sum of original 12 full-resolution terms' if a.arm == 'F' else
                'sum of original CE and Lovasz 6 coarse-resolution terms; omitted sem/geo still computed for finite audit'),
            loss_log_comparable_across_objectives=False,
            checkpoint_selection='fixed final 512 updates; no best/dev/threshold selection',
            resource_allocator_cap_bytes=cap_bytes, full_validation_automatically_requested=False)
        write(out/'manifest.json', metadata)
        seed_all(11)  # Same C RNG boundary immediately before fresh AdamW; no initial-dev RNG is retained.
        import memory_experiment as native
        context = dict(native.__dict__)
        context.update(a=a, hp=hp, out=out, model=model, migration=migration, params=params, metadata=metadata,
            train_rows=train_rows, dev_rows=dev_rows, frozen_sha=parent['m0_sha256'], np=np, torch=torch,
            load_sample=load_sample, replay=replay, control_optimizer_groups=control_optimizer_groups,
            _ft_verify_optimizer=helper.verify_optimizer)
        exec(block, context)
        complete = read(out/'complete.json')
        require(complete['updates'] == 512 and complete['examples'] == 2048, 'Incomplete fixed objective endpoint')
        complete.update(objective_protocol_sha256=sha(a.objective_protocol), parent_protocol_sha256=PARENT_SHA,
            manifest_sha256=sha(out/'manifest.json'), control_reuse_sha256=sha(out/'control_reuse.json'),
            training_log_sha256=sha(out/'training.jsonl'), final_head_state_sha256=state_digest(model.future_pred_head),
            total_process_seconds=time.monotonic()-started, new_training_seeds=1, training_seed=11,
            initial_development_repeated=False, automatic_followup=False)
        write(out/'complete.json', complete)
        event('OBJECTIVE_TRAIN_COMPLETE', arm=a.arm, out=str(out), updates=512, examples=2048)
    except BaseException as exc:
        signal.alarm(0)
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if out.is_dir():
            emergency = None
            if 'model' in context and 'optimizer' in context:
                try:
                    import torch
                    torch.save(dict(arm=a.arm, status='INTERRUPTED_NOT_FINAL_NOT_RESUMABLE', metadata=context.get('metadata'),
                        updates=context.get('update', 0), examples=context.get('example_count', 0),
                        future_pred_head={k: v.detach().cpu() for k, v in context['model'].future_pred_head.state_dict().items()},
                        optimizer=context['optimizer'].state_dict(), sample_orders=context.get('all_order', [])), out/'interrupted.pth')
                    emergency = sha(out/'interrupted.pth')
                except BaseException as error:
                    emergency = 'CHECKPOINT_SAVE_FAILED: '+repr(error)
            write(out/'failed.json', dict(status='FAILED_NO_RETRY', error=repr(exc), seconds=time.monotonic()-started,
                completed_updates=context.get('update', 0), examples=context.get('example_count', 0),
                interrupted_checkpoint=emergency, automatic_retry=False, full_validation_requested=False))
        raise
    finally:
        signal.alarm(0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
