"""Read-only O / final A / final Z common occupancy-change evaluation.

No training, oracle flow, threshold fitting, or model selection. Each sample's
complete predictions are made from authenticated inputs ONLY. Targets and raw
seven-frame box contents are first loaded afterwards, for scoring. The native
evaluator and original fine-resolution argmax are unchanged. CUDA scatter_add
in A/Z retains the training implementation's possible reduction nondeterminism.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np

SCHEMA = 'common-change-evaluation-v1'
PROTOCOL_SHA = '0752e433d9f37dbd88406099bee4ca4d3af5298b70dafb82d4cf308ec2df30b3'
CPU_SHA = '0ca592ca0b41ecf3524a0e49d8414e35518f4f9e50ccbb50937af8fded666951'
GEOMETRY_SHA = 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31'
RUNTIME_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
LABEL_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
TRAIN_REFERENCE_SHA = '4685394ec5d551a831252a5b1a35d951bc5fd88026531fac63b75f862715d76b'
DEV_REFERENCE_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
O_HEAD_SHA = '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
IDENTITY = ('sample_token', 'scene_token', 'split', 'official_index')
FINE_SHAPE = (512, 512, 40)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def source_path(name):
    require(Path(name).name == name, 'Source must be basename')
    for path in (Path(__file__).parent/name,
                 Path(__file__).parent.parent/'m0_improvement_20260915'/name):
        if path.is_file():
            return path
    raise FileNotFoundError(name)


def import_bound(name, digest):
    path = source_path(name+'.py')
    require(sha(path) == digest, 'Import source changed: '+name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identity(record):
    return {k: record[k] for k in IDENTITY}


def input_only(native, cache, record, device):
    """Exactly the input half of native.load_sample, without reading targets."""
    import torch
    folder = native._sample_directory(cache, record)
    descriptor = record['files']['inputs']; path = folder/descriptor['file']
    native._verify_file(path, descriptor)
    inputs = torch.load(str(path), map_location='cpu', weights_only=False)
    native.validate_inputs(inputs)
    inputs = native._tree_map(inputs, lambda x: x.to(device=device))
    state = inputs['prev_bev_input']
    require(state.dtype == torch.float32 and tuple(state.shape) == (1,1,40000,256)
            and native._tensor_digest(state) == record['reference_bev_tensor_sha256'],
            'Measured t0 BEV differs')
    return dict(inputs=inputs, record=copy.deepcopy(record))



def historical_O_head_digest(module):
    """Exact memory_experiment.state_digest format used by O's model card.

    The motion helper hashes JSON list shapes; the original O identity hashes
    Python tuple shapes. These digests have different domains despite equal
    tensor bytes. Keep the original O identity without replacing its value.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_targets_after_predictions(native, cache, record, sample, device):
    import torch
    folder = native._sample_directory(cache, record)
    descriptor = record['files']['targets']; path = folder/descriptor['file']
    native._verify_file(path, descriptor)
    target = np.load(path, allow_pickle=False)
    require(target.dtype == np.uint8 and tuple(target.shape) == (1,7,*FINE_SHAPE)
            and list(target.shape) == descriptor['shape'], 'Native target layout differs')
    require(np.all((target == 0) | (target == 1) | (target == 255)), 'GT class mapping differs')
    sample['targets'] = torch.from_numpy(target).to(device=device, dtype=torch.long)
    return target[0,2:]


def fine_binary(prediction, oracle, check):
    import torch
    logits = oracle.predictions_to_xyz(prediction)
    require(tuple(logits.shape) == (5,2,200,200,16) and torch.isfinite(logits).all().item(),
            'Nonfinite/wrong native predictions')
    output = np.empty((5,*FINE_SHAPE), dtype=bool)
    for h in range(5):
        check()
        fine = torch.nn.functional.interpolate(logits[h:h+1], size=FINE_SHAPE,
            mode='trilinear', align_corners=False)[0].argmax(0)
        output[h] = (fine == 1).cpu().numpy()
        del fine
    return output


def native_hist(model, prediction, sample, record):
    rows = model.evaluate_occ_records(prediction, sample['targets'], sample['inputs']['img_metas'])
    require(len(rows) == 1 and all(rows[0][k] == record[k] for k in ('sample_token','scene_token')),
            'Native evaluator identity differs')
    hist = np.asarray(rows[0]['hist_by_horizon'], dtype=np.int64)
    require(hist.shape == (5,2,2) and np.all(hist >= 0), 'Invalid native confusion')
    return hist.tolist()


def validate_completed_fusion(root, expected_complete_sha, protocol, protocol_sha, trainer, helper, train):
    """Actual final receipts only; no placeholder SHA or preflight checkpoint."""
    root = Path(root).resolve()
    require(not (root/'failed.json').exists(), 'Fusion run failed')
    require(sha(root/'complete.json') == expected_complete_sha, 'Unapproved fusion completion')
    done = read(root/'complete.json'); manifest = read(root/'manifest.json')
    require(done['schema'] == trainer.SCHEMA and done['mode'] == 'train'
            and done['status'] == 'COMPLETE_SUPPORTED_FUSION_TRAINING'
            and done['updates'] == 512 and done['examples'] == 2048
            and done['evaluated_samples'] == 200, 'Require fully completed final training and dev200')
    require(set(done['files_sha256']) == {'manifest.json','summary.json','development_records.jsonl'}
            and set(done['arm_complete_sha256']) == set(done['final_checkpoints']) == {'A','Z'},
            'Fusion completion file set differs')
    for name, digest in done['files_sha256'].items():
        require(sha(root/name) == digest, 'Fusion artifact changed: '+name)
    require(manifest['schema'] == trainer.SCHEMA and manifest['mode'] == 'train'
            and manifest['training'] == trainer.TRAINING and manifest['numerical_policy'] == protocol['numerical_policy']
            and manifest['sources']['protocol_sha256'] == protocol_sha
            and manifest['sources']['sources_sha256'] == protocol['sources_sha256']
            and manifest['sources']['motion'] == protocol['motion']
            and manifest['arms'] == ['A','Z'] and manifest['planned_updates'] == 512
            and manifest['same_initial_parameters'] and manifest['independent_optimizers']
            and manifest['boxes_or_motion_labels_read'] is False, 'Fusion manifest contract differs')
    orders = helper.planned_orders(); flat_order = [j for order in orders for j in order]
    require(manifest['sample_orders_sha256'] == helper.json_hash(orders), 'Training plan differs')
    summary = read(root/'summary.json')
    require(summary['mode'] == 'train' and summary['updates'] == 512 and summary['examples'] == 2048
            and summary['evaluated_samples'] == 200 and summary['weights_persisted'] is True
            and summary['all_gradients_finite'] and summary['O_and_motion_parameters_buffers_unchanged'],
            'Fusion training summary incomplete')
    arms = {}
    for arm in ('A','Z'):
        folder = root/'runs'/arm
        require(not (folder/'failed.json').exists(), 'Arm failed')
        require(sha(folder/'complete.json') == done['arm_complete_sha256'][arm], 'Arm completion changed')
        complete = read(folder/'complete.json'); arm_manifest = read(folder/'manifest.json')
        require(complete['schema'] == trainer.SCHEMA and complete['status'] == done['status']
                and complete['arm'] == arm and complete['mode'] == 'train'
                and complete['updates'] == 512 and complete['examples'] == 2048, 'Wrong/incomplete arm')
        require(set(complete['files_sha256']) == {'manifest.json','training.jsonl','final.pth','development_records.jsonl'},
                'Arm files differ')
        for name, digest in complete['files_sha256'].items():
            require(sha(folder/name) == digest, 'Arm artifact changed: '+arm+'/'+name)
        require(arm_manifest == dict(manifest, arm=arm, transport='predicted' if arm == 'A' else 'zero',
                common_manifest_sha256=sha(root/'manifest.json')), 'Arm manifest differs from common recipe')
        final = done['final_checkpoints'][arm]
        require(final['file'] == 'runs/'+arm+'/final.pth'
                and final['sha256'] == complete['files_sha256']['final.pth']
                and final['gate_state_sha256'] == complete['final_gate_sha256'] == summary['final_gate_sha256'][arm],
                'Final checkpoint receipt mismatch')
        logs = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        require(len(logs) == 512, 'Incomplete actual training ledger')
        for u, log in enumerate(logs):
            require(log['update'] == u+1 and log['examples'] == 4*(u+1)
                    and log['pass_index'] == u//128 and log['lr'] == helper.schedule_lr(u)
                    and np.isfinite(log['preclip_grad_norm']) and log['preclip_grad_norm'] >= 0,
                    'Actual training step/LR/gradient ledger differs')
            require(len(log['samples']) == 4, 'Actual accumulate group differs')
            for item, ordinal in zip(log['samples'], flat_order[4*u:4*u+4]):
                record = train[ordinal]
                require(item['ordinal'] == ordinal and item['sample_token'] == record['sample_token']
                        and item['scene_token'] == record['scene_token']
                        and item['inputs_sha256'] == record['files']['inputs']['sha256']
                        and item['targets_sha256'] == record['files']['targets']['sha256'], 'Actual training order/input differs')
            require(log['sample_group_sha256'] == helper.json_hash([r['sample_token'] for r in log['samples']]),
                    'Actual group identity digest differs')
        arms[arm] = dict(folder=folder, complete=complete, manifest=arm_manifest,
                        checkpoint=folder/'final.pth', final=final)
    return root, done, manifest, arms


def load_gates(arms, common_manifest, helper, fusion, device):
    import torch
    gates = {}; loaded = {}
    for arm, binding in arms.items():
        payload = torch.load(str(binding['checkpoint']), map_location='cpu', weights_only=False)
        require(payload['schema'] == common_manifest['schema'] and payload['mode'] == 'train'
                and payload['arm'] == arm and payload['status'] == 'FIXED_FINAL_512'
                and payload['update'] == payload['matched_updates_completed'] == 512
                and payload['examples'] == 2048 and payload['incomplete_accumulation_examples'] == 0
                and payload['resume_supported'] is False, 'Wrong actual final gate payload')
        require(payload['sample_orders'] == helper.planned_orders()
                and payload['sample_orders_sha256'] == common_manifest['sample_orders_sha256']
                and payload['sources'] == common_manifest['sources']
                and payload['manifest_sha256'] == sha(binding['folder']/'manifest.json')
                and payload['initial_gate_sha256'] == common_manifest['initial_gate_sha256'], 'Final gate source/initial/order mismatch')
        # Authenticate the actual initialization too, with the original seed.
        random.seed(11); np.random.seed(11); torch.manual_seed(11)
        gate = fusion.SupportedMotionFusion()
        require(helper.state_digest(gate) == common_manifest['initial_gate_sha256'], 'Fresh seed11 gate initialization differs')
        gate.load_state_dict(payload['fusion'], strict=True)
        actual = helper.state_digest(gate)
        require(actual == payload['final_gate_sha256'] == binding['final']['gate_state_sha256'], 'Actual gate tensor hash differs')
        require(sum(p.numel() for p in gate.parameters()) == 97
                and [n for n,_ in gate.named_parameters()] == common_manifest['trainable_names'], 'Gate architecture differs')
        for parameter in gate.parameters(): parameter.requires_grad_(False)
        gates[arm] = gate.to(device).eval()
        loaded[arm] = dict(checkpoint_sha256=sha(binding['checkpoint']), actual_gate_state_sha256=actual,
            manifest_sha256=payload['manifest_sha256'], fixed_final_update=512,
            optimizer_restored=False, optimizer_updates_in_this_evaluation=0)
        del payload
    return gates, loaded


def summarize(rows, names, metric):
    summary = {}
    for name in names:
        horizons = []; transitions = []
        for h in range(5):
            pieces = [r['metrics_by_arm'][name]['horizons'][h] for r in rows]
            hist = np.sum([p['occupancy']['confusion'] for p in pieces], axis=0, dtype=np.int64)
            item = dict(horizon_index=h, nominal_seconds=h*.5,
                occupancy=metric.confusion_statistics(hist, ('0','1')),
                global_negative={k:sum(p['global_negative'][k] for p in pieces) for k in ('GT','FP','TN')})
            if h:
                attrs = [p['motion_positive_attribution'] for p in pieces]
                groups = {}
                for group in metric.POSITIVE_GROUPS:
                    totals = {key:sum(a['groups'][group][key] for a in attrs) for key in ('GT','TP','FN')}
                    totals['recall'] = totals['TP']/totals['GT'] if totals['GT'] else None
                    groups[group] = totals
                item['motion_positive_groups'] = groups
                item['GT_association_counts'] = {key:sum(a[key] for a in attrs) for key in
                    ('foreground_voxels','unique_box_voxels','unknown_voxels','overlap_voxels',
                     'union_instance_count','active_instance_count','missing_current_instance_count',
                     'active_instances_without_unique_GT_support')}
            horizons.append(item)
        for j in range(4):
            pieces = [r['metrics_by_arm'][name]['transitions'][j] for r in rows]
            hist = np.sum([p['confusion'] for p in pieces], axis=0, dtype=np.int64)
            transitions.append(dict(horizon_index=j+1, nominal_seconds=(j+1)*.5,
                **metric.confusion_statistics(hist, metric.TRANSITION_NAMES),
                domain={key:sum(p['domain'][key] for p in pieces) for key in pieces[0]['domain']}))
        iou = [h['occupancy']['classes']['1']['IoU'] for h in horizons]
        summary[name] = dict(horizons=horizons, transitions=transitions,
            future_macro_GMO_percent=None if any(x is None for x in iou[1:]) else 100*float(np.mean(iou[1:])),
            change_01_10_both_reported=True)
    return summary


def run(a, out, started, stopping):
    require(sha(a.protocol) == PROTOCOL_SHA, 'Require frozen v2 fusion protocol bytes')
    p = read(a.protocol)
    require(p['status'] == 'FROZEN', 'Protocol not frozen')
    for name, digest in p['sources_sha256'].items():
        require(sha(source_path(name)) == digest, 'Frozen source changed: '+name)
    require(sha(a.runtime_contract) == RUNTIME_SHA, 'Runtime source contract changed')
    runtime = read(a.runtime_contract)
    for path, digest in runtime['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Runtime source changed: '+path)
    import_bound('motion_geometry', GEOMETRY_SHA)
    metric = import_bound('common_occupancy_change_metrics_v1', CPU_SHA)
    trainer = import_bound('train_supported_fusion_v2', p['sources_sha256']['train_supported_fusion_v2.py'])
    require(p['schema'] == trainer.SCHEMA and p['training'] == trainer.TRAINING, 'Training recipe differs')
    native = trainer.import_source('native_state_cache', p['sources_sha256'])
    oracle = trainer.import_source('oracle_transport_probe', p['sources_sha256'])
    helper = trainer.import_source('train_source_motion_v1', p['sources_sha256'])
    require(sha(a.config) == oracle.CONFIG_SHA and sha(a.checkpoint) == oracle.M0_SHA
            and sha(a.o_checkpoint) == oracle.O_SHA, 'Fixed O/M0/config changed')
    train_root, train, train_receipt = helper.cache_index(a.train_cache, 'train', p)
    names = ['O']; fusion_done = fusion_manifest = arm_bindings = None
    motion_file = motion_manifest = tp = pretrained = fusion = None
    if a.mode == 'o_train16_qa':
        cache, selected, cache_receipt = train_root, train[:16], train_receipt
        require(sha(a.oracle_reference_records) == TRAIN_REFERENCE_SHA, 'Train16 O-only reference changed')
        reference = [json.loads(x) for x in Path(a.oracle_reference_records).read_text().splitlines()]
        require(len(reference) == 16, 'Expected exactly 16 reference records')
        for i, row in enumerate(reference):
            require(set(row) == {'ordinal','sample_token','scene_token','hist_by_arm'}
                    and row['ordinal'] == i and set(row['hist_by_arm']) == {'O'}, 'Not an O-only reference ledger')
        expected_hist = [r['hist_by_arm']['O'] for r in reference]
        reference_sha = TRAIN_REFERENCE_SHA
    else:
        names += ['A','Z']
        cache, selected, cache_receipt = helper.cache_index(a.dev_cache, 'development', p)
        require(not ({r['scene_token'] for r in train} & {r['scene_token'] for r in selected}), 'Train/dev scenes overlap')
        require(sha(a.o_development_records) == DEV_REFERENCE_SHA == p['o_development_records_sha256'], 'O dev reference changed')
        reference = [json.loads(x) for x in Path(a.o_development_records).read_text().splitlines()]
        require(len(reference) == 200, 'Require full fixed dev200 reference')
        expected_hist = [r['hist_by_horizon'] for r in reference]; reference_sha = DEV_REFERENCE_SHA
        # Reuse original validator to authenticate the completed TRAINING
        # contract. These resource fields describe that past training only;
        # this evaluator's independent caps remain a.max_seconds / GiB.
        audit_args = SimpleNamespace(protocol=a.protocol, preflight=False,
            max_seconds=p['resources']['train']['max_seconds'],
            max_allocated_gib=p['resources']['train']['max_allocated_gib'],
            motion_training_protocol=a.motion_training_protocol, motion_run=a.motion_run)
        _, tp, _, motion_file, motion_manifest = trainer.load_contract(audit_args)
        fusion_root, fusion_done, fusion_manifest, arm_bindings = validate_completed_fusion(
            a.fusion_run, a.fusion_complete_sha256, p, PROTOCOL_SHA, trainer, helper, train)
        require(fusion_manifest['sources']['train_cache'] == train_receipt
                and fusion_manifest['sources']['development_cache'] == cache_receipt
                and fusion_manifest['sources']['O_development_records_sha256'] == reference_sha,
                'Fusion training used different cache/reference')
        trainer.import_source('transport_ops', p['sources_sha256'])
        fusion = trainer.import_source('supported_motion_fusion', p['sources_sha256'])
        pretrained = trainer.import_source('learned_transport_probe_v1', p['sources_sha256'])
    for record, old in zip(selected, reference):
        require(all(record[k] == old[k] for k in ('sample_token','scene_token')), 'Reference/cache identity order differs')
    require(len({r['scene_token'] for r in selected}) == (8 if len(selected)==16 else 100), 'Unexpected scene count')
    labels = Path(a.labels).resolve(); label_complete = read(labels/'complete.json')
    require(sha(labels/'manifest.json') == LABEL_MANIFEST_SHA == label_complete['manifest_sha256']
            and label_complete['schema'] == 'raw-nuscenes-motion-target-complete-v1'
            and label_complete['status'] == 'COMPLETE' and label_complete['samples'] == 712
            and label_complete['optimizer_steps'] == 0 and label_complete['model_or_prediction_read'] is False,
            'Require complete original raw712 labels')
    label_manifest = read(labels/'manifest.json')
    require(label_manifest['train_dev_scenes_disjoint'] and label_manifest['inputs_and_labels_physically_separate'], 'Raw-label isolation differs')
    descriptors = {r['identity']['sample_token']:r for r in label_manifest['records']}
    require(len(descriptors) == len(label_manifest['records']) == 712, 'Raw label count/duplicates')
    descs = []
    for record in selected:
        desc = descriptors[record['sample_token']]
        require(desc['identity'] == identity(record)
                and desc['file'] == record['split']+'/'+record['sample_token']+'.json.gz', 'Raw label identity/path differs')
        # Only descriptor metadata is read now, not raw box contents or GT.
        descs.append(desc)
    manifest = dict(schema=SCHEMA, mode=a.mode, status='RUNNING', arms=names,
        sources_sha256={**p['sources_sha256'], Path(__file__).name:sha(__file__),
            'common_occupancy_change_metrics_v1.py':CPU_SHA, 'motion_geometry.py':GEOMETRY_SHA},
        protocol_sha256=PROTOCOL_SHA, runtime_source_contract_sha256=RUNTIME_SHA,
        runtime_source_sha256=runtime['runtime_source_sha256'],
        sample_cache=cache_receipt, train_cache=train_receipt,
        anchors=[identity(r) for r in selected], raw_labels_manifest_sha256=LABEL_MANIFEST_SHA,
        raw_labels_complete_sha256=sha(labels/'complete.json'), raw_label_descriptors=descs,
        O_reference_sha256=reference_sha, O_checkpoint_sha256=oracle.O_SHA,
        M0_checkpoint_sha256=oracle.M0_SHA, config_sha256=oracle.CONFIG_SHA,
        fusion_complete_sha256=None if fusion_done is None else a.fusion_complete_sha256,
        fixed_final_gate_receipts=None if fusion_done is None else fusion_done['final_checkpoints'],
        numerical_policy=p['numerical_policy'], resources=dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib),
        optimizer_updates=0, seed=11, training=False, oracle_inference=False,
        target_and_raw_box_load_after_all_model_predictions=True,
        prediction_readout='native evaluator XYZ trilinear full512x512x40 align_corners=False then argmax, no GT crop',
        flow_EPE_or_identity_accuracy_claim=False, no_threshold_selection=True,
        exposure='fixed train16 engineering QA' if a.mode=='o_train16_qa' else 'historically exposed development200; not unseen confirmation')
    write(out/'manifest.json',manifest)
    import torch
    require(str(a.device).startswith('cuda'), 'Native O requires CUDA')
    torch.cuda.set_device(a.device); torch.set_num_threads(2)
    random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    total_memory=torch.cuda.get_device_properties(a.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,a.max_allocated_gib*2**30/total_memory),a.device)
    torch.cuda.reset_peak_memory_stats(a.device)
    def check():
        require(not stopping, 'Signal; no retry')
        require(time.monotonic()-started < a.max_seconds, 'Evaluation wall-clock ceiling')
        require(torch.cuda.max_memory_allocated(a.device) <= a.max_allocated_gib*2**30, 'Evaluation allocation ceiling')
    check()
    model=native.build_native_model(a.config,a.checkpoint,device=a.device,repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu',weights_only=False)
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval()
    require(historical_O_head_digest(model.future_pred_head)==O_HEAD_SHA, 'Actual O head differs under original digest format')
    original_model_sha=helper.state_digest(model)
    gates={}; motion=None; loaded_gates={}
    if a.mode=='final_dev200':
        require(original_model_sha==fusion_manifest['frozen_O_state_sha256'], 'Loaded full O state differs from gate training')
        motion=pretrained.load_motion(motion_file,motion_manifest,
            dict(sources_sha256=p['sources_sha256'],training=p['motion']),tp,helper,a.device)
        require(helper.state_digest(motion)==fusion_manifest['frozen_motion_state_sha256']==p['motion']['head_state_sha256'],
                'Loaded motion state differs from gate training')
        gates,loaded_gates=load_gates(arm_bindings,fusion_manifest,helper,fusion,a.device)
    write(out/'loaded_models.json',dict(native=model._native_state_provenance,
        O_head_state_sha256=O_HEAD_SHA,O_head_digest_format='memory_experiment.state_digest: tuple shape',
        O_head_motion_helper_sha256=helper.state_digest(model.future_pred_head),
        O_full_state_sha256=original_model_sha,
        motion_head_state_sha256=None if motion is None else helper.state_digest(motion),
        gates=loaded_gates,torch=torch.__version__,numpy=np.__version__,
        optimizer_restored=False,optimizer_updates=0))
    rows=[]; initialization_seconds=time.monotonic()-started
    with torch.no_grad(), (out/'records.jsonl').open('x') as stream:
        for ordinal,(record,desc) in enumerate(zip(selected,descs)):
            check();tick=time.monotonic()
            sample=input_only(native,cache,record,a.device)
            state=sample['inputs']['prev_bev_input']
            original=native.replay(model,sample,training=False)[0]
            require(tuple(original.shape)==(5,3,1,1,40000,16,2), 'Native output shape differs')
            predictions={'O':original}
            if gates:
                base=oracle.predictions_to_xyz(original); flow=motion(state[:,-1])
                probability=torch.softmax(base[0:1],dim=1)[:,1:2]
                foreground=base[0:1,1:2]>base[0:1,0:1]
                future=(base[1:,1]-base[1:,0])[None,:,None]
                for arm in ('A','Z'):
                    field=fusion.supported_transport(probability,foreground,flow if arm=='A' else torch.zeros_like(flow),oracle.EXTENT)
                    fused=gates[arm](future,field['probability_mass'],field['support_weight'])
                    predictions[arm]=trainer.compose_prediction(original,base,fused,oracle)
                    del field,fused
                del flow,probability,foreground,future,base
            require(native._tensor_digest(state)==record['reference_bev_tensor_sha256'], 'Observed input mutated')
            require(all(trainer.exact32(value[0],original[0]) for value in predictions.values()), 't0 native logits differ')
            torch.cuda.synchronize(a.device); prediction_seconds=time.monotonic()-tick
            # First target/box content read: all O/A/Z predictions already exist.
            gt=load_targets_after_predictions(native,cache,record,sample,a.device)
            label_path=labels/desc['file']
            require(label_path.stat().st_size==desc['bytes'] and sha(label_path)==desc['sha256'], 'Raw label bytes changed')
            label=json.loads(gzip.decompress(label_path.read_bytes()))
            require(label['identity']==identity(record), 'Raw label identity mismatch')
            outputs={};native_histograms={}
            for name in names:
                check()
                hist=native_hist(model,predictions[name],sample,record)
                if name=='O':require(hist==expected_hist[ordinal], 'Frozen O reference full-GT confusion parity failed')
                binary=fine_binary(predictions[name],oracle,check)
                result=metric.evaluate_common_occupancy_change(binary,gt,oracle.EXTENT,label)
                require([h['occupancy']['confusion'] for h in result['horizons']]==hist, 'CPU/fine readout differs from native evaluator')
                if name!='O':
                    require(result['t0_boundary']==outputs['O']['t0_boundary'], 'O/A/Z fine binary t0 boundary differs')
                    require(all([x['occupancy']['classes'][str(c)]['GT'] for x in result['horizons']]==
                        [x['occupancy']['classes'][str(c)]['GT'] for x in outputs['O']['horizons']] for c in (0,1)), 'Different GT support across models')
                outputs[name]=result;native_histograms[name]=hist
                del binary
            torch.cuda.synchronize(a.device);check()
            row=dict(ordinal=ordinal,**identity(record),metrics_by_arm=outputs,hist_by_arm=native_histograms,
                input_sha256=record['files']['inputs']['sha256'],GT_cache_sha256=record['files']['targets']['sha256'],
                raw_label_sha256=desc['sha256'],O_reference_hist_exact=True,native_CPU_hist_exact=True,
                t0_all_models_exact=True,GT_raw_labels_first_loaded_after_all_predictions=True,
                prediction_seconds=prediction_seconds,seconds=time.monotonic()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated(a.device)/2**30)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            rows.append(row)
            del sample,state,original,predictions,gt,label,outputs
            write(out/'progress.json',dict(completed_samples=len(rows),planned_samples=len(selected),
                last_token=record['sample_token'],seconds=time.monotonic()-started,optimizer_updates=0))
            print(json.dumps(dict(event='COMMON_CHANGE_SAMPLE',ordinal=ordinal,sample_token=record['sample_token'],seconds=row['seconds'])),flush=True)
    check()
    require(len(rows)==len(selected), 'Incomplete sample set')
    require(helper.state_digest(model)==original_model_sha, 'Frozen O mutated')
    if motion is not None:
        require(helper.state_digest(motion)==p['motion']['head_state_sha256'], 'Motion weights changed')
        require(all(helper.state_digest(gates[k])==loaded_gates[k]['actual_gate_state_sha256'] for k in gates), 'Gate weights changed')
    summary=dict(schema=SCHEMA,mode=a.mode,status='PASS_QA_ONLY' if a.mode=='o_train16_qa' else 'COMPLETE_FINAL_DEVELOPMENT_EVALUATION',
        samples=len(rows),scenes=len({r['scene_token'] for r in rows}),optimizer_updates=0,arms=names,
        scores=summarize(rows,names,metric),all_O_reference_hist_exact=True,all_native_CPU_hist_exact=True,
        all_t0_models_exact=True,initialization_seconds=initialization_seconds,seconds=time.monotonic()-started,
        max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
        no_flow_EPE_or_identity_claim=True,bootstrap_performed=False,
        description='Occupancy and occupancy-change sufficient statistics; GT-speed positive risks plus one full-negative ledger. Not physical flow or identity accuracy.')
    write(out/'summary.json',summary)
    (out/'report.md').write_text('# Common final-output occupancy-change evaluation\n\n'+
        '模式：'+a.mode+'；样本 '+str(len(rows))+'；零训练。\n\n'+
        'O 原始完整 GT confusion 全部精确重现；共同 CPU readout 与原生 evaluator 全部一致；t0 完全一致。\n\n'+
        '四状态 01/10 均报告，速度分层保留来源缺失/unknown/overlap，全负类 FP 只计一次。\n\n'+
        '这些是共同占据与变化证据，不是 flow EPE、身份能力或未见数据确认。无阈值选择、无bootstrap。\n')
    check()
    write(out/'complete.json',dict(schema=SCHEMA,status=summary['status'],mode=a.mode,
        samples=len(rows),scenes=summary['scenes'],arms=names,optimizer_updates=0,
        source_sha256=sha(__file__),protocol_sha256=PROTOCOL_SHA,
        files_sha256={name:sha(out/name) for name in ('manifest.json','loaded_models.json','records.jsonl','summary.json','report.md')},
        seconds=time.monotonic()-started))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('o_train16_qa','final_dev200'),required=True)
    for name in ('protocol','config','checkpoint','o-checkpoint','repo','runtime-contract','train-cache','labels','out'):
        parser.add_argument('--'+name,required=True)
    for name in ('oracle-reference-records','dev-cache','o-development-records','motion-run',
                 'motion-training-protocol','fusion-run','fusion-complete-sha256'):
        parser.add_argument('--'+name)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,required=True)
    parser.add_argument('--max-allocated-gib',type=float,required=True)
    a=parser.parse_args(argv)
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=(600 if a.mode=='o_train16_qa' else 1800)
            and math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=32, 'Invalid bounded evaluation resources')
    future_args=(a.dev_cache,a.o_development_records,a.motion_run,a.motion_training_protocol,a.fusion_run,a.fusion_complete_sha256)
    if a.mode=='o_train16_qa':
        require(a.oracle_reference_records and not any(future_args), 'QA only permits O/train16 inputs')
    else:
        require(all(future_args) and a.oracle_reference_records is None, 'Final mode requires real complete A/Z and dev200 only')
        require(len(a.fusion_complete_sha256)==64 and all(c in '0123456789abcdef' for c in a.fusion_complete_sha256), 'Require actual final completion SHA')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stopping=[]
    signal.signal(signal.SIGTERM,lambda sig,frame:stopping.append(sig))
    signal.signal(signal.SIGINT,lambda sig,frame:stopping.append(sig))
    try:
        run(a,out,started,stopping)
    except BaseException as exc:
        write(out/'failed.json',dict(schema=SCHEMA,status='FAILED_NO_RETRY',mode=a.mode,
            exception=type(exc).__name__,message=str(exc),optimizer_updates=0,
            seconds=time.monotonic()-started,source_sha256=sha(__file__)))
        raise


if __name__=='__main__':
    main()
