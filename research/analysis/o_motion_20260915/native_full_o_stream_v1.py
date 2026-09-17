"""Fresh native validation -> frozen O replay; no cache construction or dispatch.

NativeOStream(..., ordinals=None).fetch(ordinal) returns a caller-owned frame:
  sample = {inputs, targets, record}; O_prediction [5,3,1,1,40000,16,2];
  t0_bev [1,40000,256]; audit (small JSON only).
Default order is all 5119 official ordinals. A caller may supply an increasing
unique shard/subset schedule; fetch enforces that exact schedule. Call
verify_frame(frame) after downstream use, discard the frame, then fetch next.
No GT field enters future_pred or t0_bev. The ORIGINAL loader nevertheless reads
GT before capture; targets are kept separate, exclusively for evaluation.
CLI performs ONLY the existing two-anchor engineering pilot, not full scoring.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback

SCHEMA = 'native-full-o-stream-v1'
SOURCES = {
    'native_state_cache.py': '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0',
    'joint_native_evaluation.py': '280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578',
    'objective_joint_evaluation_v2.py': 'f8e3da3de084f5cde3d35b9574124d9684b7ecf77387bd03cf5828c02f7d88ca',
    'observation_memory.py': '0f01a29dc79daf2f0abb42bd2a23f8d168606cd3928212229ee5aeb3f94a5733',
    'memory_experiment.py': '42f81d5799c32bf98e6e6da4055d79c9d62f50332d8aed87797e4ea72a7d6f2c',
}
M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
O_SHA = 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
O_HEAD_SHA = '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
CONFIG_SHA = 'c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
FULL_SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
PILOT_SELECTION_SHA = 'e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d'
REFERENCE_COMPLETE_SHA = 'cdf54175b84613b1e175fe783f490ab62681c18060c5289981f2be426d2a53e5'
REFERENCE_SUMMARY_SHA = '199347570b9d34130c6366ef3fb138bbc08c1310a50010a1f200061fe428f463'
DEV_INDEX_SHA = 'fd42d2e511d754755fa5e2c06527077351869f06e868ecb8acb4404eadaa26c3'
O_DEV_SHA = '3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
HORIZONS = [0., .5, 1., 1.5, 2.]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'File SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def dependencies():
    modules = {}
    for name, digest in SOURCES.items():
        candidates = [Path(__file__).with_name(name),
                      Path(__file__).parent.parent / 'm0_improvement_20260915' / name]
        path = next((p for p in candidates if p.is_file()), None)
        require(path is not None and sha(path) == digest, 'Frozen helper missing/changed: ' + name)
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('_native_stream_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.stem] = module
    return modules


def load_full_reference(directory, selection_path):
    """Authenticate existing O arrays/identity ledger; never recompute scores."""
    import numpy as np
    directory = Path(directory)
    done = read(directory / 'complete.json', REFERENCE_COMPLETE_SHA)
    summary = read(directory / 'summary.json', REFERENCE_SUMMARY_SHA)
    require(done['schema'] == 'objective-joint-merge-complete-v2' and done['status'] == 'COMPLETE'
            and done['mode'] == 'full' and done['samples'] == 5119 and done['scenes'] == 150
            and done['summary_sha256'] == REFERENCE_SUMMARY_SHA, 'Incomplete original full reference')
    require(not (directory / 'failed.json').exists(), 'Reference has failure marker')
    verified = {'complete.json': REFERENCE_COMPLETE_SHA}
    for name in ('summary.json', 'raw_confusions.npz', 'sample_hash_ledger.json'):
        require(sha(directory / name) == done['files_sha256'][name], 'Reference artifact changed: ' + name)
        verified[name] = done['files_sha256'][name]
    selection = read(selection_path, FULL_SELECTION_SHA)
    require(selection['schema'] == 'native-joint-validation-selection-v1'
            and selection['samples'] == 5119 and selection['scenes'] == 150
            and selection['config_sha256'] == CONFIG_SHA, 'Full selection scope changed')
    rows = selection['records']
    require(len(rows) == 5119 and [r['official_index'] for r in rows] == list(range(5119))
            and len({r['sample_token'] for r in rows}) == 5119
            and len({r['scene_token'] for r in rows}) == 150, 'Official full order/identity changed')
    ledger = read(directory / 'sample_hash_ledger.json')
    with np.load(directory / 'raw_confusions.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    require(arrays['models'].tolist() == ['M0', 'M0_fp32', 'native1', 'O']
            and arrays['horizon_seconds'].tolist() == HORIZONS, 'Reference model/horizon contract')
    hist = arrays['hist_by_model_sample_horizon']
    counts = arrays['target_counts_0_1_255_by_sample_frame']
    require(hist.shape == (4, 5119, 5, 2, 2) and hist.dtype == np.uint64
            and counts.shape == (5119, 7, 3) and counts.dtype == np.uint64, 'Reference GT/hist shape or dtype')
    require(np.array_equal(hist.sum(-1), np.broadcast_to(counts[None, :, 2:, :2], (4,5119,5,2)))
            and np.all(counts.sum(-1) == 512*512*40), 'Reference GT confusion orientation/counts')
    require(len(ledger) == 5119 and len(summary['ordered_identities']) == 5119, 'Reference identity ledger length')
    for i, row in enumerate(rows):
        identity = ledger[i]['identity']
        require(identity == summary['ordered_identities'][i]
                and all(identity[k] == row[k] for k in ('sample_token','scene_token','official_index'))
                and identity['selection_ordinal'] == i and identity['split'] == 'validation'
                and arrays['sample_tokens'][i] == row['sample_token']
                and arrays['scene_tokens'][i] == row['scene_token']
                and int(arrays['official_indices'][i]) == i, 'Reference row identity/order: ' + str(i))
    require(summary['sources']['selection_sha256'] == FULL_SELECTION_SHA
            and summary['sources']['final_checkpoints_sha256']['O'] == O_SHA
            and summary['sources']['final_head_state_sha256']['O'] == O_HEAD_SHA,
            'Reference O checkpoint/selection source differs')
    return dict(selection=selection, summary=summary, ledger=ledger, hist=hist[3], counts=counts,
                verified_files_sha256=verified)


class NativeOStream:
    """One O model, exact ordered fresh loader, caller-owned one-sample outputs."""
    def __init__(self, *, config, checkpoint, o_checkpoint, repo, runtime_contract,
                 selection, reference_dir, device='cuda:0', ordinals=None,
                 max_allocated_gib=32., check=None):
        self.started = time.monotonic()
        self.check_callback = check or (lambda: None)
        self.modules = dependencies()
        self.native = self.modules['native_state_cache']
        self.joint = self.modules['joint_native_evaluation']
        self.policy = self.modules['objective_joint_evaluation_v2']
        self.state_digest = self.modules['memory_experiment'].state_digest
        self.reference = load_full_reference(reference_dir, selection)
        self.config, self.checkpoint, self.o_checkpoint = config, checkpoint, o_checkpoint
        self.runtime_path = Path(runtime_contract)
        runtime = read(runtime_contract)['runtime_source_sha256']
        require(runtime == self.reference['summary']['sources']['runtime_source_sha256'], 'Runtime source contract differs from original full')
        self.runtime_files = dict(runtime)
        # Previously supplemental source coverage remains explicitly separate.
        self.additional_files = dict(self.reference['summary']['inactive_boundary_policy']['sources'])
        require(len(self.additional_files) == 3, 'Missing inactive-planning supplemental sources')
        for suffix, digest in self.policy.ADDITIONAL_SOURCE_SHA.items():
            matches = [p for p in self.additional_files if p.endswith('/' + suffix)]
            require(len(matches) == 1 and self.additional_files[matches[0]] == digest, 'Supplemental source differs')
        self.verify_sources()
        self.ordinals = list(range(5119)) if ordinals is None else list(ordinals)
        require(self.ordinals and all(type(i) is int and 0 <= i < 5119 for i in self.ordinals)
                and self.ordinals == sorted(set(self.ordinals)), 'Schedule must be increasing unique official ordinals')
        self.position = 0
        self.max_allocated_bytes = int(max_allocated_gib * 2**30)
        require(math.isfinite(max_allocated_gib) and 0 < max_allocated_gib <= 64, 'Invalid memory ceiling')
        self.check_callback()
        import numpy as np
        import torch
        from mmcv import Config
        from mmcv.parallel import MMDataParallel, collate
        from mmdet3d.datasets import build_dataset
        require(str(device).startswith('cuda') and torch.cuda.is_available(), 'Native image forward requires CUDA')
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        torch.set_num_threads(2)
        try:
            import cv2
            cv2.setNumThreads(2)
        except ImportError:
            pass
        random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(min(1., self.max_allocated_bytes / torch.cuda.get_device_properties(self.device).total_memory), self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.native._prepare_repo(repo)
        cfg = Config.fromfile(str(config))
        dc = copy.deepcopy(cfg.data.test); dc.test_mode = True; dc.pop('samples_per_gpu', None)
        require(dc.radar_cfg.cache_readonly is True and dc.radar_cfg.nsweeps == 5, 'Require unchanged readonly five-sweep radar')
        require(dc.get('radar_observation_cfg') is None and not dc.get('allow_dual_radar_inputs')
                and dc.future_metadata_only is True, 'Alternate radar/future data interface')
        require(sha(dc.ann_file) == self.reference['selection']['ann_sha256'], 'Native annotation SHA differs')
        self.dataset = build_dataset(dc)
        require(len(self.dataset.usable_index) == 5119, 'Native usable validation count differs')
        self.rows = self.native._selected_rows(self.reference['selection'], 'validation', self.dataset)
        for i, row in enumerate(self.rows):
            row['selection_ordinal'] = i
            require(row == self.reference['ledger'][i]['identity'], 'Resolved original data_info_index differs')
        self.model = self.native.build_native_model(config, checkpoint, device=device, repo=repo)
        payload = torch.load(str(o_checkpoint), map_location='cpu', weights_only=False)
        rng = np.random.RandomState(11)
        require(payload['arm'] == 'O' and payload['pass_index'] == 4 and payload['update'] == 512
                and payload['sample_orders'] == [rng.permutation(512).tolist() for _ in range(4)], 'O is not fixed final seed11 checkpoint')
        self.model.future_pred_head.load_state_dict(payload['future_pred_head'], strict=True)
        for name, value in self.model.future_pred_head.state_dict().items():
            require(torch.equal(value.detach().cpu(), payload['future_pred_head'][name]), 'Actual O head tensor changed: ' + name)
        del payload
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        require(self.state_digest(self.model.future_pred_head) == O_HEAD_SHA, 'Actual historical tuple-shape O head digest differs')
        self.initial_model_sha = self.state_digest(self.model)
        self.wrapped = MMDataParallel(self.model, device_ids=[self.device_index])
        self.collate = collate
        self.guard()
        self.initialization_seconds = time.monotonic() - self.started
        self.loaded = dict(O_checkpoint_sha256=O_SHA, O_head_state_sha256=O_HEAD_SHA,
            O_full_state_sha256=self.initial_model_sha, state_digest_format='memory_experiment: sorted name/dtype/str(tuple(shape))/bytes',
            native_provenance=self.model._native_state_provenance, torch_version=torch.__version__,
            cuda_version=torch.version.cuda, device=str(self.device), gpu=torch.cuda.get_device_name(self.device),
            model_initialization_seconds=self.initialization_seconds, optimizer_created=False,
            observation_matmul_tf32=True, future_matmul_tf32=False, cudnn_tf32=True, training=False)

    def verify_sources(self):
        for path, digest in [(self.config,CONFIG_SHA),(self.checkpoint,M0_SHA),(self.o_checkpoint,O_SHA)] + list(self.runtime_files.items()) + list(self.additional_files.items()):
            require(sha(path) == digest, 'Source changed: ' + str(path))

    def guard(self):
        self.check_callback()
        import torch
        require(not any(p.requires_grad for p in self.model.parameters()), 'O parameter became trainable')
        # Frozen helper calls its single base slot M0; here that slot holds O.
        audit = self.policy.validate_runtime_consumer(self.model, {}, ['M0'])
        require(torch.cuda.max_memory_allocated(self.device) <= self.max_allocated_bytes, 'Allocation ceiling')
        return {'O': audit['M0']}

    def fetch(self, ordinal):
        import numpy as np
        import torch
        require(self.position < len(self.ordinals) and ordinal == self.ordinals[self.position], 'Fetch must follow fixed official schedule, no retry/replacement')
        self.guard(); started = time.monotonic(); row = self.rows[ordinal]
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        example = self.dataset._prepare_data_info(row['data_info_index'], rand_interval=None)
        require(example is not None, 'Native loader returned no sample; do not replace')
        identity = self.native._identity_from_metas(example['img_metas'].data)
        require(all(identity[k] == row[k] for k in ('sample_token','scene_token')), 'Prepared sample identity differs')
        data = self.collate([example], samples_per_gpu=1); data.pop('instance', None); context = {}
        with self.native._capture_native(self.model, context), torch.no_grad():
            self.wrapped(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize(self.device)
        capture_seconds = time.monotonic() - started
        del data, example
        self.native.validate_inputs(context['inputs'])
        require(context['preds'].shape == (5,3,1,1,40000,16,2) and context['preds'].dtype == np.float32
                and np.isfinite(context['preds']).all(), 'Capture-only prediction contract')
        # This is O/TF32-on, not M0; never compare it to cached M0 predictions.
        del context['preds'], context['native_record']
        gt = context['targets']
        require(gt.shape == (1,7,512,512,40) and gt.dtype == np.uint8, 'Original full segmentation contract')
        counts = np.asarray([[int((t == c).sum()) for c in (0,1,255)] for t in gt[0]], dtype=np.uint64)
        old = self.reference['ledger'][ordinal]
        input_sha = self.joint.tree_digest(context['inputs'])
        gt_sha = self.joint.array_digest(gt)
        bev_sha = self.native._tensor_digest(context['inputs']['prev_bev_input'])
        radar_sha = self.native._tensor_digest(context['inputs']['radar_bev'])
        require(gt_sha == old['targets_array_sha256'] and np.array_equal(counts, self.reference['counts'][ordinal]), 'Fresh full GT differs from original full')
        require(bev_sha == old['observed_bev_tensor_sha256'] and radar_sha == old['native_radar_tensor_sha256'], 'Fresh measured BEV/radar differs from original full')
        sample = dict(inputs=self.native._tree_map(context['inputs'], lambda x: x.to(self.device)),
                      targets=torch.from_numpy(gt).to(self.device, dtype=torch.long), record=row)
        del context
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        tick = time.monotonic()
        with torch.no_grad():
            pred = self.native.replay(self.model, sample, training=False)[0]
            require(tuple(pred.shape) == (5,3,1,1,40000,16,2) and pred.dtype == torch.float32
                    and bool(torch.isfinite(pred).all()), 'O replay prediction contract')
            result = self.model.evaluate_occ_records(pred, sample['targets'], sample['inputs']['img_metas'])
        require(len(result) == 1, 'Native evaluator record count')
        evaluated = self.joint.evaluator_record(result[0], row, counts)
        require(evaluated['hist_by_horizon'] == self.reference['hist'][ordinal].tolist(), 'Original full O 5h confusion not exact')
        logits_sha = self.joint.array_digest(pred.detach().cpu().numpy())
        require(logits_sha == old['logits_array_sha256']['O'], 'Original full O all5h x3layer logit bytes differ')
        torch.cuda.synchronize(self.device)
        audit = dict(schema=SCHEMA, identity=row, input_tree_sha256=input_sha,
            historical_full_input_tree_sha256=old['input_tree_sha256'],
            historical_full_input_hash_equal=input_sha == old['input_tree_sha256'],
            historical_full_active_tree_comparison_performed=False,
            targets_array_sha256=gt_sha, observed_bev_tensor_sha256=bev_sha, native_radar_tensor_sha256=radar_sha,
            O_logits_array_sha256=logits_sha, hist_by_horizon=evaluated['hist_by_horizon'],
            target_counts_0_1_255_by_frame=counts.tolist(),
            original_full_O_hist_exact=True, original_full_O_all5h_3layer_logit_bytes_exact=True,
            planning_guard=self.guard(), observation_capture_seconds=capture_seconds,
            O_replay_and_evaluation_seconds=time.monotonic()-tick, seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device), optimizer_steps=0,
            inputs_gt_separate=True, GT_loaded_by_original_dataset_before_prediction=True)
        frame = dict(sample=sample, O_prediction=pred, t0_bev=sample['inputs']['prev_bev_input'][:,-1], audit=audit)
        require(tuple(frame['t0_bev'].shape) == (1,40000,256) and frame['t0_bev'].dtype == torch.float32, 'Measured t0 BEV contract')
        self.verify_frame(frame)
        self.position += 1
        return frame

    def verify_frame(self, frame):
        import torch
        sample, audit = frame['sample'], frame['audit']
        require(self.joint.tree_digest(sample['inputs']) == audit['input_tree_sha256'], 'Actual replay input mutated')
        require(self.joint.array_digest(sample['targets'].detach().cpu().numpy().astype('uint8')) == audit['targets_array_sha256']
                and bool(((sample['targets']==0)|(sample['targets']==1)|(sample['targets']==255)).all()), 'Full GT changed')
        require(self.joint.array_digest(frame['O_prediction'].detach().cpu().numpy()) == audit['O_logits_array_sha256'], 'O prediction modified by caller')
        require(torch.equal(frame['t0_bev'].view(torch.int32), sample['inputs']['prev_bev_input'][:,-1].view(torch.int32)), 'Measured t0 tokens changed')
        self.guard()

    def finish(self):
        require(self.position == len(self.ordinals), 'Scheduled streaming evaluation incomplete')
        self.guard(); self.verify_sources()
        require(self.state_digest(self.model) == self.initial_model_sha, 'Frozen O parameters/buffers changed')
        return dict(completed_samples=self.position, model_unchanged=True, optimizer_steps=0,
                    peak_allocated_bytes=__import__('torch').cuda.max_memory_allocated(self.device))


def cached_pilot_parity(stream, frame, cache, records, expected):
    import numpy as np
    import torch
    row = frame['sample']['record']; token = row['sample_token']; record = records[token]
    old = stream.native.load_sample(cache, record, device=stream.device)
    parity = stream.policy.compare_cache_boundary(frame['sample']['inputs'], old['inputs'],
        frame['audit']['input_tree_sha256'], stream.guard(), ['O'], stream.joint.tree_digest)
    require(torch.equal(old['targets'], frame['sample']['targets']), 'Old dev cache full GT differs')
    with torch.no_grad():
        cached_pred = stream.native.replay(stream.model, old, training=False)[0]
    require(torch.equal(cached_pred.view(torch.int32), frame['O_prediction'].view(torch.int32)), 'Same-time O replay from old cache not exact all5h x3layer')
    require(frame['audit']['hist_by_horizon'] == expected[token], 'Historical O development 5h confusion differs')
    parity.update(exact_gt_equal=True, same_time_O_old_cache_replay_all5h_3layer_bitwise_equal=True,
                  historical_O_development_hist_exact=True, historical_O_all_layer_cache_exists=False)
    del cached_pred, old
    stream.verify_frame(frame)
    return parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','o-checkpoint','repo','runtime-contract','selection','reference-dir',
                 'pilot-selection','parity-cache','o-development-records','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--max-allocated-gib', type=float, default=32.)
    a = parser.parse_args()
    require(0 < a.max_seconds <= 600 and 0 < a.max_allocated_gib <= 32, 'Two-anchor pilot resource cap')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def stop(signum, _frame):
        raise TimeoutError('Pilot stopped by signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGALRM, stop); signal.alarm(a.max_seconds)
    def check():
        require(time.monotonic()-started < a.max_seconds, 'Pilot deadline')
    try:
        pilot = read(a.pilot_selection, PILOT_SELECTION_SHA)
        require(pilot['samples'] == pilot['scenes'] == 2, 'Pilot requires original two anchors')
        ordinals = [r['official_index'] for r in pilot['records']]
        cache = Path(a.parity_cache)
        index = read(cache/'index.json', DEV_INDEX_SHA); done = read(cache/'complete.json')
        require(done['status'] == 'COMPLETE_NATIVE_STATE_CACHE' and done['index_sha256'] == DEV_INDEX_SHA
                and index['schema'] == 'm0-native-state-cache-v1' and index['status'] == 'COMPLETE'
                and len(index['records']) == 200, 'Original development cache incomplete')
        records = {r['sample_token']:r for r in index['records']}
        require(sha(a.o_development_records) == O_DEV_SHA, 'Original O development ledger changed')
        expected_rows = [json.loads(s) for s in Path(a.o_development_records).read_text().splitlines()]
        require(len(expected_rows) == 200 and [(r['sample_token'],r['scene_token']) for r in expected_rows]
                == [(r['sample_token'],r['scene_token']) for r in index['records']], 'Original O dev order differs')
        expected = {r['sample_token']:r['hist_by_horizon'] for r in expected_rows}
        manifest = dict(schema=SCHEMA, mode='pilot', script_sha256=sha(__file__), sources_sha256=SOURCES,
            config_sha256=CONFIG_SHA, M0_checkpoint_sha256=M0_SHA, O_checkpoint_sha256=O_SHA,
            runtime_contract_sha256=sha(a.runtime_contract), full_selection_sha256=FULL_SELECTION_SHA,
            pilot_selection_sha256=PILOT_SELECTION_SHA, original_full_complete_sha256=REFERENCE_COMPLETE_SHA,
            original_full_summary_sha256=REFERENCE_SUMMARY_SHA, dev_cache_index_sha256=DEV_INDEX_SHA,
            O_development_records_sha256=O_DEV_SHA, selected=pilot['records'], optimizer_steps=0,
            max_seconds=a.max_seconds, max_allocated_gib=a.max_allocated_gib,
            purpose='Only engineering parity/resources; no candidate performance or full-run completion')
        write(out/'manifest.json', manifest)
        stream = NativeOStream(config=a.config, checkpoint=a.checkpoint, o_checkpoint=a.o_checkpoint,
            repo=a.repo, runtime_contract=a.runtime_contract, selection=a.selection,
            reference_dir=a.reference_dir, device=a.device, ordinals=ordinals,
            max_allocated_gib=a.max_allocated_gib, check=check)
        write(out/'loaded_model.json', stream.loaded)
        rows = []
        with (out/'records.jsonl').open('x') as f:
            for ordinal in ordinals:
                sample_started = time.monotonic()
                frame = stream.fetch(ordinal)
                audit = frame['audit']
                audit['cached_pilot_parity'] = cached_pilot_parity(stream, frame, cache, records, expected)
                audit['complete_pilot_sample_seconds'] = time.monotonic()-sample_started
                f.write(json.dumps(audit, allow_nan=False)+'\n'); f.flush(); rows.append(audit)
                print(json.dumps(dict(event='PILOT_SAMPLE_COMPLETE', official_index=ordinal, seconds=audit['seconds'])), flush=True)
                del frame
        final = stream.finish(); check()
        summary = dict(schema=SCHEMA, status='PASS_NATIVE_FULL_O_STREAM_ENGINEERING', mode='pilot',
            samples=2, scenes=2, optimizer_steps=0, candidate_evaluation_performed=False,
            full_evaluation_performed=False, resources=dict(elapsed_seconds=time.monotonic()-started,
                model_initialization_seconds=stream.initialization_seconds,
                max_sample_seconds=max(r['complete_pilot_sample_seconds'] for r in rows), peak_allocated_bytes=final['peak_allocated_bytes']),
            model_unchanged=final['model_unchanged'], reference_verified_files_sha256=stream.reference['verified_files_sha256'],
            historical_validation_exposure=True, fixed_training_seed=11,
            boundary_exception='Only comparison projection of inactive plan_dict.sample_traj; actual inputs untouched')
        write(out/'summary.json', summary)
        write(out/'complete.json', dict(schema=SCHEMA,status=summary['status'],mode='pilot',samples=2,
            optimizer_steps=0,files_sha256={n:sha(out/n) for n in ('manifest.json','loaded_model.json','records.jsonl','summary.json')}))
    except BaseException as exc:
        write(out/'failed.json', dict(schema=SCHEMA,status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),
              seconds=time.monotonic()-started,optimizer_steps=0,completed_file=(out/'complete.json').exists()))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
