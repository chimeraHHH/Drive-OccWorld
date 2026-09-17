"""Official CRN frozen-weight inference on train512; Python 3.8.

New output only. Full original train infos remain intact inside the Dataset;
Subset forwards original indices. No optimizer, GT metric, fitting or retry.
An original bottom-centered export and a deterministic centered copy are saved.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

CHECKPOINT_SHA = 'f725aafc7f033f484f13704134178f2377b3e12bb7e85207b447f8a3ebfa0cb3'
ORIGIN_ADAPTER_SHA = 'aa05e4cd8d2c9ffbf41e751ebc0bda7114040753e404c0560975ee134e23a101'
HERE = Path(__file__).resolve().parent


def helpers():
    path = HERE/'prepare_crn_train512_inputs_v1.py'
    spec = importlib.util.spec_from_file_location('_crn_train512_asset_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_assets(args, helper):
    root = Path(args.assets).resolve()
    helper.require(helper.sha(root/'complete.json') == args.assets_complete_sha256, 'Asset complete changed')
    complete = helper.read(root/'complete.json')
    helper.require(complete['status'] == 'COMPLETE_CRN_TRAIN512_ASSETS', 'Assets incomplete')
    for name, digest in complete['files_sha256'].items():
        helper.require(helper.sha(root/name) == digest, 'Asset metadata changed: '+name)
    manifest = helper.read(root/'manifest.json')
    helper.require(complete['source_sha256'] == helper.sha(helper.__file__) == manifest['preparation_source_sha256'],
                   'Preparation source differs')
    helper.require(manifest['selection_sha256'] == helper.SELECTION_SHA and
        manifest['selected_samples'] == 512 and manifest['selected_scenes'] == 256 and
        manifest['full_train_samples'] == 28130 and manifest['key_offsets'] == [0, -2, -4, -6], 'Asset identity contract')
    helper.require(manifest['source_contract_sha256'] == args.source_contract_sha256, 'Asset/code contract differs')
    helper.check_sources(args.repo, args.source_contract, args.source_contract_sha256)
    info_path = root/manifest['train_info']['file']
    helper.require(helper.sha(info_path) == complete['train_info_sha256'] == manifest['train_info']['sha256'], 'Train infos changed')
    for record in manifest['radar_files']:
        for item in record['files']:
            path = root/item['file']
            helper.require(path.stat().st_size == item['bytes'] and helper.sha(path) == item['sha256'], 'Radar preparation changed')
    helper.require(helper.sha(args.checkpoint) == CHECKPOINT_SHA, 'Official checkpoint changed')
    return root, manifest, info_path


def idle_gpu(index, uuid, require):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(index), 'Exactly the reviewed physical GPU must be visible')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], universal_newlines=True, timeout=15)
    mapping = {int(v[0].strip()): (v[1].strip(), int(v[2]))
        for v in (line.split(',') for line in rows.splitlines())}
    require(mapping[index][0] == uuid, 'GPU UUID differs')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], universal_newlines=True, timeout=15)
    require(not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()), 'GPU already has compute process')
    return dict(gpu_rows=rows, compute_apps=apps, selected_index=index, selected_uuid=uuid)


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(json.dumps(list(array.shape)).encode()); h.update(array.tobytes())
    return h.hexdigest()


def origin_adapter(helper):
    path = HERE/'crn_box_origin_adapter_v1.py'
    helper.require(helper.sha(path) == ORIGIN_ADAPTER_SHA, 'Frozen origin adapter changed')
    spec = importlib.util.spec_from_file_location('_crn_verified_origin_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args, out, helper):
    start = time.monotonic()
    def check():
        helper.require(time.monotonic()-start < args.max_seconds, 'Inference deadline exceeded')
    repo = Path(args.repo).resolve()
    assets, asset_manifest, info_path = authenticate_assets(args, helper)
    adapter = origin_adapter(helper)
    selection = helper.selected_records(args.selection)
    plans = helper.read(assets/'history_plan.json')['records']
    helper.require([p['identity'] for p in plans] == selection, 'History/selection order differs')
    pilot_ordinals = [0, next(i for i, r in enumerate(selection) if r['scene_token'] != selection[0]['scene_token'])]
    ordinals = pilot_ordinals if args.mode == 'pilot' else list(range(512))
    if args.mode == 'full':
        helper.require(args.pilot is not None, 'A completed same-source pilot is required')
        prior_root = Path(args.pilot)
        prior = helper.read(prior_root/'complete.json')
        helper.require(prior['status'] == 'PASS_CRN_TRAIN512_PILOT' and prior['source_sha256'] == helper.sha(__file__), 'Pilot incomplete/source changed')
        for name, digest in prior['files_sha256'].items():
            helper.require(helper.sha(prior_root/name) == digest, 'Pilot output changed')
        helper.require(prior['assets_complete_sha256'] == args.assets_complete_sha256 and
            prior['source_contract_sha256'] == args.source_contract_sha256 and
            prior['checkpoint_sha256'] == CHECKPOINT_SHA and prior['selection_sha256'] == helper.SELECTION_SHA,
            'Pilot binding differs')
    gpu_before = idle_gpu(args.gpu_index, args.gpu_uuid, helper.require)
    os.chdir(str(repo))
    # Resolve author modules from the verified checkout, not merely the cwd.
    sys.path[:0] = [str(repo), str(repo/'mmdetection3d')]
    import numpy as np
    import torch
    from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
    from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
    from layers.heads.bev_depth_head_det import BEVDepthHead
    helper.require(Path(inspect.getfile(CenterHead)).resolve().is_relative_to(repo)
                   if hasattr(Path('.'), 'is_relative_to') else
                   str(Path(inspect.getfile(CenterHead)).resolve()).startswith(str(repo)+os.sep), 'Unexpected mmdet3d import')
    helper.require(BEVDepthHead.get_bboxes is CenterHead.get_bboxes, 'CRN decode source changed')
    torch.set_num_threads(4)
    # Preserve the previous official inference entry seed; no training seed/run.
    torch.manual_seed(0); np.random.seed(0)
    model = CRNLightningModel(gpus=1, batch_size_per_device=1, default_root_dir=str(out/'author_output'))
    payload = torch.load(str(Path(args.checkpoint).resolve()), map_location='cpu')
    model.load_state_dict(payload['state_dict'], strict=True)
    del payload
    for value in model.state_dict().values():
        helper.require(bool(torch.isfinite(value).all()), 'Nonfinite official weights')
    model.requires_grad_(False)
    model.return_depth = False
    model.val_info_paths = str(info_path)
    model.data_root = str(assets/'data')
    # This invokes the unchanged evaluation pipeline on full train metadata.
    original_loader = model.val_dataloader()
    dataset = original_loader.dataset
    helper.require(not dataset.is_train and not dataset.use_cbgs and len(dataset) == 28130 and
        dataset.key_idxes == [0, -2, -4, -6] and dataset.sweeps_idx == [] and
        not dataset.return_depth and dataset.return_image and dataset.return_radar_pv, 'Native eval dataset settings changed')
    helper.require(helper.history_plan(dataset.infos, selection) == plans, 'Native full-index history differs')
    subset_indices = [plans[i]['native_index'] for i in ordinals]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, subset_indices),
        batch_size=1, shuffle=False, num_workers=original_loader.num_workers,
        collate_fn=original_loader.collate_fn)
    model.eval().cuda()
    initial_state = state_digest(model)
    predictions, metadata, timing = [], [], []
    initialization_seconds = time.monotonic()-start
    previous_end = time.monotonic()
    for row_index, batch in enumerate(loader):
        check(); t = time.monotonic()
        ordinal = ordinals[row_index]
        helper.require(batch[2][0]['token'] == selection[ordinal]['sample_token'], 'Unexpected input token')
        # Native eval_step consumes positions 0,1,2,7 only. GT is not supplied.
        input_only = (batch[0], batch[1], batch[2], None, None, None, None, batch[7])
        del batch
        with torch.no_grad(), torch.cuda.amp.autocast():
            result = model.eval_step(input_only, row_index, 'test')
        torch.cuda.synchronize()
        helper.require(len(result) == 1 and result[0][3]['token'] == selection[ordinal]['sample_token'], 'Output token differs')
        for values in result[0][:3]:
            helper.require(np.isfinite(values).all(), 'Nonfinite native prediction')
        predictions.append(result[0][:3]); metadata.append(result[0][3])
        peak = torch.cuda.max_memory_allocated()
        helper.require(peak <= args.max_allocated_gib*1024**3, 'GPU memory ceiling exceeded')
        timing.append(dict(ordinal=ordinal, sample_token=selection[ordinal]['sample_token'],
            inference_seconds=time.monotonic()-t, peak_allocated_bytes=peak,
            data_and_inference_seconds=time.monotonic()-previous_end,
            native_boxes=len(result[0][0]), data_and_inference_elapsed_seconds=time.monotonic()-start))
        helper.write(out/'progress.json', dict(completed_samples=len(timing), planned_samples=len(ordinals),
            last=timing[-1], optimizer_updates=0))
        del input_only, result
        previous_end = time.monotonic()
    helper.require(state_digest(model) == initial_state, 'Frozen official state changed')
    # Invoke only native formatting, never its NuScenesEval/GT scoring method.
    raw_path = Path(model.evaluator._format_bbox(predictions, metadata, str(out/'raw_export')))
    raw = helper.read(raw_path)
    expected_tokens = [selection[i]['sample_token'] for i in ordinals]
    helper.require(list(raw['results']) == expected_tokens, 'Export order/identity differs')
    records = []
    for ordinal in ordinals:
        identity = selection[ordinal]
        boxes = raw['results'][identity['sample_token']]
        for box in boxes:
            helper.require(box['sample_token'] == identity['sample_token'], 'Export box token differs')
            for key in ('translation', 'size', 'rotation', 'velocity', 'detection_score'):
                helper.require(np.isfinite(np.asarray(box[key], dtype=np.float64)).all(), 'Nonfinite export')
            helper.require(np.all(np.asarray(box['size']) > 0), 'Invalid box size')
        records.append(adapter.adapt_record(dict(ordinal=ordinal, **identity, boxes=boxes)))
    helper.check_sources(repo, args.source_contract, args.source_contract_sha256)
    check()
    helper.write(out/'predictions.json', dict(schema='crn-state-train512-predictions-v1', meta=raw['meta'],
        box_origin='global_geometric_center', raw_export_origin='global_transformed_ego_bottom_center',
        center_conversion='translation + R(export_quaternion) @ [0,0,height/2]', records=records))
    helper.write(out/'manifest.json', dict(schema='crn-train512-inference-v1', mode=args.mode,
        selection_sha256=helper.SELECTION_SHA, assets_complete_sha256=args.assets_complete_sha256,
        source_contract_sha256=args.source_contract_sha256, source_sha256=helper.sha(__file__),
        preparation_source_sha256=helper.sha(helper.__file__), checkpoint_sha256=CHECKPOINT_SHA,
        origin_adapter_sha256=ORIGIN_ADAPTER_SHA,
        official_commit='5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d',
        fixed_model_state_sha256=initial_state, selected_samples=len(records),
        selected_boxes=sum(len(r['boxes']) for r in records), full_train_infos_samples=28130,
        native_indices=subset_indices, selection_ordinals=ordinals, gpu_before=gpu_before,
        model_training=False, optimizer_updates=0, GT_batch_passed_to_model=False,
        detector_rerun_on_development=False, changed_original_data_or_code=False,
        inference_seed=0, seed_semantics='unchanged original official inference entry; no new training seed',
        dataloader_workers=original_loader.num_workers,
        radar_point_cap_policy='Unmodified author transform: random choice without replacement above 1536 points, including eval; no pilot/full bitwise prediction claim',
        precision='original torch.cuda.amp.autocast fp16', initial_model_state_equals_final=True,
        source_contract=helper.read(args.source_contract), asset_manifest_sha256=helper.sha(assets/'manifest.json'),
        raw_export_sha256=helper.sha(raw_path), raw_export_box_fields_other_than_translation_unchanged=True,
        initialization_seconds=initialization_seconds, timing=timing,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-start,
        runtime=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)))
    status = 'PASS_CRN_TRAIN512_PILOT' if args.mode == 'pilot' else 'COMPLETE_CRN_TRAIN512_INFERENCE'
    helper.write(out/'complete.json', dict(schema='crn-train512-inference-complete-v1', status=status, mode=args.mode,
        files_sha256={f: helper.sha(out/f) for f in ('manifest.json', 'predictions.json', 'raw_export/results_nusc.json')},
        source_sha256=helper.sha(__file__), assets_complete_sha256=args.assets_complete_sha256,
        source_contract_sha256=args.source_contract_sha256, checkpoint_sha256=CHECKPOINT_SHA,
        origin_adapter_sha256=ORIGIN_ADAPTER_SHA,
        selection_sha256=helper.SELECTION_SHA, samples=len(records), optimizer_updates=0))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('pilot', 'full'), required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--assets-complete-sha256', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source-contract', required=True)
    p.add_argument('--source-contract-sha256', required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--pilot')
    p.add_argument('--out', required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--max-allocated-gib', type=float, default=16)
    args = p.parse_args()
    helper = helpers()
    helper.require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= (600 if args.mode == 'pilot' else 1800), 'Bounded inference budget')
    helper.require(0 < args.max_allocated_gib <= 16, 'Memory ceiling must be at most 16 GiB')
    # Resolve arguments before entering the author's working directory.
    for key in ('assets', 'selection', 'repo', 'checkpoint', 'source_contract', 'pilot'):
        if getattr(args, key) is not None:
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        print(json.dumps(dict(status=run(args, out, helper), out=str(out))))
    except Exception as error:
        helper.write(out/'failed.json', dict(status='FAILED_NO_RETRY', error=repr(error), source_sha256=helper.sha(__file__)))
        raise


if __name__ == '__main__':
    main()
