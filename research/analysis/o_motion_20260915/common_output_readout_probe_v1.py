"""Frozen O, first 16 TRAIN cache anchors: component-readout feasibility QA.

No optimizer, flow prediction, threshold search, development data or GT-assisted
prediction components. Box labels are deserialized only AFTER the native O
forward. Association concerns GT components only; it is not a motion metric.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np

M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
O_SHA = 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
O_HEAD_SHA = '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
TRAIN_INDEX_SHA = '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
CONFIG_SHA = 'c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
LABEL_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
ORACLE_MANIFEST_SHA = '5681684de97f5a78dcb8abae84d6fdb598360bae2c3c3053dc986ffbfc37fda4'
ORACLE_RECORDS_SHA = '4388c1181d76e4447a3b6301c23f0d2d4cd092c13be96abe68f7463793a1013b'
ORACLE_COMPLETE_SHA = '581933f14cf08e47c7651addf4b3dbb551e04a4b6747d58702a053170beed8d1'
RUNTIME_CONTRACT_SHA = '5972a886bb4bd48c1ecf621c3295ff29720cd12fce51315c4c7e34fe84d1cd9c'
HELPER_SHA = {
    'native_state_cache.py': '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0',
    'common_output_gospa.py': '7b72817719d84d650d0887cfd6b6bf4ead930ffa13e7e486688184255f3cfd6d',
    'motion_geometry.py': 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31'}
EXTENT = np.asarray([-51.2, -51.2, -5., 51.2, 51.2, 3.], dtype=np.float64)
IDENTITY = ('sample_token', 'scene_token', 'split', 'official_index')
STOP = False


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def identity(record):
    return {key: record[key] for key in IDENTITY}


def state_digest(module):
    # Byte-identical algorithm to frozen memory_experiment.state_digest.
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def component_stats(readout):
    return dict(count=len(readout['area_cells']),
                centers_xy_m=readout['centers_xy'].tolist(),
                area_cells=readout['area_cells'].tolist(),
                total_bev_cells=int(readout['area_cells'].sum()))


def associate_gt(foreground, readout, label, frame_index, check=lambda: None):
    """Exact GT-only association; no component edits and no purity threshold.

    Assignment uses all 7-frame union tracks valid at this frame. A pure
    component has all its foreground 3D centres uniquely in exactly one box.
    A complete pure instance has exactly one such component, all foreground
    centres inside its box in that component, and no overlap-assigned centres
    in its box. This is completeness within observed valid fine-GT foreground,
    NOT completeness of a physical object, annotation, or unobserved surface.
    """
    from motion_geometry import box_to_global, unique_box_assignment
    tracks = label['tracks']
    require(len({t['instance_token'] for t in tracks}) == len(tracks), 'Duplicate instance track')
    active = [t for t in tracks if t['valid_mask'][frame_index]]
    g0 = np.asarray(label['frames'][2]['lidar_to_global_column_matrix'], dtype=np.float64)
    boxes = np.asarray([np.linalg.inv(g0) @ box_to_global(
        t['global_centers_m'][frame_index], t['global_rotations_wxyz'][frame_index])
        for t in active], dtype=np.float64).reshape(-1, 4, 4)
    sizes = np.asarray([t['sizes_wlh_m'][frame_index] for t in active], dtype=np.float64).reshape(-1, 3)
    indices = np.argwhere(foreground)
    points = EXTENT[:3] + (indices.astype(np.float64)+.5) * (
        (EXTENT[3:]-EXTENT[:3])/np.asarray(foreground.shape))
    component_ids = readout['component_map'][indices[:, 0], indices[:, 1]]
    check()
    owners = unique_box_assignment(points, boxes, sizes)
    count = len(readout['area_cells'])
    total = np.bincount(component_ids, minlength=count+1)
    unknown = np.bincount(component_ids[owners == -1], minlength=count+1)
    overlaps = np.bincount(component_ids[owners == -2], minlength=count+1)
    by_component = [{} for _ in range(count+1)]
    by_instance = [[] for _ in active]
    good = owners >= 0
    if good.any():
        codes, amounts = np.unique(component_ids[good]*max(1, len(active))+owners[good], return_counts=True)
        for code, amount in zip(codes, amounts):
            cid, owner = divmod(int(code), len(active))
            by_component[cid][active[owner]['instance_token']] = int(amount)
            by_instance[owner].append(cid)
    components = []
    pure_by_token = {}
    for cid in range(1, count+1):
        pure = bool(len(by_component[cid]) == 1 and unknown[cid] == 0 and overlaps[cid] == 0)
        pure_token = next(iter(by_component[cid])) if pure else None
        if pure:
            pure_by_token.setdefault(pure_token, []).append(cid)
        components.append(dict(component_id=cid, foreground_voxels=int(total[cid]),
            unique_instance_voxels=by_component[cid], unique_instance_count=len(by_component[cid]),
            unknown_voxels=int(unknown[cid]), overlap_voxels=int(overlaps[cid]),
            pure_single_instance=pure, pure_instance_token=pure_token))
    instance_rows = []
    active_index = {t['instance_token']: i for i, t in enumerate(active)}
    for track in tracks:
        check()
        token = track['instance_token']; j = active_index.get(token)
        if j is None:
            instance_rows.append(dict(instance_token=token, frame_present=False,
                foreground_inside_box=0, unique_foreground_voxels=0, overlap_foreground_voxels=0,
                component_ids=[], pure_component_ids=[], complete_pure_component_id=None))
            continue
        # Per-box pass records overlaps involving THIS instance; -2 alone does
        # not identify its owners. Never certify completeness by ignoring them.
        in_box = unique_box_assignment(points, boxes[j:j+1], sizes[j:j+1]) == 0
        inside_count = int(in_box.sum()); overlap_count = int((in_box & (owners == -2)).sum())
        pure_ids = pure_by_token.get(token, [])
        whole = (len(by_instance[j]) == 1 and len(pure_ids) == 1 and overlap_count == 0
                 and int(total[pure_ids[0]]) == inside_count)
        instance_rows.append(dict(instance_token=token, frame_present=True,
            foreground_inside_box=inside_count, unique_foreground_voxels=int((owners == j).sum()),
            overlap_foreground_voxels=overlap_count, component_ids=by_instance[j],
            pure_component_ids=pure_ids, complete_pure_component_id=pure_ids[0] if whole else None))
    return dict(frame_index=frame_index, union_instances=len(tracks), active_instances=len(active),
        foreground_voxels=len(points), unknown_voxels=int((owners == -1).sum()),
        overlap_voxels=int((owners == -2).sum()), unique_voxels=int(good.sum()),
        component_count=count, pure_single_instance_components=len([c for c in components if c['pure_single_instance']]),
        mixed_components=sum(c['unique_instance_count'] > 1 for c in components),
        instances_with_multiple_components=sum(len(r['component_ids']) > 1 for r in instance_rows),
        complete_pure_instances=sum(r['complete_pure_component_id'] is not None for r in instance_rows),
        components=components, instances=instance_rows)


def endpoint_coverage(associations):
    initial = {r['instance_token']: r for r in associations[0]['instances']}
    rows = []
    for h, association in enumerate(associations[1:], 1):
        outcomes = []
        for later in association['instances']:
            early = initial[later['instance_token']]
            both = early['frame_present'] and later['frame_present']
            support = both and early['foreground_inside_box'] > 0 and later['foreground_inside_box'] > 0
            eligible = both and all(r['complete_pure_component_id'] is not None for r in (early, later))
            outcomes.append(dict(instance_token=later['instance_token'], both_frames_present=both,
                both_frames_have_GT_foreground=support, both_unique_complete_pure=eligible,
                t0_component=early['complete_pure_component_id'], h_component=later['complete_pure_component_id']))
        denominator = sum(r['both_frames_present'] for r in outcomes)
        support_denominator = sum(r['both_frames_have_GT_foreground'] for r in outcomes)
        numerator = sum(r['both_unique_complete_pure'] for r in outcomes)
        rows.append(dict(horizon_index=h, nominal_seconds=.5*h, union_instances=len(outcomes),
            both_present_instances=denominator, both_GT_supported_instances=support_denominator,
            usable_instances=numerator, usable_over_both_present=numerator/denominator if denominator else None,
            usable_over_both_GT_supported=numerator/support_denominator if support_denominator else None,
            instances=outcomes))
    return rows


def run(args):
    started = time.monotonic(); deadline = started+args.max_seconds
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    def check():
        if STOP or time.monotonic() >= deadline:
            raise TimeoutError('Signal/deadline; no retry or replacement sample')
    try:
        package = Path(__file__).resolve().parent
        for name, expected in HELPER_SHA.items():
            require(sha(package/name) == expected, 'Helper source mismatch: '+name)
        require(sha(args.runtime_contract) == RUNTIME_CONTRACT_SHA, 'Runtime contract bytes differ')
        source_contract = read(args.runtime_contract)
        for path, expected in source_contract['runtime_source_sha256'].items():
            require(sha(path) == expected, 'Frozen runtime source mismatch: '+path)
        for path, expected in [(args.config, CONFIG_SHA), (args.checkpoint, M0_SHA),
                               (args.o_checkpoint, O_SHA), (Path(args.train_cache)/'index.json', TRAIN_INDEX_SHA)]:
            require(sha(path) == expected, 'Frozen input mismatch: '+str(path))
        index = read(Path(args.train_cache)/'index.json'); selected = index['records'][:16]
        require(len(selected) == 16 and all(r['split'] == 'train' for r in selected), 'Require first 16 training anchors')
        oracle = Path(args.oracle)
        require(sha(oracle/'complete.json') == ORACLE_COMPLETE_SHA, 'Oracle completion bytes differ')
        require(sha(oracle/'manifest.json') == ORACLE_MANIFEST_SHA and sha(oracle/'records.jsonl') == ORACLE_RECORDS_SHA,
                'Completed oracle16 provenance mismatch')
        oc = read(oracle/'complete.json')
        require(oc['status'] == 'COMPLETE_DIAGNOSTIC_ONLY' and oc['samples'] == 16 and oc['optimizer_updates'] == 0
                and oc['records_sha256'] == ORACLE_RECORDS_SHA and oc['manifest_sha256'] == ORACLE_MANIFEST_SHA,
                'Oracle completion mismatch')
        oracle_manifest = read(oracle/'manifest.json')
        oracle_rows = [json.loads(line) for line in (oracle/'records.jsonl').read_text().splitlines()]
        require(oracle_manifest['anchors'] == [identity(r) for r in selected] and len(oracle_rows) == 16, 'Anchor order differs')
        labels = Path(args.labels)
        require(sha(labels/'manifest.json') == LABEL_MANIFEST_SHA, 'Raw-label manifest mismatch')
        lm = read(labels/'manifest.json'); lc = read(labels/'complete.json')
        require(lc['status'] == 'COMPLETE' and lc['samples'] == 712 and lc['manifest_sha256'] == LABEL_MANIFEST_SHA
                and lc['optimizer_steps'] == 0 and lc['model_or_prediction_read'] is False, 'Raw-label completion invalid')
        require(lm['train_dev_scenes_disjoint'] and lm['inputs_and_labels_physically_separate'], 'Label isolation invalid')
        bindings = {r['identity']['sample_token']: r for r in lm['records']}
        descriptors = []
        for record, prior in zip(selected, oracle_rows):
            require(prior['sample_token'] == record['sample_token'] and prior['scene_token'] == record['scene_token'], 'Oracle row order mismatch')
            desc = bindings[record['sample_token']]; path = labels/desc['file']
            require(desc['identity'] == identity(record) and desc['file'] == 'train/'+record['sample_token']+'.json.gz', 'Raw-label identity mismatch')
            require(path.stat().st_size == desc['bytes'] and sha(path) == desc['sha256'], 'Raw-label bytes mismatch')
            descriptors.append(desc)
        manifest = dict(schema='common-output-readout-probe-v1', status='RUNNING', seed=11,
            anchors=[identity(r) for r in selected], samples=16, scenes=len({r['scene_token'] for r in selected}),
            optimizer_updates=0, training=False, development_read=False, QA_not_formal_performance=True,
            raw_box_deserialization_after_O_forward=True, prediction_components_use_boxes=False,
            cutoff_m=4., p=2, alpha=2, max_pairwise_entries=4000000, component_connectivity=4,
            area_filter=None, extent_xyz=EXTENT.tolist(), output_shape_xyz=[512,512,40],
            output_contract='native last logits -> trilinear align_corners=False -> argmax -> same GT-valid mask -> z-any -> all components',
            association_contract='all seven-frame union tracks valid at each frame; no purity threshold; completeness only within observed foreground',
            max_seconds=args.max_seconds, max_allocated_gib=32.,
            source_sha256={**HELPER_SHA, Path(__file__).name:sha(__file__)},
            runtime_source_contract_sha256=sha(args.runtime_contract), runtime_source_sha256=source_contract['runtime_source_sha256'],
            config_sha256=CONFIG_SHA, m0_sha256=M0_SHA, o_checkpoint_sha256=O_SHA,
            expected_o_head_state_sha256=O_HEAD_SHA, train_index_sha256=TRAIN_INDEX_SHA,
            oracle_manifest_sha256=ORACLE_MANIFEST_SHA, oracle_records_sha256=ORACLE_RECORDS_SHA,
            oracle_complete_sha256=sha(oracle/'complete.json'), labels_manifest_sha256=LABEL_MANIFEST_SHA,
            labels_complete_sha256=sha(labels/'complete.json'), labels=descriptors,
            cache_files=[dict(sample_token=r['sample_token'], inputs=r['files']['inputs'], targets=r['files']['targets']) for r in selected])
        write(out/'manifest.json', manifest); event('MANIFEST_READY', seconds=time.monotonic()-started)
        import torch
        import scipy
        from native_state_cache import build_native_model, load_sample, replay
        from common_output_gospa import bev_component_centers, gospa2
        random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(.30)
        check()
        model=build_native_model(args.config,args.checkpoint,device='cuda',repo=args.repo)
        payload=torch.load(args.o_checkpoint,map_location='cpu')
        model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True); del payload
        for param in model.parameters(): param.requires_grad_(False)
        model.eval()
        require(model.future_pred_head.history_queue_length == 2, 'GT slicing differs')
        require(state_digest(model.future_pred_head) == O_HEAD_SHA, 'Loaded O state differs')
        write(out/'loaded_model.json', dict(native=model._native_state_provenance, o_head_state_sha256=O_HEAD_SHA,
            torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
            matmul_tf32=False,cudnn_tf32=True,grad_enabled=False,optimizer_loaded=False))
        event('MODEL_READY', seconds=time.monotonic()-started)
        collected=[]
        with torch.no_grad(), (out/'records.jsonl').open('x') as stream:
            for ordinal,(record,desc,prior) in enumerate(zip(selected,descriptors,oracle_rows)):
                check(); tick=time.monotonic(); torch.cuda.reset_peak_memory_stats()
                sample=load_sample(args.train_cache,record,device='cuda')
                pred=replay(model,sample,training=False)[0]
                require(tuple(pred.shape)==(5,3,1,1,40000,16,2) and torch.isfinite(pred).all().item(), 'Invalid native predictions')
                native_rows=model.evaluate_occ_records(pred,sample['targets'],sample['inputs']['img_metas'])
                require(len(native_rows)==1 and native_rows[0]['sample_token']==record['sample_token']
                        and native_rows[0]['scene_token']==record['scene_token'], 'Native evaluation identity mismatch')
                native_hist=np.asarray(native_rows[0]['hist_by_horizon'])
                require(np.array_equal(native_hist,np.asarray(prior['hist_by_arm']['O'])), 'O/fullGT hist changed vs oracle16')
                torch.cuda.synchronize(); inference_seconds=time.monotonic()-tick
                # The first semantic read of this raw box file is after O inference/evaluation.
                label=json.loads(gzip.decompress((labels/desc['file']).read_bytes()))
                require(label['schema']=='raw-nuscenes-motion-target-v1' and label['identity']==identity(record)
                        and len(label['frames'])==7 and not label['missing_fill_applied'] and not label['old_refine_applied'], 'Invalid raw-box sidecar')
                require(label['frames'][2]['sample_token']==record['sample_token']
                        and all(f['scene_token']==record['scene_token'] for f in label['frames']), 'Raw frame anchor mismatch')
                logits=pred[:,-1,0,0].reshape(5,200,200,16,2).permute(0,4,2,1,3)
                targets=sample['targets'][0,2:]
                horizons=[]; associations=[]
                for h in range(5):
                    check(); ht=time.monotonic()
                    fine=torch.nn.functional.interpolate(logits[h:h+1],size=targets.shape[-3:],mode='trilinear',align_corners=False)[0].argmax(0)
                    gt=targets[h]; valid=(gt>=0)&(gt<2)
                    cm=torch.bincount(2*gt[valid]+fine[valid],minlength=4).reshape(2,2).cpu().numpy()
                    require(np.array_equal(cm,native_hist[h]), 'Readout vs native full histogram differs')
                    gtfg=((gt==1)&valid).cpu().numpy(); predfg=((fine==1)&valid).cpu().numpy()
                    truth=bev_component_centers(gtfg,EXTENT); prediction=bev_component_centers(predfg,EXTENT)
                    self_result=gospa2(truth['centers_xy'],truth['centers_xy'])
                    require(self_result['squared_cost_m2']==0. and not self_result['missed_truth_indices']
                            and not self_result['false_prediction_indices'], 'GT/self readout is not zero')
                    result=gospa2(truth['centers_xy'],prediction['centers_xy'])
                    association=associate_gt(gtfg,truth,label,h+2,check)
                    associations.append(association)
                    horizons.append(dict(index=h,nominal_seconds=.5*h,actual_label_dt_seconds=label['frames'][h+2]['dt_seconds'],
                        hist=cm.tolist(),valid_voxels=int(valid.sum()),GT=component_stats(truth),O=component_stats(prediction),
                        gospa=result,GT_self_squared_cost_m2=0.,GT_association=association,seconds=time.monotonic()-ht))
                    del fine,gt,valid,gtfg,predfg,truth,prediction
                torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()/2**30
                require(peak<=32., 'GPU allocation exceeded 32 GiB')
                row=dict(ordinal=ordinal,**identity(record),horizons=horizons,endpoint_coverage=endpoint_coverage(associations),
                    native_and_oracle_hist_exact=True,gt_sha256=record['files']['targets']['sha256'],label_sha256=desc['sha256'],
                    inference_and_native_eval_seconds=inference_seconds,seconds=time.monotonic()-tick,peak_allocated_gib=peak,
                    process_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                collected.append(row); del sample,pred,logits,targets
                write(out/'progress.json',dict(completed_samples=len(collected),last_token=record['sample_token'],seconds=time.monotonic()-started))
                event('SAMPLE_COMPLETE',ordinal=ordinal,seconds=row['seconds'],peak_allocated_gib=peak)
                if ordinal==1:
                    projected=max(r['seconds'] for r in collected)*14 + (time.monotonic()-started)
                    require(projected<=args.max_seconds, 'First-two measured projection exceeds fixed runtime budget')
                    write(out/'first_two_engineering.json',dict(status='PASS',same_source=True,samples=2,
                        source_sha256=sha(__file__),projected_total_seconds=projected,max_seconds=args.max_seconds,
                        hist_exact=True,GT_self_zero=True,no_samples_dropped=True))
            check()
        require(state_digest(model.future_pred_head)==O_HEAD_SHA, 'O head changed during inference')
        totals=[]
        for h in range(5):
            hs=[r['horizons'][h] for r in collected]
            totals.append(dict(horizon_index=h,nominal_seconds=.5*h,
                GT_components=sum(r['GT']['count'] for r in hs),O_components=sum(r['O']['count'] for r in hs),
                GT_pure_components=sum(r['GT_association']['pure_single_instance_components'] for r in hs),
                GT_complete_pure_instances=sum(r['GT_association']['complete_pure_instances'] for r in hs),
                GT_foreground_voxels=sum(r['GT_association']['foreground_voxels'] for r in hs),
                unknown_voxels=sum(r['GT_association']['unknown_voxels'] for r in hs),
                overlap_voxels=sum(r['GT_association']['overlap_voxels'] for r in hs),
                sum_G_squared_m2=sum(r['gospa']['squared_cost_m2'] for r in hs)))
        endpoint=[]
        for h in range(4):
            rows=[r['endpoint_coverage'][h] for r in collected]
            denominator=sum(r['both_present_instances'] for r in rows);numerator=sum(r['usable_instances'] for r in rows)
            endpoint.append(dict(horizon_index=h+1,both_present_instances=denominator,usable_instances=numerator,
                usable_rate=numerator/denominator if denominator else None))
        summary=dict(schema='common-output-readout-summary-v1',status='PASS_QA_ONLY',samples=16,
            scenes=len({r['scene_token'] for r in collected}),optimizer_updates=0,training=False,development_read=False,
            horizons=totals,endpoint_coverage=endpoint,native_oracle_exact_rows=80,GT_self_zero_rows=80,
            max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in collected),seconds=time.monotonic()-started,
            no_motion_EPE_computed=True,no_displacement_identity_claim=True,
            interpretation='BEV connected components can merge objects or fragment one object; box association is GT-only feasibility, not instance ground truth or physical object completeness.')
        write(out/'summary.json',summary)
        report=['# Common-output readout QA','', '固定前 16 个训练 anchors；零训练、零开发集读取。全部组件保留，GOSPA c=4 m。',
                '80 个时域原生 hist 与已完成 oracle16 的 O 精确相同；GT 自匹配为零。',
                '这是组件读出与关联覆盖 QA，不是正式 GOSPA 成绩、实例检测或运动 EPE。', '',
                '| h | GT 组件 | O 组件 | GT 纯净组件 | GT 完整纯净实例 | 未知体素 | 重叠体素 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for r in totals:
            report.append('| {nominal_seconds} | {GT_components} | {O_components} | {GT_pure_components} | {GT_complete_pure_instances} | {unknown_voxels} | {overlap_voxels} |'.format(**r))
        report += ['', '“完整”只覆盖该实例框内实际可见有效 GT 前景体素；不表示整物体表面或精确实例标签。',
                   '所有实例的缺帧、无前景、混合、重叠与碎片详情保存在 records.jsonl；没有拟合纯度阈值。',
                   'G² 仅以 QA 样本的和记录，不冒充正式场景宏平均；无跨时间运动误差。']
        (out/'report.md').write_text('\n'.join(report)+'\n')
        write(out/'complete.json',dict(schema='common-output-readout-complete-v1',status='PASS_QA_ONLY',samples=16,
            optimizer_updates=0,seconds=time.monotonic()-started,source_sha256=sha(__file__),
            files_sha256={name:sha(out/name) for name in ['manifest.json','loaded_model.json','first_two_engineering.json','records.jsonl','summary.json','report.md']}))
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_NO_RETRY',exception=type(exc).__name__,message=str(exc),
            seconds=time.monotonic()-started,optimizer_updates=0,source_sha256=sha(__file__)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','o-checkpoint','train-cache','labels','oracle','runtime-contract','out','repo']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-seconds',type=int,default=840)
    args=parser.parse_args()
    require(0<args.max_seconds<=900,'Maximum 900 seconds')
    def stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    run(args)


if __name__=='__main__':
    main()
