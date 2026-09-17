"""No-training source-occupancy transport diagnostic on fixed training anchors.

Every transported arm uses GT current boxes; the oracle also uses GT future
poses. These are diagnostic interventions, NEVER valid candidate predictions.
O inference is completed before reading the separate annotation sidecar.
The original model, future coordinate routing, full GT and evaluator are kept.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
O_SHA = 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
TRAIN_INDEX_SHA = '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
CONFIG_SHA = 'c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
LABEL_MANIFEST_SHA = '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
EXTENT = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
SHAPE = (200, 200, 16)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    p = Path(path); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n'); tmp.replace(p)


def event(name, **kwargs):
    print(json.dumps(dict(event=name, **kwargs), allow_nan=False), flush=True)


def predictions_to_xyz(pred):
    # Original evaluate_occ_records: HW is stored Y-major, then transposes XY.
    assert tuple(pred.shape) == (5, 3, 1, 1, 40000, 16, 2), pred.shape
    return pred[:, -1, 0, 0].reshape(5, 200, 200, 16, 2).permute(0, 4, 2, 1, 3)


def replace_last_logits(pred, xyz):
    assert tuple(xyz.shape) == (5, 2, *SHAPE)
    out = pred.clone()
    out[:, -1, 0, 0] = xyz.permute(0, 3, 2, 4, 1).reshape(5, 40000, 16, 2)
    return out


def annotation_flows(label):
    """Full rigid future oracle and past-only translation CV, both in t0 R.

    Overlapping source boxes are excluded rather than arbitrarily resolved.
    Unassociated/missing-future sources persist with zero motion, and coverage
    is reported. CV uses the most recent present history annotation only.
    """
    from motion_geometry import box_to_global, rigid_displacement, unique_box_assignment, voxel_centers_xyz
    G0 = np.asarray(label['frames'][2]['lidar_to_global_column_matrix'], dtype=np.float64)
    dt = np.asarray([x['dt_seconds'] for x in label['frames']], dtype=np.float64)
    tracks = [x for x in label['tracks'] if x['valid_mask'][2]]
    points = voxel_centers_xyz(SHAPE, EXTENT).reshape(-1, 3)
    A0 = [box_to_global(x['global_centers_m'][2], x['global_rotations_wxyz'][2]) for x in tracks]
    boxes_R = np.asarray([np.linalg.inv(G0) @ a for a in A0]).reshape(-1, 4, 4)
    sizes = np.asarray([x['sizes_wlh_m'][2] for x in tracks]).reshape(-1, 3)
    assignment = unique_box_assignment(points, boxes_R, sizes)
    flow = np.zeros((4, len(points), 3), dtype=np.float32)
    cv = np.zeros_like(flow)
    valid = np.zeros((4, len(points)), dtype=bool)
    cv_valid = np.zeros_like(valid)
    moving = np.zeros_like(valid)
    center_records = []
    for j, track in enumerate(tracks):
        indices = np.flatnonzero(assignment == j)
        past = [k for k in (1, 0) if track['valid_mask'][k]]
        velocity = None
        if past:
            k = past[0]
            velocity = G0[:3, :3].T @ (np.asarray(track['global_centers_m'][2])-
                       np.asarray(track['global_centers_m'][k])) / (-dt[k])
        item = dict(instance_token=track['instance_token'], category=track['category_name'],
                    source_voxels=len(indices), history_cv_available=velocity is not None, horizons=[])
        for h in range(4):
            if velocity is not None:
                cv[h, indices] = velocity * dt[h+3]
                cv_valid[h, indices] = True
            if not track['valid_mask'][h+3]:
                item['horizons'].append(None); continue
            Ah = box_to_global(track['global_centers_m'][h+3], track['global_rotations_wxyz'][h+3])
            flow[h, indices] = rigid_displacement(points[indices], G0, A0[j], Ah)
            valid[h, indices] = True
            center_delta = G0[:3, :3].T @ (np.asarray(track['global_centers_m'][h+3])-
                                         np.asarray(track['global_centers_m'][2]))
            # Moving is physical GLOBAL horizontal center speed, not GMO class.
            world_delta = np.asarray(track['global_centers_m'][h+3])-np.asarray(track['global_centers_m'][2])
            speed = float(np.linalg.norm(world_delta[:2])/dt[h+3])
            moving[h, indices] = speed >= .5
            item['horizons'].append(dict(dt_seconds=float(dt[h+3]), center_delta_R_m=center_delta.tolist(),
                global_xy_average_speed_mps=speed,
                cv_center_error_m=None if velocity is None else float(np.linalg.norm(velocity*dt[h+3]-center_delta))))
        center_records.append(item)
    return dict(oracle=flow.reshape(4, *SHAPE, 3), cv=cv.reshape(4, *SHAPE, 3),
                oracle_valid=valid.reshape(4, *SHAPE), cv_valid=cv_valid.reshape(4, *SHAPE),
                moving=moving.reshape(4, *SHAPE), source_assignment=assignment.reshape(SHAPE),
                tracks=center_records, overlap_voxels=int((assignment == -2).sum()))


def hist(model, pred, sample):
    rows = model.evaluate_occ_records(pred, sample['targets'], sample['inputs']['img_metas'])
    assert len(rows) == 1
    return rows[0]['hist_by_horizon'].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'o-checkpoint', 'train-cache', 'labels', 'out', 'repo'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--anchors', type=int, default=16)
    a = p.parse_args()
    assert a.anchors in (2, 16), 'Engineering=2 or fixed training diagnostic=16 only'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch
    from native_state_cache import build_native_model, load_sample, replay
    from transport_ops import forward_splat_3d
    torch.set_num_threads(2); torch.manual_seed(11); np.random.seed(11)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    assert sha(a.checkpoint) == M0_SHA and sha(a.o_checkpoint) == O_SHA and sha(a.config) == CONFIG_SHA
    index_path = Path(a.train_cache)/'index.json'
    assert sha(index_path) == TRAIN_INDEX_SHA
    index = json.loads(index_path.read_text()); rows = index['records'][:a.anchors]
    assert all(x['split'] == 'train' for x in rows)
    label_root = Path(a.labels)
    label_complete = json.loads((label_root/'complete.json').read_text())
    assert label_complete['status']=='COMPLETE' and label_complete['samples']==712
    assert label_complete['optimizer_steps']==0 and not label_complete['model_or_prediction_read']
    assert label_complete['manifest_sha256']==LABEL_MANIFEST_SHA==sha(label_root/'manifest.json')
    label_manifest=json.loads((label_root/'manifest.json').read_text())
    assert label_manifest['train_dev_scenes_disjoint'] and label_manifest['inputs_and_labels_physically_separate']
    bindings={x['identity']['sample_token']:x for x in label_manifest['records']}
    label_files = [label_root/'train'/(r['sample_token']+'.json.gz') for r in rows]
    assert all(x.is_file() for x in label_files)
    for record,path in zip(rows,label_files):
        desc=bindings[record['sample_token']]
        assert all(desc['identity'][k]==record[k] for k in ('sample_token','scene_token','split','official_index'))
        assert desc['file']==str(path.relative_to(label_root)) and desc['bytes']==path.stat().st_size
        assert desc['sha256']==sha(path)
    manifest = dict(schema='o-motion-oracle-transport-probe-v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seed=11, optimizer_updates=0, candidate_performance=False, labels_used_at_inference_in_diagnostic=True,
        original_O_inference_receives_labels=False, source='frozen O current predicted probabilities; no GT occupancy source',
        physical_space='fixed t0 LiDAR R, matching original fine GT; O internal frames unchanged',
        source_shape_xyz=list(SHAPE), extent_xyz=list(EXTENT),
        arms=['O','persist_O0','static_splat_O0','oracle_O0','cv_O0','blend_oracle_half','blend_static_half'],
        fusion='fixed 0.5 native binary log-odds + 0.5 transported log-odds; no fitted weight',
        splat_probability='sum trilinear source foreground probability mass; clamp to [1e-6,1-1e-6] only before logit',
        missing_flow='zero displacement; no source suppression',
        anchors=[{k:r[k] for k in ('sample_token','scene_token','split','official_index')} for r in rows],
        train_cache_index_sha256=sha(index_path), o_checkpoint_sha256=O_SHA,
        labels_complete_sha256=sha(label_root/'complete.json'),
        labels_manifest_sha256=LABEL_MANIFEST_SHA,
        label_sha256={x.name:sha(x) for x in label_files},
        source_sha256={x.name:sha(x) for x in Path(__file__).parent.glob('*.py')})
    write(out/'manifest.json', manifest)
    model=build_native_model(a.config,a.checkpoint,device='cuda',repo=a.repo)
    payload=torch.load(a.o_checkpoint,map_location='cpu')
    model.future_pred_head.load_state_dict(payload['future_pred_head'],strict=True); del payload
    for param in model.parameters(): param.requires_grad_(False)
    model.eval()
    event('MODEL_READY', seconds=time.monotonic()-started)
    collected=[]
    with torch.no_grad(), (out/'records.jsonl').open('x') as stream:
        for ordinal,(record,label_path) in enumerate(zip(rows,label_files)):
            tick=time.monotonic(); sample=load_sample(a.train_cache,record,device='cuda')
            pred=replay(model,sample,training=False)[0]
            base=predictions_to_xyz(pred)
            assert torch.equal(replace_last_logits(pred,base),pred), 'XYZ/native inverse permutation failed'
            original_hist=hist(model,pred,sample)
            # Labels are first deserialized only AFTER unmodified O inference.
            label=json.loads(gzip.decompress(label_path.read_bytes()))
            assert label['schema']=='raw-nuscenes-motion-target-v1'
            assert label['identity']['sample_token']==record['sample_token']
            assert not label['missing_fill_applied'] and not label['old_refine_applied']
            flows=annotation_flows(label)
            prob=torch.softmax(base[0:1],dim=1)[:,1:2]
            field_results={}; transport_stats={}
            for name in ('oracle','cv','static'):
                hs=[]; stats=[]
                for h in range(4):
                    flow_numpy=flows[name][h] if name!='static' else np.zeros((*SHAPE,3),dtype=np.float32)
                    displacement=torch.from_numpy(flow_numpy).to(prob.device).permute(3,0,1,2).unsqueeze(0)
                    result=forward_splat_3d(prob,displacement,EXTENT)
                    mass=result['numerator']
                    assert mass.shape==prob.shape and torch.isfinite(mass).all()
                    p2=mass.clamp(1e-6,1-1e-6)
                    hs.append((torch.log(p2)-torch.log1p(-p2))[0,0])
                    support=torch.from_numpy(flows[name+'_valid'][h] if name!='static' else np.ones(SHAPE,dtype=bool)).to(prob.device)
                    stats.append(dict(source_probability_mass=float(prob.sum()),destination_mass=float(mass.sum()),
                        labeled_source_mass=float(prob[0,0][support].sum()),valid_source_voxels=int(support.sum()),
                        collision_mass_above_one=float((mass-1).clamp_min(0).sum()),
                        clipped_low_voxels=int((mass<1e-6).sum()),clipped_high_voxels=int((mass>1-1e-6).sum())))
                field_results[name]=torch.stack(hs); transport_stats[name]=stats
            base_delta=base[:,1]-base[:,0]
            deltas={'persist_O0':base_delta[0:1].expand(4,-1,-1,-1),
                    'static_splat_O0':field_results['static'],
                    'oracle_O0':field_results['oracle'],'cv_O0':field_results['cv'],
                    'blend_oracle_half':.5*base_delta[1:]+.5*field_results['oracle'],
                    'blend_static_half':.5*base_delta[1:]+.5*field_results['static']}
            matrices={'O':original_hist}
            for name,d in deltas.items():
                logits=base.clone(); logits[1:,1]=logits[1:,0]+d
                changed=replace_last_logits(pred,logits)
                assert torch.equal(changed[0],pred[0]), 't0 must remain exactly O'
                matrices[name]=hist(model,changed,sample)
                del changed,logits
            row=dict(ordinal=ordinal,sample_token=record['sample_token'],scene_token=record['scene_token'],
                hist_by_arm=matrices, transport_stats=transport_stats, tracks=flows['tracks'],
                overlap_source_voxels=flows['overlap_voxels'], seconds=time.monotonic()-tick)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();collected.append(row)
            event('ANCHOR_COMPLETE',completed=ordinal+1,planned=len(rows),seconds=row['seconds'])
            del sample,pred,base,prob,field_results,flows,deltas,base_delta
    scores={}
    for name in manifest['arms']:
        c=np.asarray([r['hist_by_arm'][name] for r in collected],dtype=np.int64).sum(0)
        den=c[:,1,:].sum(-1)+c[:,:,1].sum(-1)-c[:,1,1]
        iou=c[:,1,1]/np.maximum(den,1)*100
        scores[name]=dict(hist_by_horizon=c.tolist(),iou_by_horizon_percent=iou.tolist(),
                          future_mean_iou_percent=float(iou[1:].mean()))
    summary=dict(status='COMPLETE_DIAGNOSTIC_ONLY',samples=len(rows),scenes=len({x['scene_token'] for x in rows}),
                 seconds=time.monotonic()-started,scores=scores,optimizer_updates=0,candidate_performance=False,
                 inference_uses_oracle_labels=True,no_blind_test_claim=True,
                 purpose='Determine whether source-shape transport with true object motion has headroom; no learned method result')
    write(out/'summary.json',summary)
    write(out/'complete.json',dict(status=summary['status'],summary_sha256=sha(out/'summary.json'),
        manifest_sha256=sha(out/'manifest.json'),records_sha256=sha(out/'records.jsonl'),samples=len(rows),optimizer_updates=0))
    event('COMPLETE',seconds=summary['seconds'])


if __name__=='__main__':
    main()
