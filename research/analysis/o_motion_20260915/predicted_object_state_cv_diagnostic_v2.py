"""CPU-only CRN state CV with source-confirmed bottom-to-centre geometry correction.

Prediction uses only current exported boxes/velocities, G0, and query points.
GT is used for evaluation support and targets, never prediction association.
No detector forward, parameter update, new confidence threshold, or D vector
reconstruction is performed. Run real scoring only after coordinator review.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
EXPORT_SHA = 'fec6541461bf6eb0ad5f3e113e4fcb13c937896731203ef5de003a6ac0055653'
ORIGIN_PROOF_SHA = '2d9b711857e82ff1c3e88be16064a05d12b03d63f23f1d2545941c5b6fca0149'
ORIGIN_CONTRACT_SHA = 'c4b99aadab4bdd20485c3a544ba9cd735441101f59ec228230a6d79f5d46a51b'
BOUND = {
    'crn_box_origin_adapter_v1.py': 'aa05e4cd8d2c9ffbf41e751ebc0bda7114040753e404c0560975ee134e23a101',
    'motion_geometry.py': 'e9d232c3f7aaacd073cfda645868e357afad9e5a2684d07b2f3f2394bd796a31',
    'history_state_predictability_diagnostic_v1.py': '2b14c95bfa027416304288eec1282d1cc001984be42261cf656ee2cb6d6c96da',
    'train_source_motion_v1.py': 'e34efdfd6dca7b60a8a4ddb3359417bbe5ba9e7b04a4b150e8b6428b81dd582e',
    'summarize_future_state_physical_v1.py': '920f4f96e0bd5df247b3d096bdf1709d80fed593622b70373093ea1160db2de3',
}
GMO = ('car', 'bus', 'truck', 'trailer', 'construction_vehicle',
       'motorcycle', 'bicycle', 'pedestrian')
HORIZONS = (.5, 1., 1.5, 2.)
GROUPS = ('stationary', 'ambiguous', 'moving')
ARMS = ('D', 'CRN_CV')
PAIRS = (('CRN_CV', 'D'), ('CRN_CV', 'zero'), ('D', 'zero'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(name):
    path = HERE / name
    require(sha(path) == BOUND[name], 'Frozen dependency changed: ' + name)
    key = '_predicted_cv_' + path.stem
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def predicted_velocity_field(boxes, G0, query_points_R):
    """Return velocity [N,3] in R and original-export owner indices [N].

    The signature deliberately accepts no annotation identity or future data.
    nuScenes box size is w,l,h; local axes are l,w,h. Closed box faces use
    the existing geometry helper's 1e-9 metre tolerance. Highest score wins;
    strict > retains the earlier export box on ties. Uncovered velocity is 0.
    """
    geo = bound('motion_geometry.py')
    pose = geo._rigid_pose(G0, 'G0')
    points = geo._points(query_points_R)
    global_points = points @ pose[:3, :3].T + pose[:3, 3]
    velocity_global = np.zeros((len(points), 3), dtype=np.float64)
    owner = np.full(len(points), -1, dtype=np.int64)
    best = np.full(len(points), -np.inf, dtype=np.float64)
    require(isinstance(boxes, list), 'Export boxes must retain their list order')
    for index, box in enumerate(boxes):
        if box['detection_name'] not in GMO:
            continue
        score = float(box['detection_score'])
        require(math.isfinite(score), 'Nonfinite detection score')
        centre = geo._finite_array(box['translation'], (3,), 'predicted centre')
        size = geo._finite_array(box['size'], (3,), 'predicted wlh')
        require(np.all(size > 0), 'Nonpositive predicted box size')
        rotation = geo.quaternion_wxyz_to_matrix(box['rotation'])
        velocity = geo._finite_array(box['velocity'], (2,), 'global velocity')
        local = (global_points - centre) @ rotation
        inside = np.all(np.abs(local) <= size[[1, 0, 2]] / 2. + 1e-9, axis=1)
        choose = inside & (score > best)
        owner[choose] = index
        best[choose] = score
        velocity_global[choose, :2] = velocity
    # Points include translation above; vectors receive rotation only here.
    velocity_R = velocity_global @ np.linalg.inv(pose)[:3, :3].T
    return velocity_R, owner


def self_test():
    def box(center=(0., 0., 0.), size=(2., 8., 2.), rotation=(1., 0., 0., 0.),
            velocity=(2., 0.), score=.5, name='car'):
        return dict(translation=list(center), size=list(size), rotation=list(rotation),
                    velocity=list(velocity), detection_score=score, detection_name=name)
    # 1: translating both poses changes no vector; yaw rotates the vector.
    G = np.array([[0., -1., 0., 100.], [1., 0., 0., -20.],
                  [0., 0., 1., 3.], [0., 0., 0., 1.]])
    v, own = predicted_velocity_field([box(center=G[:3, 3])], G, np.zeros((1, 3)))
    require(np.array_equal(v, [[0., -2., 0.]]) and own.tolist() == [0], 'Vector rotation test failed')
    G2 = G.copy(); G2[:3, 3] += [200., 30., -10.]
    v2, _ = predicted_velocity_field([box(center=G2[:3, 3])], G2, np.zeros((1, 3)))
    require(np.array_equal(v, v2), 'Translation leaked into velocity')
    # 2: wlh, rotation, higher score, exact score tie, and class exclusion.
    q = (np.sqrt(.5), 0., 0., np.sqrt(.5))
    boxes = [box(rotation=q, score=.2, velocity=(1., 0.)),
             box(rotation=q, score=.9, velocity=(3., 4.)),
             box(rotation=q, score=.9, velocity=(7., 8.)),
             box(rotation=q, score=1., velocity=(9., 9.), name='barrier')]
    v, own = predicted_velocity_field(boxes, np.eye(4), np.array([[0., 3., 0.], [3., 0., 0.]]))
    require(own.tolist() == [1, -1] and np.array_equal(v, [[3., 4., 0.], [0., 0., 0.]]),
            'wlh/rotation/score/tie/class rule failed')
    # 3: no coverage is a prediction of zero, never an excluded sample.
    v, own = predicted_velocity_field([], np.eye(4), np.array([[7., 8., 9.]]))
    require(np.array_equal(v, np.zeros((1, 3))) and own.tolist() == [-1], 'Uncovered rule failed')
    require('torch' not in sys.modules, 'CPU-only tests imported torch')
    return dict(status='THREE_CPU_GEOMETRY_CHECKS_PASS', checks=3,
                geometry_sha256=BOUND['motion_geometry.py'], script_sha256=sha(__file__),
                real_prediction_scoring_performed=False, model_forward=False, optimizer_updates=0)


def authenticate(base, selection_path, extraction_root):
    origin = bound('crn_box_origin_adapter_v1.py')
    proof_path = base/'receipts/crn_z_origin_runtime_v1.json'
    require(sha(proof_path) == ORIGIN_PROOF_SHA, 'Actual origin-source proof changed')
    require(sha(base/'crn_origin_correction_contract_v1.json') == ORIGIN_CONTRACT_SHA, 'Origin correction contract changed')
    proof = read(proof_path)
    require(proof['returncode'] == 0 and proof['result']['head_inherited_get_bboxes'] and
            not proof['result']['cuda_initialized'], 'Invalid actual source origin proof')
    history = bound('history_state_predictability_diagnostic_v1.py')
    helper = bound('train_source_motion_v1.py')
    stats = bound('summarize_future_state_physical_v1.py')
    require(sha(selection_path) == SELECTION_SHA, 'Fixed selection changed')
    selection = [r for r in read(selection_path)['records'] if r['split'] == 'development']
    require(len(selection) == 200 and len({r['scene_token'] for r in selection}) == 100,
            'Expected full dev200/100 scenes')
    ec = read(extraction_root / 'complete.json'); em = read(extraction_root / 'manifest.json')
    require(ec['status'] == 'COMPLETE_CRN_STATE_DEV200_EXTRACTION', 'Incomplete CRN extraction')
    require(set(ec['files_sha256']) == {'predictions.json', 'manifest.json'}, 'Unexpected extraction ledger')
    history.small_chain(extraction_root, ec)
    require(em['selection_sha256'] == SELECTION_SHA and em['export_sha256'] == EXPORT_SHA,
            'CRN extraction source/selection differs')
    pred = read(extraction_root / 'predictions.json')
    require(pred['schema'] == 'crn-state-dev200-predictions-v1', 'Wrong extracted prediction schema')
    records = pred['records']; total_boxes = 0
    require(len(records) == em['selected_samples'] == 200, 'Missing extracted predictions')
    for ordinal, (record, identity) in enumerate(zip(records, selection)):
        require(record['ordinal'] == ordinal and all(record[k] == v for k, v in identity.items()),
                'CRN sample/order/scene identity differs')
        require(isinstance(record['boxes'], list) and
                all(b['sample_token'] == identity['sample_token'] for b in record['boxes']), 'Wrong box sample identity')
        total_boxes += len(record['boxes'])
    require(total_boxes == em['selected_boxes'], 'Extracted box count differs')
    # Exactly once on fresh raw records. No GT, selection, score or size changes.
    records = [origin.adapt_record(record) for record in records]
    training = base / 'server_results/training/connected_motion_train_v2'
    require(sha(training / 'complete.json') == history.D_COMPLETE and
            sha(base / 'connected_motion_protocol_v2.json') == history.D_PROTOCOL, 'D fixed-final source changed')
    done = read(training / 'complete.json'); manifest = read(training / 'manifest.json')
    protocol = read(base / 'connected_motion_protocol_v2.json')
    require(done['status'] == 'COMPLETE_CONNECTED_MOTION_TRAINING' and
            (done['updates'], done['examples'], done['evaluated_samples']) == (512, 2048, 200), 'Incomplete D')
    history.small_chain(training, done)
    require(manifest['sources']['protocol_sha256'] == history.D_PROTOCOL and
            manifest['sources']['sources_sha256'] == protocol['sources_sha256'], 'D source/protocol mismatch')
    for name, digest in manifest['sources']['sources_sha256'].items():
        require(sha(history.source(name)) == digest, 'D source changed: ' + name)
    ar = training / 'runs/D'; ac = read(ar / 'complete.json')
    require(sha(ar / 'complete.json') == done['arm_complete_sha256']['D'], 'D arm ledger changed')
    history.small_chain(ar, ac, skip=('final.pth',))
    require(done['final_checkpoints']['D']['sha256'] == ac['files_sha256']['final.pth'], 'D checkpoint pointer changed')
    actual = base / 'server_results/training/common_connected_motion_dev200_v2'
    actual_done = read(actual / 'complete.json'); history.small_chain(actual, actual_done)
    proof = read(actual / 'loaded_models.json')['arms']['D']['training_loader_receipt']
    require(actual_done['status'] == 'COMPLETE_FINAL_DEVELOPMENT_EVALUATION' and
            actual_done['training_complete_sha256'] == history.D_COMPLETE and
            proof['checkpoint_sha256'] == done['final_checkpoints']['D']['sha256'] and
            proof['actual_motion_state_sha256'] == done['final_checkpoints']['D']['motion_state_sha256'] and
            proof['actual_optimizer_parameter_steps_all512'] and proof['fixed_final_update'] == 512,
            'Recorded actual D load proof differs')
    sparse = base / 'sparse_motion_v1'; rawroot = base / 'motion_targets_v1'
    for path, digest in ((sparse/'manifest.json', history.SPARSE_MANIFEST),
                         (sparse/'complete.json', history.SPARSE_COMPLETE),
                         (rawroot/'manifest.json', history.RAW_MANIFEST),
                         (rawroot/'complete.json', history.RAW_COMPLETE)):
        require(sha(path) == digest, 'Original labels changed: ' + str(path))
    sm, sc, rm, rc = (read(p) for p in (sparse/'manifest.json', sparse/'complete.json',
                                       rawroot/'manifest.json', rawroot/'complete.json'))
    require(sc['manifest_sha256'] == history.SPARSE_MANIFEST and sc['raw_manifest_sha256'] == history.RAW_MANIFEST and
            sc['raw_complete_sha256'] == history.RAW_COMPLETE and rc['manifest_sha256'] == history.RAW_MANIFEST and
            protocol['labels'] == dict(manifest_sha256=history.SPARSE_MANIFEST, complete_sha256=history.SPARSE_COMPLETE),
            'Original label chain differs')
    require(sha(history.source('build_sparse_motion_supervision_v1.py')) == sc['script_sha256'] and
            sc['geometry_sha256'] == BOUND['motion_geometry.py'], 'Label geometry changed')
    descriptors = [r for r in sm['records'] if r['identity']['split'] == 'development']
    require([d['identity'] for d in descriptors] == selection, 'Sparse order differs')
    rawdesc = {r['identity']['sample_token']: r for r in rm['records']}
    Drows = [r for r in history.jsonl(training/'development_objects.jsonl') if r['arm'] == 'D']
    require(len(Drows) == 16074, 'D complete support count differs')
    Dmap = {(r['sample_token'], r['instance_token'], r['horizon_seconds']): r for r in Drows}
    require(len(Dmap) == len(Drows), 'Duplicate D object-horizon')
    sources = dict(analysis_sha256=sha(__file__), dependencies_sha256=BOUND, selection_sha256=SELECTION_SHA,
        origin_runtime_proof_sha256=ORIGIN_PROOF_SHA, origin_contract_sha256=ORIGIN_CONTRACT_SHA,
        origin_correction='raw bottom-face centroid + R_export[:,2]*height/2; applied once on deep copies; no GT or fitting',
        extraction_complete_sha256=sha(extraction_root/'complete.json'), extraction_files_sha256=ec['files_sha256'],
        original_export_sha256=EXPORT_SHA, extraction_manifest=em, extraction_meta=pred['meta'],
        D_complete_sha256=history.D_COMPLETE, D_protocol_sha256=history.D_PROTOCOL,
        D_checkpoint_declaration=done['final_checkpoints']['D'], D_actual_load_complete_sha256=sha(actual/'complete.json'),
        D_loaded_models_sha256=sha(actual/'loaded_models.json'), D_objects_sha256=sha(training/'development_objects.jsonl'),
        sparse_manifest_sha256=history.SPARSE_MANIFEST, sparse_complete_sha256=history.SPARSE_COMPLETE,
        raw_manifest_sha256=history.RAW_MANIFEST, raw_complete_sha256=history.RAW_COMPLETE,
        authentication_scope='Existing actual D tensor-load receipt; no new checkpoint read; full CRN export authenticated by extraction receipt')
    return helper, stats, selection, records, descriptors, rawdesc, Drows, Dmap, sources


def subset_stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(np.isfinite(a).all(), 'Nonfinite subset error')
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None)


def coverage_summary(rows):
    output = []
    for horizon in HORIZONS:
        for group in ('all', *GROUPS):
            chosen = [r for r in rows if r['horizon_seconds'] == horizon and (group == 'all' or r['group'] == group)]
            points = sum(r['source_points'] for r in chosen); covered = sum(r['covered_points'] for r in chosen)
            item = dict(horizon_seconds=horizon, group=group, object_anchor_pairs=len(chosen),
                source_points=points, covered_points=covered, uncovered_points=points-covered,
                point_coverage=covered/points if points else None,
                objects_any_covered=sum(r['covered_points'] > 0 for r in chosen),
                objects_fully_covered=sum(r['covered_points'] == r['source_points'] for r in chosen),
                objects_uncovered=sum(r['covered_points'] == 0 for r in chosen),
                objects_partly_covered=sum(0 < r['covered_points'] < r['source_points'] for r in chosen), subsets={})
            for part in ('covered', 'uncovered'):
                subset = [r[part] for r in chosen if r[part]['source_points'] > 0]
                item['subsets'][part] = dict(object_anchor_pairs=len(subset),
                    source_points=sum(r['source_points'] for r in subset),
                    metrics={m: {d: subset_stats([r[m][d] for r in subset]) for d in ('xy_m', 'xyz_m')}
                             for m in ('CRN_CV', 'zero')})
            output.append(item)
    return output


def run(args):
    start = time.monotonic(); base = Path(args.root).resolve(); extraction = Path(args.predictions_root).resolve()
    helper, stats, selection, records, descriptors, rawdesc, Drows, Dmap, sources = authenticate(
        base, Path(args.selection), extraction)
    geo = bound('motion_geometry.py')
    points = geo.voxel_centers_xyz(helper.SHAPE, helper.EXTENT).reshape(-1, 3)
    objects, coverage, anchors, input_sources, expected = [], [], [], [], set()
    for ordinal, (identity, record, desc) in enumerate(zip(selection, records, descriptors)):
        require(time.monotonic()-start < args.max_seconds, 'CPU time ceiling exceeded')
        token = identity['sample_token']; rd = rawdesc[token]
        require(rd['identity'] == identity and desc['label_source_sha256'] == rd['sha256'], 'Raw/sparse source differs')
        rawpath = base/'motion_targets_v1'/rd['file']
        require(rawpath.stat().st_size == rd['bytes'] and sha(rawpath) == rd['sha256'], 'Raw compressed bytes differ')
        rawbytes = gzip.decompress(rawpath.read_bytes())
        require(hashlib.sha256(rawbytes).hexdigest() == rd['uncompressed_json_sha256'], 'Raw JSON bytes differ')
        raw = json.loads(rawbytes)
        require(raw['identity'] == identity and raw['ordinal'] == desc['ordinal'] and
                raw['selection_sha256'] == SELECTION_SHA, 'Raw identity/order differs')
        frames = raw['frames']
        require([f['relative_frame_index'] for f in frames] == list(range(-2, 5)) and
                frames[2]['sample_token'] == token, 'Raw frame contract differs')
        label = helper.load_sparse(base/'sparse_motion_v1', desc, identity)
        # Only t0 pose, predicted state and physical query positions cross here.
        velocity, owner = predicted_velocity_field(record['boxes'], frames[2]['lidar_to_global_column_matrix'],
                                                   points[label['source_flat_indices']])
        displacement = np.asarray([velocity*h for h in HORIZONS], dtype=np.float64)
        require(np.isfinite(displacement).all(), 'Nonfinite CV field')
        anchors.append(dict(ordinal=ordinal, identity=identity, original_prediction_boxes=len(record['boxes']),
            GMO_prediction_boxes=sum(b['detection_name'] in GMO for b in record['boxes']),
            source_points=len(owner), covered_points=int((owner >= 0).sum()),
            boxes_owning_source_points=len(set(owner[owner >= 0].tolist())),
            owners_original_export_index_sha256=hashlib.sha256(owner.tobytes()).hexdigest()))
        input_sources.append(dict(identity=identity, sparse_file=desc['file'], sparse_sha256=desc['sha256'],
            raw_file=rd['file'], raw_sha256=rd['sha256'], raw_uncompressed_sha256=rd['uncompressed_json_sha256']))
        for h, horizon in enumerate(HORIZONS):
            dt = (frames[h+3]['timestamp_us']-frames[2]['timestamp_us'])/1e6
            require(dt == float(label['dt_future_seconds'][h]), 'Target time differs')
            for k, instance in enumerate(label['instance_tokens'].tolist()):
                own = label['object_index'] == k; valid = label['valid'][h] & own
                require(np.count_nonzero(valid) == (np.count_nonzero(own) if label['object_future_valid'][h, k] else 0),
                        'Partial object target validity differs')
                if not label['object_future_valid'][h, k]:
                    continue
                key = (token, instance, horizon); require(key in Dmap, 'Missing D object support'); expected.add(key)
                dr = Dmap[key]; target = label['target_displacement_m'][h, valid].astype(np.float64)
                delta = displacement[h, valid] - target
                err = np.linalg.norm(delta, axis=1); errxy = np.linalg.norm(delta[:, :2], axis=1)
                zero = np.linalg.norm(target, axis=1); zeroxy = np.linalg.norm(target[:, :2], axis=1)
                n = int(valid.sum()); group = GROUPS[int(label['object_speed_group'][h, k])]
                require(dr['scene_token'] == identity['scene_token'] and dr['source_points'] == n and
                        dr['group'] == group and dr['dt_seconds'] == dt and
                        dr['zero_epe_xy_m'] == float(zeroxy.mean()) and dr['zero_epe_3d_m'] == float(zero.mean()),
                        'D source/label/group/zero-reference differs')
                common = dict(sample_token=token, scene_token=identity['scene_token'], instance_token=instance,
                    horizon_seconds=horizon, dt_seconds=dt, group=group, source_points=n,
                    zero_epe_xy_m=float(zeroxy.mean()), zero_epe_3d_m=float(zero.mean()))
                objects.append(dict(common, arm='CRN_CV', epe_xy_m=float(errxy.mean()), epe_3d_m=float(err.mean())))
                covered = owner[valid] >= 0
                row = dict(common, covered_points=int(covered.sum()))
                for part, mask in (('covered', covered), ('uncovered', ~covered)):
                    count = int(mask.sum())
                    row[part] = dict(source_points=count,
                        CRN_CV=dict(xy_m=float(errxy[mask].mean()) if count else None,
                                    xyz_m=float(err[mask].mean()) if count else None),
                        zero=dict(xy_m=float(zeroxy[mask].mean()) if count else None,
                                  xyz_m=float(zero[mask].mean()) if count else None))
                require(row['covered']['source_points'] + row['uncovered']['source_points'] == n,
                        'Coverage partition differs')
                coverage.append(row)
    require(expected == set(Dmap) and len(objects) == 16074, 'Missing/extra original object-horizon support')
    grouped, keys = stats.physical_rows([*Drows, *objects], {r['sample_token']: r['scene_token'] for r in selection}, ARMS)
    # Bind every original object and point count, not only agreement between arms.
    support_rows = [dict(r, sparse_label_sha256=d['sha256'], raw_label_sha256=d['label_source_sha256'])
                    for r, d in zip(selection, descriptors)]
    support = stats.check_manifest(support_rows, grouped, base/'sparse_motion_v1/manifest.json')
    scenes = sorted({r['scene_token'] for r in selection})
    result = stats.aggregate_physical(grouped, keys, ARMS, scenes, PAIRS, repetitions=10000, seed=11)
    require(time.monotonic()-start < args.max_seconds, 'CPU time ceiling exceeded after aggregation')
    require('torch' not in sys.modules, 'CPU-only diagnostic imported torch')
    result.update(schema='predicted-object-state-cv-diagnostic-v2', status='COMPLETE_READ_ONLY_DIAGNOSTIC',
        elapsed_seconds=time.monotonic()-start, samples=200, scenes=100, scene_tokens=scenes,
        arms=list(ARMS), object_horizon_rows_per_arm=len(keys), support=support,
        prediction_contract=dict(classes=list(GMO), score_threshold=None, box_size='wlh; local xyz=lwh',
            box_origin='source-confirmed raw bottom centroid converted to geometric centre along full quaternion up axis',
            overlap='highest original detection_score, exact tie first export order', uncovered_velocity='zero',
            global_velocity='[vx,vy,0] metres/second', vector_transform='inverse(G0) rotation only',
            prediction_horizon_seconds=list(HORIZONS), future_pose_or_dt_in_prediction=False,
            GT_association_in_prediction=False, query_semantics='restriction of a GT-independent whole-space field'),
        interpretation=dict(historical_development_exposure=True, official_CRN_has_four_keyframe_history=True,
            same_sensor_input_or_capacity_as_D_claim=False, no_O_native_flow=True,
            no_occupancy_improvement_or_goal_success_claim=True, rigid_box_proxy_not_observed_scene_flow=True,
            subset_D_EPE_available=False, zero_denominator='null, never zero error',
            coverage_subset_aggregation='mean over subset points within each object, then equal nonempty anchor-instance weight',
            bootstrap='10000 paired whole-scene draws seed11; mean differences only; unadjusted',
            training_seed_uncertainty_measured=False, model_or_threshold_selected=False),
        model_forward_performed=False, GPU_used=False, optimizer_updates=0,
        sources=sources, input_sources=input_sources, per_anchor_coverage=anchors,
        coverage=coverage_summary(coverage), coverage_object_records=coverage,
        physical_object_records=[*Drows, *objects])
    return result


def markdown(result):
    lines = ['# CRN 预测对象状态的同源点 CV 诊断：修正框中心约定', '',
        '已按实际源码链将原始导出的底面中心转换为几何中心：translation + R_export[:,2]*(height/2)。仅这一确定的坐标语义适配改变；没有利用 GT 拟合偏移，没有更改框尺寸/分数/速度、预测时域或原评价支持。v1 原始结果保留，不能把坐标修正当成新模型收益。', '',
        '这是已暴露 dev200/100 场景的 CPU 诊断，不是新的训练结果或占据候选。CRN 的历史输入、预训练与模型容量不同，不能把整模差异归因于融合机制。', '',
        '预测只使用当前原始导出框/速度、t0 位姿和查询坐标，无新分数阈值、无 GT 匹配筛选；框外速度置零，全部原合法源点保留。位移乘名义时域，真实 future dt 仅随评价标签记录。原 sparse 框刚体代理不是直接观测 scene flow；D 来自原完整逐对象记录，本次没有重载权重。', '',
        '| h(s) | group | objects | D XY mean/median/p90 | CRN-CV XY mean/median/p90 | CV−D mean CI95 (m) |',
        '|---:|---|---:|---|---|---|']
    def number(value):
        return '—' if value is None else f'{value:.6f}'
    for row in result['physical']:
        m = row['metrics']['epe_xy_m']; c = m['comparisons']['CRN_CV-minus-D']
        cell = lambda arm: '/'.join(number(m['values'][arm][k]) for k in ('mean', 'median', 'p90'))
        lines.append(f"| {row['horizon_seconds']} | {row['group']} | {row['object_anchor_pairs']} | {cell('D')} | {cell('CRN_CV')} | {number(c['difference'])} [{number(c['lower95'])}, {number(c['upper95'])}] |")
    lines += ['', '全部 XY/XYZ、零位移、覆盖与未覆盖的分组统计和逐对象记录保存在 JSON。覆盖子集没有 D 逐点预测，未伪造其子集 EPE。未覆盖点的零速错误仍计入主结果；该物理评价不能衡量原 GT 支持外预测框带来的全部误报。场景区间不是训练 seed 稳定性，mAVE 没有被乘时间冒充未来 EPE。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--root', default=str(HERE))
    parser.add_argument('--predictions-root', default=str(HERE/'crn_state_dev200_v1'))
    parser.add_argument('--selection', default=str(HERE.parent/'m0_improvement_20260915/selection_v1.json'))
    parser.add_argument('--out', help='New output directory; real scoring requires reviewed source and completed extraction')
    parser.add_argument('--max-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    require(args.out is not None, '--out required for real scoring')
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 600, 'Bounded CPU time required')
    out = Path(args.out); require(not out.exists(), 'Never overwrite a previous diagnostic')
    result = run(args)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (out/'report.md').write_text(markdown(result))
    complete = dict(schema='predicted-object-state-cv-complete-v2', status=result['status'], samples=200, scenes=100,
                    files_sha256={name: sha(out/name) for name in ('summary.json', 'report.md')}, script_sha256=sha(__file__))
    (out/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], out=str(out), summary_sha256=sha(out/'summary.json'))))


if __name__ == '__main__':
    main()
