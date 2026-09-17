"""CPU-only raw nuScenes box chains, separate from all model/cache inputs.

Reads only enumerated original JSON tables, installed official split literals,
and the fixed selection. No dataset pipeline, SDK import, pkl, image, radar,
occupancy, checkpoint, prediction, optimizer, model or GPU access.
"""
import argparse
import ast
import codecs
from collections import Counter, defaultdict
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import time

SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
SCHEMA = 'raw-nuscenes-motion-target-v1'
CATEGORY_TO_FINE = {
    'vehicle.bicycle': 2, 'vehicle.bus.bendy': 3, 'vehicle.bus.rigid': 3,
    'vehicle.car': 4, 'vehicle.construction': 5, 'vehicle.motorcycle': 6,
    'human.pedestrian.adult': 7, 'human.pedestrian.child': 7,
    'human.pedestrian.construction_worker': 7, 'human.pedestrian.police_officer': 7,
    'vehicle.trailer': 9, 'vehicle.truck': 10,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def table(path, ledger):
    """Stream a JSON array, SHA all bytes, retain only requested records upstream."""
    before = path.stat()
    hasher = hashlib.sha256(); decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder('utf-8')()
    buffer = ''; pos = 0; eof = False; count = 0
    with path.open('rb') as stream:
        def fill():
            nonlocal buffer, pos, eof
            chunk = stream.read(1024*1024)
            hasher.update(chunk); eof = not chunk
            buffer = buffer[pos:] + utf8.decode(chunk, final=eof); pos = 0
            require(len(buffer) <= 32*1024*1024, 'Single metadata object exceeds buffer cap')
        def skip_space():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace(): pos += 1
                if pos < len(buffer) or eof: return
                fill()
        fill(); skip_space()
        require(pos < len(buffer) and buffer[pos] == '[', 'Expected JSON table array')
        pos += 1; first = True
        while True:
            skip_space()
            require(pos < len(buffer), 'Truncated JSON table')
            if buffer[pos] == ']':
                pos += 1; skip_space()
                require(pos == len(buffer) and eof, 'Trailing JSON bytes')
                break
            if not first:
                require(buffer[pos] == ',', 'Missing JSON object delimiter')
                pos += 1; skip_space()
            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos); pos = end; break
                except json.JSONDecodeError:
                    if eof: raise
                    fill()
            require(isinstance(obj, dict), 'Non-object metadata entry')
            first = False; count += 1
            yield obj
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Metadata changed during read')
    ledger[str(path)] = dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                             sha256=hasher.hexdigest(), records_read=count, all_file_bytes_hashed=True)


def quaternion_matrix(q):
    """wxyz rotation, original quaternion remains in target without modification."""
    require(len(q) == 4 and all(math.isfinite(v) for v in q), 'Invalid quaternion')
    norm = math.sqrt(sum(v*v for v in q))
    require(abs(norm-1) < 1e-3, 'Non-unit raw quaternion')
    w, x, y, z = (v/norm for v in q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def lidar_to_global(ego, calibrated):
    e = quaternion_matrix(ego['rotation']); l = quaternion_matrix(calibrated['rotation'])
    out = [[sum(e[i][k]*l[k][j] for k in range(3)) for j in range(3)] +
           [sum(e[i][k]*calibrated['translation'][k] for k in range(3))+ego['translation'][i]] for i in range(3)]
    return out + [[0., 0., 0., 1.]]


def official_splits(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('train_detect', 'train_track', 'val'):
                values[name] = ast.literal_eval(node.value)
    train = set(values['train_detect']) | set(values['train_track']); val = set(values['val'])
    require(len(train) == 700 and len(val) == 150 and not train & val, 'Official split source unexpected')
    return {'train': train, 'development': val}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('selection', 'data-root', 'split-source', 'out'):
        parser.add_argument('--'+arg, required=True)
    parser.add_argument('--max-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    require(0 < args.max_seconds <= 600, 'CPU deadline must be at most600s')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU extraction deadline')))
    signal.alarm(args.max_seconds)
    started = time.monotonic(); script_sha = sha(__file__)
    selection_path = Path(args.selection).resolve(); split_path = Path(args.split_source).resolve()
    require(sha(selection_path) == SELECTION_SHA, 'Wrong frozen selection')
    selection = json.loads(selection_path.read_text()); records = selection['records']
    require(len(records) == 712 and len({r['sample_token'] for r in records}) == 712, 'Require all712 unique anchors')
    require(Counter(r['split'] for r in records) == {'train': 512, 'development': 200}, 'Wrong training/development counts')
    require([r['split'] for r in records] == ['train']*512+['development']*200, 'Frozen order changed')
    split_sha = sha(split_path); official = official_splits(split_path)
    root = Path(args.data_root).resolve(); tables = root/'v1.0-trainval'; out = Path(args.out).resolve()
    require(out != root and root not in out.parents, 'Targets must be physically separate from data root')
    out.mkdir(parents=True, exist_ok=False)
    ledger = {}; descriptors = []; stats = {'by_split': {}}
    try:
        def load(name):
            return {r['token']: r for r in table(tables/(name+'.json'), ledger)}
        samples = load('sample'); scenes = load('scene'); categories = load('category')
        sensors = load('sensor'); calibrated = load('calibrated_sensor')
        attrs = load('attribute'); visibility = load('visibility'); instances = load('instance')
        sequences = []; wanted = set(); scene_sets = defaultdict(set)
        for row in records:
            s = samples[row['sample_token']]
            require(s['scene_token'] == row['scene_token'], 'Sample/scene mismatch')
            require(scenes[s['scene_token']]['name'] in official[row['split']], 'Official train/val mismatch')
            scene_sets[row['split']].add(row['scene_token']); seq = [s]
            for _ in range(4):
                require(bool(seq[-1]['next']), 'Missing future sample')
                n = samples[seq[-1]['next']]; require(n['prev'] == seq[-1]['token'], 'Non-reciprocal sample chain'); seq.append(n)
            for _ in range(2):
                require(bool(seq[0]['prev']), 'Missing history sample')
                n = samples[seq[0]['prev']]; require(n['next'] == seq[0]['token'], 'Non-reciprocal sample chain'); seq.insert(0, n)
            require(all(z['scene_token'] == row['scene_token'] for z in seq), 'Cross-scene sample chain')
            require(all(a['timestamp'] < b['timestamp'] for a,b in zip(seq,seq[1:])), 'Non-monotone actual timestamps')
            sequences.append(seq); wanted.update(z['token'] for z in seq)
        require(len(scene_sets['train']) == 256 and len(scene_sets['development']) == 100 and
                not scene_sets['train'] & scene_sets['development'], 'Scene split leakage')
        lidar = {}
        for sd in table(tables/'sample_data.json', ledger):
            if sd['sample_token'] in wanted and sd['is_key_frame']:
                cs = calibrated[sd['calibrated_sensor_token']]
                if sensors[cs['sensor_token']]['channel'] == 'LIDAR_TOP':
                    require(sd['sample_token'] not in lidar, 'Duplicate keyframe LiDAR'); lidar[sd['sample_token']] = sd
        require(set(lidar) == wanted, 'Missing keyframe LiDAR')
        wanted_ego = {r['ego_pose_token'] for r in lidar.values()}
        ego = {r['token']: r for r in table(tables/'ego_pose.json', ledger) if r['token'] in wanted_ego}
        require(set(ego) == wanted_ego, 'Missing ego poses')
        annotations = defaultdict(dict); ignored_categories = Counter(); instance_scene = {}
        for ann in table(tables/'sample_annotation.json', ledger):
            if ann['sample_token'] not in wanted: continue
            inst = instances[ann['instance_token']]; cat = categories[inst['category_token']]['name']
            if cat not in CATEGORY_TO_FINE:
                ignored_categories[cat] += 1; continue
            inst_token = ann['instance_token']; sample_token = ann['sample_token']; scene_token = samples[sample_token]['scene_token']
            require(inst_token not in annotations[sample_token], 'Duplicate instance in one sample')
            require(instance_scene.get(inst_token, scene_token) == scene_token, 'Instance crosses scenes')
            instance_scene[inst_token] = scene_token
            for key, length in [('translation',3), ('size',3), ('rotation',4)]:
                require(len(ann[key]) == length and all(math.isfinite(v) for v in ann[key]), 'Invalid raw box values')
            require(all(v > 0 for v in ann['size']), 'Invalid box dimensions'); quaternion_matrix(ann['rotation'])
            require(ann['visibility_token'] in visibility or ann['visibility_token'] == '', 'Unknown visibility')
            require(all(t in attrs for t in ann['attribute_tokens']), 'Unknown attributes')
            require(all(type(ann[k]) is int and ann[k] >= 0 for k in ('num_lidar_pts','num_radar_pts')), 'Invalid point count')
            annotations[sample_token][inst_token] = dict(ann)
        print(json.dumps(dict(event='METADATA_READ', seconds=time.monotonic()-started, selected_frame_count=len(wanted))), flush=True)
        for split in ('train','development'):
            (out/split).mkdir()
            stats['by_split'][split] = dict(anchors=0, scenes=len(scene_sets[split]), anchor_instance_pairs=0,
                valid_box_frames=0, missing_box_slots=0, born_after_t0=0, absent_at_t0=0,
                disappeared_before_last=0, complete_7_frames=0, no_GMO_anchors=0, zero_sensor_points_box_frames=0,
                category_instance_pairs={}, visibility_valid_box_frames={})
        for ordinal,(row,seq) in enumerate(zip(records,sequences)):
            ss = stats['by_split'][row['split']]; ss['anchors'] += 1
            instance_tokens = sorted(set().union(*(annotations[s['token']] for s in seq)))
            ss['anchor_instance_pairs'] += len(instance_tokens); ss['no_GMO_anchors'] += not bool(instance_tokens)
            frames = []
            for offset,s in enumerate(seq):
                sd = lidar[s['token']]; ep = ego[sd['ego_pose_token']]; cs = calibrated[sd['calibrated_sensor_token']]
                frames.append(dict(sequence_index=offset, relative_frame_index=offset-2, sample_token=s['token'],
                    scene_token=s['scene_token'], timestamp_us=s['timestamp'],
                    dt_seconds=(s['timestamp']-seq[2]['timestamp'])/1e6,
                    lidar_sample_data_token=sd['token'], lidar_timestamp_us=sd['timestamp'],
                    lidar_sample_timestamp_delta_seconds=(sd['timestamp']-s['timestamp'])/1e6,
                    calibrated_sensor_token=sd['calibrated_sensor_token'], ego_pose_token=sd['ego_pose_token'],
                    ego_to_global_translation=ep['translation'], ego_to_global_quaternion_wxyz=ep['rotation'],
                    lidar_to_ego_translation=cs['translation'], lidar_to_ego_quaternion_wxyz=cs['rotation'],
                    lidar_to_global_column_matrix=lidar_to_global(ep,cs)))
            tracks = []
            for token in instance_tokens:
                cat = categories[instances[token]['category_token']]['name']
                raw = [annotations[s['token']].get(token) for s in seq]; valid = [a is not None for a in raw]
                ss['valid_box_frames'] += sum(valid); ss['missing_box_slots'] += 7-sum(valid)
                ss['born_after_t0'] += not any(valid[:3]); ss['absent_at_t0'] += not valid[2]
                ss['disappeared_before_last'] += any(valid[:3]) and not valid[-1]; ss['complete_7_frames'] += all(valid)
                ss['category_instance_pairs'][cat] = ss['category_instance_pairs'].get(cat,0)+1
                for a in raw:
                    if a is not None:
                        k = a['visibility_token']; ss['visibility_valid_box_frames'][k] = ss['visibility_valid_box_frames'].get(k,0)+1
                        ss['zero_sensor_points_box_frames'] += a['num_lidar_pts']+a['num_radar_pts'] == 0
                tracks.append(dict(instance_token=token, category_name=cat, fine_semantic_class=CATEGORY_TO_FINE[cat],
                    gmo_class=1, valid_mask=valid,
                    annotation_tokens=[None if a is None else a['token'] for a in raw],
                    global_centers_m=[None if a is None else a['translation'] for a in raw],
                    sizes_wlh_m=[None if a is None else a['size'] for a in raw],
                    global_rotations_wxyz=[None if a is None else a['rotation'] for a in raw],
                    visibility_tokens=[None if a is None else a['visibility_token'] for a in raw],
                    num_lidar_pts=[None if a is None else a['num_lidar_pts'] for a in raw],
                    num_radar_pts=[None if a is None else a['num_radar_pts'] for a in raw],
                    attribute_tokens=[None if a is None else a['attribute_tokens'] for a in raw],
                    attribute_names=[None if a is None else [attrs[t]['name'] for t in a['attribute_tokens']] for a in raw],
                    annotation_prev_tokens=[None if a is None else a['prev'] for a in raw],
                    annotation_next_tokens=[None if a is None else a['next'] for a in raw]))
            value = dict(schema=SCHEMA, selection_sha256=SELECTION_SHA, ordinal=ordinal, identity=row,
                coordinates='raw global centers/box rotations; column homogeneous LiDAR-to-global matrices; meters',
                frames=frames, tracks=tracks, missing_fill_applied=False, old_refine_applied=False,
                visibility_or_point_count_filter_applied=False, spatial_ROI_filter_applied=False,
                dynamic_speed_threshold_applied=False, instance_union='all GMO instances in seven original sample annotations',
                model_inputs_read=False, predictions_read=False)
            raw_bytes = (json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
            compressed = gzip.compress(raw_bytes,compresslevel=6,mtime=0)
            rel = row['split']+'/'+row['sample_token']+'.json.gz'; path=out/rel
            with path.open('xb') as stream: stream.write(compressed)
            require(json.loads(gzip.decompress(path.read_bytes())) == value, 'Serialized target differs')
            descriptors.append(dict(ordinal=ordinal,identity=row,file=rel,bytes=len(compressed),
                sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_json_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                instance_count=len(tracks),valid_box_frames=sum(sum(t['valid_mask']) for t in tracks)))
        for p, receipt in ledger.items():
            st=Path(p).stat(); require((st.st_size,st.st_mtime_ns)==(receipt['bytes'],receipt['mtime_ns']), 'Source changed after extraction')
        require(sha(selection_path)==SELECTION_SHA and sha(split_path)==split_sha and sha(__file__)==script_sha,'Frozen source changed')
        manifest=dict(schema='raw-nuscenes-motion-target-manifest-v1', status='COMPLETE_RAW_LABEL_EXTRACTION',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=script_sha,
            selection={'path':str(selection_path),'sha256':SELECTION_SHA},
            official_split_source={'path':str(split_path),'sha256':split_sha,'load_method':'AST literal lists, no SDK import'},
            data_root=str(root), metadata_sha256=ledger, category_to_fine_semantic_class=CATEGORY_TO_FINE,
            category_mapping_source='https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md',
            ignored_non_GMO_annotations_by_category=dict(ignored_categories),records=descriptors,
            samples=712,unique_frame_samples=len(wanted),train_dev_scenes_disjoint=True,all_official_scene_memberships_checked=True,
            inputs_and_labels_physically_separate=True,model_or_prediction_read=False,optimizer_steps=0,
            historical_validation_exposure=True,training_seed=11,training_performed=False,
            missing_fill=False,old_refine=False,visibility_filter=False,speed_or_ROI_filter=False)
        stats.update(seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     maximum_seconds=args.max_seconds,total_compressed_bytes=sum(r['bytes'] for r in descriptors),
                     repeated_box_frames_across_anchors_are_not_independent=True)
        write_json(out/'manifest.json',manifest); write_json(out/'statistics.json',stats)
        write_json(out/'complete.json',dict(schema='raw-nuscenes-motion-target-complete-v1',status='COMPLETE',
            samples=712,train_samples=512,development_samples=200,script_sha256=script_sha,
            selection_sha256=SELECTION_SHA,manifest_sha256=sha(out/'manifest.json'),statistics_sha256=sha(out/'statistics.json'),
            seconds=time.monotonic()-started,all712_files_roundtrip_verified=True,model_or_prediction_read=False,optimizer_steps=0))
        signal.alarm(0)
        print(json.dumps(dict(event='COMPLETE',out=str(out),seconds=time.monotonic()-started,
                              manifest_sha256=sha(out/'manifest.json'),statistics=stats)),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-started,
                                        model_or_prediction_read=False,optimizer_steps=0))
        raise


if __name__ == '__main__':
    main()
