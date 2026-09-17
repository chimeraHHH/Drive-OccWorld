"""Frozen O versus persistence and privileged future-observation readout.

Teacher images/radar are used ONLY in the separately identified diagnostic arm.
No optimizer, student fit, future label input, or physical motion claim.
All outputs are scored in original t0 LiDAR coordinates with original fine GT.
Future observation logits are SE(3)-resampled after the shared O readout: this
avoids pretending that 256 BEV channels have an explicit metric height axis.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
O_SHA = 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
CONFIG_SHA = 'c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'
INDEX_SHA = '1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1'
NATIVE_SHA = '41953909ddfdfd08d51d98385a450640e5314b775ab01458cb55ff1b18e4bfb0'
EXTENT = (-51.2, -51.2, -5., 51.2, 51.2, 3.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), allow_nan=False), flush=True)


def xyz_logits(model, state):
    """The unchanged final O token readout, in B,C,X,Y,Z order."""
    out = model.future_pred_head.bev_pred_head[-1](state)
    assert out.shape == (1, 40000, 32)
    return out.reshape(1, 200, 200, 16, 2).permute(0, 4, 2, 1, 3).contiguous()


def resample(source, reference_to_source):
    import torch
    import torch.nn.functional as F
    shape = source.shape[-3:]
    lo = source.new_tensor(EXTENT[:3])
    size = source.new_tensor(EXTENT[3:]) - lo
    axes = [lo[k] + (torch.arange(shape[k], device=source.device) + .5) * size[k] / shape[k]
            for k in range(3)]
    points = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    transformed = homogeneous @ source.new_tensor(reference_to_source)
    coords = transformed[..., :3]
    grid = 2 * (coords - lo) / size - 1
    # Valid interpolation support uses voxel centers, not merely grid extent.
    half_cell = size / source.new_tensor(shape) / 2
    valid = ((coords >= lo + half_cell - 1e-5) &
             (coords <= lo + size - half_cell + 1e-5)).all(-1)
    result = F.grid_sample(source, grid[..., [2, 1, 0]].unsqueeze(0),
                           mode='bilinear', padding_mode='border', align_corners=False)
    return result, valid


def geometry_checks(device):
    import numpy as np
    import torch
    x = torch.arange(200, device=device).float()[:, None, None].expand(200, 200, 16)
    y = torch.arange(200, device=device).float()[None, :, None].expand(200, 200, 16)
    z = torch.arange(16, device=device).float()[None, None, :].expand(200, 200, 16)
    field = torch.stack((x, y, z), 0).unsqueeze(0)
    identity, valid = resample(field, np.eye(4))
    error = (identity-field).abs().max().item()
    assert error < 1e-4 and bool(valid.all()), ('identity', error)
    translation = np.eye(4); translation[3, 0] = .512
    shifted, valid = resample(field, translation)
    assert torch.allclose(shifted[0, 0, :-1], x[1:], rtol=0, atol=1e-4)
    assert bool(valid[:-1].all()) and not bool(valid[-1].any())
    rotation = np.eye(4); rotation[:2, :2] = [[0, 1], [-1, 0]]
    turned, _ = resample(field, rotation)
    assert torch.allclose(turned[0, 0], 199-y, rtol=0, atol=1e-4)
    assert torch.allclose(turned[0, 1], x, rtol=0, atol=1e-4)
    return dict(identity_max_abs=error, positive_x_translation_pass=True, quarter_turn_pass=True)


def observed_state(dataset, model, index, device):
    """Same camera history/current encoder as native, without label loading.

    Only fields consumed by that encoder are prepared; cached-anchor parity
    must pass before this input-only path is used on future observations.
    """
    import numpy as np
    import torch
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    assert dataset.queue_length == 2 and list(dataset.rand_frame_interval) == [1]
    rows = [dataset.data_infos[i] for i in range(index-2, index+1)]
    assert len({r['scene_token'] for r in rows}) == 1
    examples = [dataset._prepare_data_info_single(i, occ_load_flag=False)
                for i in range(index-2, index+1)]
    assert all(e is not None for e in examples)
    assert all('gt_occ' not in e and 'segmentation' not in e for e in examples)
    metas = {i: copy.deepcopy(e['img_metas'].data) for i, e in enumerate(examples)}
    last_pos = last_angle = None
    for i, meta in metas.items():
        pos, angle = copy.deepcopy(meta['can_bus'][:3]), copy.deepcopy(meta['can_bus'][-1])
        meta['can_bus'] = copy.deepcopy(meta['can_bus'])
        meta['prev_bev_exists'] = i > 0
        meta['can_bus'][:3] = 0 if i == 0 else pos-last_pos
        meta['can_bus'][-1] = 0 if i == 0 else angle-last_angle
        last_pos, last_angle = pos, angle
    images = torch.stack([e['img'].data for e in examples]).unsqueeze(0).to(device)
    radar = torch.from_numpy(dataset.radar_bev_loader(rows[-1]['token'])).unsqueeze(0).to(device)
    assert radar.dtype == torch.float32
    with torch.no_grad():
        previous, _ = model.obtain_history_bev(images[:, :-1], [metas])
        state = model.obtain_ref_bev(images[:, -1], [metas[2]], previous, radar_bev=radar)
    current = rows[-1]
    lidar_to_global = (transform_matrix(current['ego2global_translation'], Quaternion(current['ego2global_rotation'])) @
                       transform_matrix(current['lidar2ego_translation'], Quaternion(current['lidar2ego_rotation'])))
    return state, dict(sample_tokens=[r['token'] for r in rows],
                       timestamps_us=[int(r['timestamp']) for r in rows],
                       scene_token=current['scene_token'], labels_loaded=False,
                       lidar_to_global_column=lidar_to_global.tolist())


def histograms(logits, targets, masks=None):
    import torch
    import torch.nn.functional as F
    result = []
    for h in range(5):
        pred = F.interpolate(logits[h:h+1], size=targets.shape[-3:], mode='trilinear',
                             align_corners=False)[0].argmax(0)
        gt = targets[0, h+2].long()
        valid = (gt >= 0) & (gt < 2)
        if masks is not None:
            support = F.interpolate(masks[h][None, None].float(), size=gt.shape,
                                    mode='trilinear', align_corners=False)[0, 0] >= 1-1e-6
            valid &= support
        result.append(torch.bincount(2*gt[valid]+pred[valid], minlength=4).reshape(2, 2).cpu().tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'o-checkpoint', 'cache', 'repo', 'helpers', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--anchors', type=int, default=16)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    started = time.monotonic()
    out = Path(args.out); out.mkdir(exist_ok=False)
    assert args.anchors in (2, 16)
    sys.path.insert(0, args.helpers)
    assert sha(Path(args.helpers)/'native_state_cache.py') == NATIVE_SHA
    import native_state_cache as native
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(11); np.random.seed(11)
    torch.cuda.set_per_process_memory_fraction(32*1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for path, digest in ((args.config, CONFIG_SHA), (args.checkpoint, M0_SHA),
                         (args.o_checkpoint, O_SHA), (Path(args.cache)/'index.json', INDEX_SHA)):
        assert sha(path) == digest, path
    # Geometry must use the same FP32 matmul policy as production resampling.
    torch.backends.cuda.matmul.allow_tf32 = False
    geometry = geometry_checks('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = True
    event('GEOMETRY_PASS', **geometry)
    model = native.build_native_model(args.config, args.checkpoint, 'cuda:0', repo=args.repo)
    checkpoint = torch.load(args.o_checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['arm'] == 'O' and checkpoint['update'] == 512
    model.future_pred_head.load_state_dict(checkpoint['future_pred_head'], strict=True)
    del checkpoint
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    initial_digest = native._parameter_digest(model)
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(args.config)
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = cfg.data.train.ann_file
    dataset_cfg.test_mode = True; dataset_cfg.pop('samples_per_gpu', None)
    assert dataset_cfg.radar_cfg.cache_readonly and dataset_cfg.radar_cfg.nsweeps == 5
    dataset = build_dataset(dataset_cfg)
    index = json.loads((Path(args.cache)/'index.json').read_text())
    rows = index['records'][:args.anchors]
    assert len(rows) == args.anchors
    save(out/'manifest.json', dict(source_sha256=sha(__file__), train_index_sha256=INDEX_SHA,
        anchor_tokens=[r['sample_token'] for r in rows], geometry_checks=geometry,
        training=False, optimizer_steps=0, privileged_teacher=True, seed=11,
        selection='first N original train cache records, no performance filtering',
        teacher_input='native historical/current cameras plus native current radar at each future timestamp',
        readout='same frozen final O token MLP', frame='original t0 LiDAR R',
        teacher_resampling='full SE3 output logits, trilinear; unsupported sites use t0 persistence',
        feature_correspondence_or_metric_motion_claim=False, max_allocated_gib=32))
    records = []
    for ordinal, row in enumerate(rows):
        assert time.monotonic()-started < args.max_seconds
        begin = time.monotonic()
        folder = Path(args.cache)/row['directory']
        descriptor = row['files']['inputs']; native._verify_file(folder/descriptor['file'], descriptor)
        inputs = torch.load(str(folder/descriptor['file']), map_location='cpu', weights_only=False)
        native.validate_inputs(inputs)
        inputs = native._tree_map(inputs, lambda value: value.to('cuda:0'))
        info_index = row['data_info_index']
        assert dataset.data_infos[info_index]['token'] == row['sample_token']
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            native_preds = native.replay(model, dict(inputs=inputs), training=False)[0]
            O = native_preds[:, -1, 0, 0].reshape(5, 200, 200, 16, 2).permute(0, 4, 2, 1, 3).contiguous()
            direct = xyz_logits(model, inputs['prev_bev_input'][:, -1])
        direct_error = (direct-O[:1]).abs().max().item()
        assert torch.allclose(direct, O[:1], rtol=2e-5, atol=2e-5), ('direct readout', direct_error)
        persistence = O[:1].expand_as(O).clone()
        teacher = persistence.clone()
        masks = torch.ones((5, 200, 200, 16), device='cuda:0', dtype=torch.bool)
        observations = []
        current_max_error = None
        for h in range(5):
            assert time.monotonic()-started < args.max_seconds
            torch.backends.cuda.matmul.allow_tf32 = True
            state, observation = observed_state(dataset, model, info_index+h, 'cuda:0')
            assert observation['scene_token'] == row['scene_token']
            if h == 0:
                current_max_error = (state-inputs['prev_bev_input'][:, -1]).abs().max().item()
                assert torch.allclose(state, inputs['prev_bev_input'][:, -1], rtol=1e-5, atol=1e-5), ('encoder parity', current_max_error)
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                measured = xyz_logits(model, state)
                transform = np.asarray(inputs['img_metas'][0]['ref2future_lidar_transform'][h])
                inverse = np.asarray(inputs['img_metas'][0]['future2ref_lidar_transform'][h])
                assert np.allclose(transform@inverse, np.eye(4), rtol=0, atol=1e-6)
                G = np.asarray(observation['lidar_to_global_column'])
                G0 = G if h == 0 else np.asarray(observations[0]['lidar_to_global_column'])
                assert np.allclose(transform, (np.linalg.inv(G)@G0).T, rtol=0, atol=1e-6), 'Teacher pose/token mismatch'
                warped, support = resample(measured, transform)
            if h == 0:
                assert torch.allclose(warped, direct, rtol=1e-4, atol=2e-4)
                # Avoid resampling/GEMM arithmetic changes in the common t0 arm.
                teacher[0] = O[0]
            else:
                teacher[h] = torch.where(support[None], warped[0], persistence[h])
                masks[h] = support
            observation.update(horizon_seconds=.5*h, supported_fraction=float(support.float().mean()),
                               actual_dt_seconds=(observation['timestamps_us'][-1]-dataset.data_infos[info_index]['timestamp'])/1e6)
            assert abs(observation['actual_dt_seconds']-.5*h) < .1
            observations.append(observation)
            del state, measured, warped
        # Targets are opened only after all forecast and teacher outputs exist.
        descriptor = row['files']['targets']; native._verify_file(folder/descriptor['file'], descriptor)
        targets = torch.from_numpy(np.load(folder/descriptor['file'], allow_pickle=False)).to('cuda:0', dtype=torch.long)
        native_hist = model.evaluate_occ_records(native_preds, targets, inputs['img_metas'])[0]['hist_by_horizon'].tolist()
        full = {name: histograms(logits, targets) for name, logits in
                (('O', O), ('persistence', persistence), ('future_observation', teacher))}
        assert full['O'] == native_hist
        common = {name: histograms(logits, targets, masks) for name, logits in
                  (('O', O), ('persistence', persistence), ('future_observation', teacher))}
        direct_all = O.clone(); direct_all[0] = direct[0]
        assert histograms(direct_all, targets)[0] == native_hist[0]
        record = dict(ordinal=ordinal, sample_token=row['sample_token'], scene_token=row['scene_token'],
            full_hist=full, common_support_hist=common, teacher_observations=observations,
            current_encoder_max_abs=current_max_error, direct_readout_max_abs=direct_error,
            native_O_confusion_exact=True, seconds=time.monotonic()-begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        with (out/'records.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False)+'\n')
        records.append(record)
        event('ANCHOR_DONE', ordinal=ordinal, seconds=record['seconds'], current_encoder_max_abs=current_max_error)
        del native_preds, O, persistence, teacher, masks, targets, direct_all, direct, inputs
    assert native._parameter_digest(model) == initial_digest
    summary = dict(status='COMPLETE_FROZEN_PRIVILEGED_READOUT_DIAGNOSTIC', anchors=len(records),
                   scenes=len({r['scene_token'] for r in records}), seconds=time.monotonic()-started,
                   frozen_parameters_unchanged=True, optimizer_steps=0, comparisons={})
    for population in ('full_hist', 'common_support_hist'):
        summary['comparisons'][population] = {}
        for arm in ('O', 'persistence', 'future_observation'):
            cm = np.asarray([r[population][arm] for r in records], dtype=np.int64).sum(0)
            future = cm[1:].sum(0)
            def values(c):
                return dict(GMO_IoU_percent=float(100*c[1,1]/(c[1,1]+c[1,0]+c[0,1])),
                            GMO_recall_percent=float(100*c[1,1]/c[1].sum()))
            per_horizon = [values(c) for c in cm]
            macro = {key: float(np.mean([v[key] for v in per_horizon[1:]])) for key in per_horizon[0]}
            summary['comparisons'][population][arm] = dict(future_macro=macro, future_pooled=values(future),
                per_horizon=per_horizon, pooled_confusion=cm.tolist())
    save(out/'complete.json', summary)
    event('COMPLETE', **summary)


if __name__ == '__main__':
    main()
