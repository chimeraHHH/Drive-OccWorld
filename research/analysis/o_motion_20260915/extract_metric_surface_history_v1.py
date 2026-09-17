"""Frozen Metric3D + RAFT: image-grid surface queries, history only, no GT.

This constructs a spatially queryable constant-velocity field from predicted
surfaces. It is a baseline, not a calibrated posterior or a future predictor
trained here. Invalid correspondences are retained as NaN and explicit masks.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import runpy
import types

import numpy as np

HISTORY_SHA = '040d62242a541b9685f3f39f1534b3663b0e1142b4cd7d9737cdbfd15d94314d'
WEIGHT_SHA = 'b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6'
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def sample(field, uv):
    """Bilinear sampling on closed original pixel centers; no extrapolation."""
    field = np.asarray(field)
    if field.ndim == 2:
        field = field[..., None]
    h, w, c = field.shape
    uv = np.asarray(uv, np.float64)
    valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    out = np.full((len(uv), c), np.nan, np.float64)
    q = uv[valid]
    x, y = np.floor(q).astype(np.int64).T
    x1, y1 = np.minimum(x+1, w-1), np.minimum(y+1, h-1)
    a, b = (q - np.column_stack([x, y])).T
    a, b = a[:, None], b[:, None]
    out[valid] = ((1-b)*((1-a)*field[y, x]+a*field[y, x1]) + b*((1-a)*field[y1, x]+a*field[y1, x1]))
    return out, valid


def lift(uv, depth, intrinsic, camera_to_ref):
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(intrinsic).T
    pc = rays * depth[:, None]
    return pc @ camera_to_ref[:3, :3].T + camera_to_ref[:3, 3]


def geometry(row, pair, uv, dcur, dpast, forward, reverse):
    dc, _ = sample(dcur, uv)
    f, _ = sample(forward, uv)
    past_uv = uv + f
    dp, inside = sample(dpast, past_uv)
    r, reverse_valid = sample(reverse, past_uv)
    fb = np.linalg.norm(f+r, axis=1)
    cur, past = pair['current'], pair['past']
    dt = (cur['sample_data']['timestamp'] - past['sample_data']['timestamp']) / 1e6
    assert dt > 0
    g0 = np.asarray(row['lidar_to_global'], np.float64)
    c2r = np.linalg.inv(g0) @ np.asarray(cur['camera_to_global'], np.float64)
    p2r = np.linalg.inv(g0) @ np.asarray(past['camera_to_global'], np.float64)
    pc = lift(uv, dc[:, 0], np.asarray(cur['K']), c2r)
    pp = lift(past_uv, dp[:, 0], np.asarray(past['K']), p2r)
    velocity = (pc-pp) / dt
    # Current exposure can precede/follow t0 slightly. Explicit CV registration
    # to t0, using only this estimated historical velocity, not future labels.
    dt_to_t0 = (row['t0_lidar_us']-cur['sample_data']['timestamp'])/1e6
    p0 = pc + dt_to_t0*velocity
    valid = inside & reverse_valid & np.isfinite(velocity).all(1) & (dc[:, 0] > 0) & (dc[:, 0] < 300) & (dp[:, 0] > 0) & (dp[:, 0] < 300)
    roi = valid & (np.abs(p0[:, :2]) < 51.2).all(1) & (p0[:, 2] > -5) & (p0[:, 2] < 3)
    return dict(uv=uv.astype(np.float32), past_uv=past_uv.astype(np.float32),
                depth_current_m=dc[:, 0].astype(np.float32), depth_past_m=dp[:, 0].astype(np.float32),
                point_current_R_m=pc.astype(np.float32), point_past_R_m=pp.astype(np.float32),
                point_t0_R_m=p0.astype(np.float32), velocity_R_mps=velocity.astype(np.float32),
                fb_error_px=fb.astype(np.float32), correspondence_valid=valid, in_roi=roi,
                fb1_in_roi=roi & (fb <= 1.0), camera_dt_s=np.asarray(dt), exposure_to_t0_s=np.asarray(dt_to_t0))


def analytic_checks():
    # Nontrivial camera translation + rotation, nonidentity reference, same 3D
    # static plane observed from two cameras: object speed must remain zero.
    h, w = 11, 15
    K = np.array([[10., 0, 7], [0, 10., 5], [0, 0, 1.]])
    uv = np.array([[4., 4.], [7., 5.], [10., 6.]])
    current = np.eye(4); past = np.eye(4); past[0, 3] = 1.
    G = np.array([[0., -1, 0, 3.], [1, 0, 0, -2.], [0, 0, 1, 1.], [0, 0, 0, 1.]])
    row = {'lidar_to_global': G, 't0_lidar_us': 1000000}
    def frame(t, pose):
        return {'sample_data': {'timestamp': t}, 'camera_to_global': pose, 'K': K}
    pair = {'current': frame(1000000, current), 'past': frame(500000, past)}
    depth = np.full((h, w), 10., np.float32)
    f = np.zeros((h, w, 2), np.float32); f[..., 0] = -1.
    result = geometry(row, pair, uv, depth, depth, f, -f)
    assert np.max(np.abs(result['velocity_R_mps'])) < 1e-6
    # Moving plane: +1m world-X in .5s => +2m/s, rotates to -Y in R.
    f[..., 0] = -2.
    result = geometry(row, pair, uv, depth, depth, f, -f)
    assert np.max(np.abs(result['velocity_R_mps'] - np.array([0., -2., 0.]))) < 1e-6
    q, mask = sample(depth, np.array([[14., 10.], [15., 5.], [np.nan, 0.]]))
    assert mask.tolist() == [True, False, False] and q[0, 0] == 10 and np.isnan(q[1:]).all()
    return {'static_pose_compensation': True, 'moving_coordinate_direction': True, 'closed_boundary_and_missing': True}


def authenticate(a):
    assert sha(a.history/'complete.json') == HISTORY_SHA
    done = read(a.history/'complete.json')
    for name, digest in done['files_sha256'].items():
        assert sha(a.history/name) == digest
    m = read(a.depth_asset/'manifest.json')
    assert m['commit'] == 'eb5b6fac0dc155e4e52f576e304fbf11655ff339'
    assert m['weight']['sha256'] == WEIGHT_SHA and sha(a.depth_asset/m['weight']['name']) == WEIGHT_SHA
    for name, desc in m['source_files'].items():
        assert sha(a.depth_asset/'source'/name) == desc['sha256']
    assert sha(a.raft_asset/'manifest.json') == 'db378bcf49a348a8968d571f834a33e3f434582e7b0d551b805f49cefb320cdd'
    rm = read(a.raft_asset/'manifest.json')
    inference_files = {name:desc for name,desc in rm['files'].items()
                       if name.startswith('source/') or name == 'weights/raft-things.pth'}
    assert len(inference_files) == 36
    for name, desc in inference_files.items():
        assert sha(a.raft_asset/name) == desc['sha256'], name
    return m, rm


def metric_config(source):
    """Preserve official config values; old MMCV cannot deepcopy imported np.

    Use MMCV's own inheritance merge after dropping Python module bindings,
    which are not model settings. No edits to the pinned official source.
    """
    from mmcv import Config
    def resolve(path):
        scope = runpy.run_path(str(path))
        bases = scope.pop('_base_', [])
        if isinstance(bases, str): bases = [bases]
        merged = {}
        for base in bases:
            merged = Config._merge_a_into_b(resolve(path.parent/base), merged)
        own = {k:v for k,v in scope.items() if not k.startswith('__') and not isinstance(v,types.ModuleType)}
        return Config._merge_a_into_b(own, merged)
    return Config(resolve(source/'mono/configs/HourglassDecoder/vit.raft5.small.py'))


def load_models(a, depth_manifest, raft_manifest):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    sys.path.insert(0, str(a.depth_asset/'source'))
    from mono.model.monodepth_model import get_configured_monodepth_model
    depth = get_configured_monodepth_model(metric_config(a.depth_asset/'source'))
    state = torch.load(a.depth_asset/depth_manifest['weight']['name'], map_location='cpu', weights_only=True)['model_state_dict']
    keys = depth.load_state_dict(state, strict=False)
    # Official checkpoint excludes the DINO mask token. DensePredModel.forward
    # calls encoder(input) without masks; ViT_DINO_reg.py:939 only reads this
    # zero-initialized token when masks is not None. No inference-used parameter
    # is allowed to be missing.
    assert keys.missing_keys == ['depth_model.encoder.mask_token'] and not keys.unexpected_keys, str(keys)
    assert torch.count_nonzero(depth.depth_model.encoder.mask_token).item() == 0
    write(a.out/'model_load.json', {'missing_keys':keys.missing_keys, 'unexpected_keys':keys.unexpected_keys,
          'missing_token_used':False, 'reason':'official dense encoder call supplies no masks; unused token initialized to zero'})
    depth.eval().requires_grad_(False).cuda()
    del state
    sys.path.insert(0, str(a.raft_asset/'source/core'))
    from raft import RAFT
    from utils.utils import InputPadder
    class Settings(argparse.Namespace):
        def __contains__(self, key):
            return hasattr(self, key)
    raft = RAFT(Settings(small=False, mixed_precision=False, alternate_corr=False, dropout=0.))
    state = torch.load(a.raft_asset/'weights/raft-things.pth', map_location='cpu', weights_only=True)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, 'module.')
    raft.load_state_dict(state, strict=True)
    raft.eval().requires_grad_(False).cuda()
    return torch, depth, raft, InputPadder


def infer_depth(torch, model, rgb, K):
    import cv2
    H, W = rgb.shape[:2]; size = (616, 1064)
    scale = min(size[0]/H, size[1]/W)
    resized = cv2.resize(rgb, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_LINEAR)
    dh, dw = size[0]-resized.shape[0], size[1]-resized.shape[1]
    top, left = dh//2, dw//2
    padded = cv2.copyMakeBorder(resized, top, dh-top, left, dw-left, cv2.BORDER_CONSTANT, value=[123.675,116.28,103.53])
    x = torch.from_numpy(padded.transpose(2,0,1).copy()).float().cuda()
    mean = x.new_tensor([123.675,116.28,103.53])[:,None,None]
    std = x.new_tensor([58.395,57.12,57.375])[:,None,None]
    pred, _, _ = model.inference({'input': ((x-mean)/std)[None]})
    pred = pred.squeeze()[top:size[0]-(dh-top), left:size[1]-(dw-left)]
    pred = torch.nn.functional.interpolate(pred[None,None], (H,W), mode='bilinear', align_corners=False).squeeze()
    # Match official canonical-camera preprocessing: scaled fx, not original fx.
    pred = (pred * (float(K[0][0])*scale/1000.)).clamp(0,300)
    assert pred.shape == (H,W) and torch.isfinite(pred).all()
    return pred.cpu().numpy().astype(np.float32)


def main(a):
    from PIL import Image
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'samples').mkdir()
    progress = {'phase':'authenticate', 'completed_pairs':0, 'anchors':a.anchors}
    write(a.out/'progress.json', progress)
    dm, rm = authenticate(a)
    qa = analytic_checks()
    rows = read(a.history/'records.json')['records'][:a.anchors]
    image_root = Path(read(a.history/'manifest.json')['source_data_root'])
    protocol = {'status':'INPUT_ONLY_FROZEN_INFERENCE', 'source_sha256':sha(__file__), 'seed':11,
        'history_complete_sha256':HISTORY_SHA, 'depth_weight_sha256':WEIGHT_SHA,
        'depth_commit':dm['commit'], 'raft_commit':rm['commit'], 'depth_asset_manifest_sha256':sha(a.depth_asset/'manifest.json'),
        'config_compatibility':'MMCV native inheritance merge; exclude unused Python module bindings from config; official model settings preserved',
        'stride':a.stride, 'anchors':[r['identity'] for r in rows], 'cameras':list(CAMERAS), 'GT_read':False,
        'future_images_read':False, 'optimizer_updates':0, 'analytic_checks':qa,
        'geometry':'point_t0 in t0 LiDAR reference; current exposure registered to t0 by estimated historical CV',
        'evaluation_planned':'full original material-point population, nearest valid surface within 1m; zero/CV fallbacks; allvalid and fb<=1px variants, no fitted thresholds'}
    write(a.out/'protocol.json', protocol)
    torch, depth, raft, Padder = load_models(a, dm, rm)
    records = []
    with torch.inference_mode():
        for row in rows:
            per_camera = {p['channel']:p for p in row['cameras']}
            for ci, channel in enumerate(CAMERAS):
                pair = per_camera[channel]; images=[]; tick=time.monotonic()
                progress.update(phase='inference', ordinal=row['ordinal'], camera=channel)
                write(a.out/'progress.json',progress)
                for when in ('current','past'):
                    f=pair[when]; desc=f['image']; path=image_root/desc['file']
                    assert f['sample_data']['timestamp'] <= row['input_availability_us']
                    assert sha(path)==desc['sha256']
                    images.append(np.asarray(Image.open(path).convert('RGB')))
                depths=[infer_depth(torch,depth,x,pair[w]['K']) for x,w in zip(images,('current','past'))]
                xs=[torch.from_numpy(x.transpose(2,0,1).copy()).float()[None].cuda() for x in images]
                padder=Padder(xs[0].shape); current,past=padder.pad(*xs); flows=[]
                for first,second in ((current,past),(past,current)):
                    _,high=raft(first,second,iters=20,test_mode=True)
                    flow=padder.unpad(high)[0].permute(1,2,0).cpu().numpy().copy()
                    assert np.isfinite(flow).all();flows.append(flow)
                H,W=images[0].shape[:2]
                yy,xx=np.meshgrid(np.arange(a.stride//2,H,a.stride),np.arange(a.stride//2,W,a.stride),indexing='ij')
                uv=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float64)
                result=geometry(row,pair,uv,*depths,*flows)
                name=f"{row['ordinal']:04d}_{channel}.npz"
                np.savez_compressed(a.out/'samples'/name,**result)
                sel=result['fb1_in_roi']; speed=np.linalg.norm(result['velocity_R_mps'][sel,:2],axis=1)
                rec={'ordinal':row['ordinal'],'identity':row['identity'],'camera':channel,'file':'samples/'+name,
                     'sha256':sha(a.out/'samples'/name),'queries':len(uv),'finite_correspondences':int(result['correspondence_valid'].sum()),
                     'roi_queries':int(result['in_roi'].sum()),'fb1_roi_queries':int(sel.sum()),
                     'fb1_roi_speed_quantiles_mps':np.quantile(speed,[0,.5,.9,.99,1]).tolist() if len(speed) else None,
                     'seconds':time.monotonic()-tick,'camera_dt_s':float(result['camera_dt_s'])}
                records.append(rec)
                with (a.out/'records.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
                progress['completed_pairs']+=1
                progress['seconds']=time.monotonic()-started
                write(a.out/'progress.json',progress)
                print(json.dumps(rec),flush=True)
                del xs,current,past,high,flows,depths,result
    torch.cuda.synchronize()
    assert all(p.grad is None for m in (depth,raft) for p in m.parameters())
    complete={'status':'COMPLETE_FROZEN_SURFACE_HISTORY','anchors':len(rows),'pairs':len(records),
        'queries':sum(r['queries'] for r in records),'fb1_roi_queries':sum(r['fb1_roi_queries'] for r in records),
        'seconds':time.monotonic()-started,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
        'optimizer_updates':0,'GT_read':False,'protocol_sha256':sha(a.out/'protocol.json'),'records_sha256':sha(a.out/'records.jsonl')}
    write(a.out/'complete.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('depth-asset','raft-asset','history','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--anchors',type=int,default=16);p.add_argument('--stride',type=int,default=8)
    p.add_argument('--max-seconds',type=int,default=1800);a=p.parse_args()
    signal.signal(signal.SIGALRM,lambda s,f: (_ for _ in ()).throw(TimeoutError('bounded inference time exceeded')))
    signal.alarm(a.max_seconds)
    try:main(a)
    except BaseException:
        if a.out.exists():write(a.out/'failed.json',{'error':traceback.format_exc()})
        raise
