"""Post-fit train-only full-grid and object-balanced readout measurements.

Not a held-out evaluation, checkpoint selection, or O comparison. Includes
zero/reversed displacement interventions at the fixed final checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    started = time.monotonic()
    root = Path(a.fit); package = Path(a.package)
    done = json.loads((root / 'complete.json').read_text())
    p = json.loads((root / 'protocol.json').read_text())
    assert done['status'] == 'COMPLETE_TRAIN_ONLY_ENGINEERING_FIT' and p['schema'] == 'dense-task-state-fit-v2'
    assert sha(root / 'protocol.json') == done['protocol_sha256']
    for name, digest in done['files_sha256'].items():
        assert sha(root / name) == digest, name
    for name, digest in p['sources_sha256'].items():
        assert sha(package / name) == digest, name
    for path, digest in p['runtime_source_sha256'].items():
        assert sha(path) == digest, path
    sys.path.insert(0, str(package))
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_task_state_v2 import DenseTaskState
    native._prepare_repo(a.repo)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    assert free > 16*2**30
    torch.cuda.set_per_process_memory_fraction(12*2**30/total)
    torch.cuda.reset_peak_memory_stats()
    cache, records, _ = helper.cache_index(a.train_cache, 'train', p)
    labelroot, labels = helper.labels_manifest(a.sparse_labels, p)
    manifest = json.loads((root / 'manifest.json').read_text())
    selected = manifest['selected_ordinals']
    result = dict(scope='post-fit measurements on the same four training samples; not generalization',
        protocol_sha256=done['protocol_sha256'], fit_complete_sha256=sha(root/'complete.json'),
        source_sha256=sha(__file__), seed=11, development_samples=0, selected_ordinals=selected, arms={})
    for arm in ('transport', 'direct'):
        model = DenseTaskState().cuda().eval()
        payload = torch.load(str(root / (arm + '_final.pth')), map_location='cpu')
        assert payload['protocol_sha256'] == done['protocol_sha256'] and payload['fixed_final_update'] == 32
        model.load_state_dict(payload['state_dict'], strict=True); del payload
        hist = {name: np.zeros((5,2,2), dtype=np.int64) for name in ('normal','zero','reverse')}
        physical = [[[] for _ in range(3)] for _ in range(4)]
        zero_physical = [[[] for _ in range(3)] for _ in range(4)]
        rows = []
        for index in selected:
            assert time.monotonic()-started < 300
            r = records[index]; tokens = helper.load_tokens(cache, r, native, 'cuda')
            with torch.no_grad():
                forecast = model(tokens, arm)
            # GT is read after this sample's dense prediction exists.
            path = native._sample_directory(cache, r) / r['files']['targets']['file']
            native._verify_file(path, r['files']['targets'])
            target = np.load(path, allow_pickle=False)[0, 2:]
            label = helper.load_sparse(labelroot, labels[r['sample_token']], r)
            pred = helper.gather_sparse(forecast['displacement'], label).cpu().numpy()
            epe = np.linalg.norm(pred-label['target_displacement_m'], axis=-1)
            zero_epe = np.linalg.norm(label['target_displacement_m'], axis=-1)
            for h in range(4):
                for obj, group in enumerate(label['object_speed_group'][h]):
                    if group < 0:
                        continue
                    mask = label['valid'][h] & (label['object_index'] == obj)
                    assert mask.any()
                    physical[h][int(group)].append(float(epe[h, mask].mean(dtype=np.float64)))
                    zero_physical[h][int(group)].append(float(zero_epe[h, mask].mean(dtype=np.float64)))
            sample_hist = {}
            for condition in ('normal', 'zero', 'reverse'):
                with torch.no_grad():
                    output = forecast if condition == 'normal' else model(tokens, arm, condition)
                    if condition != 'normal':
                        assert torch.equal(output['displacement'], forecast['displacement'])
                    counts = []
                    for h in range(5):
                        dense = F.interpolate(output['logits'][:, h], size=(512,512,40),
                            mode='trilinear', align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
                        valid = target[h] != 255
                        count = np.bincount((2*target[h][valid] + dense[valid]).astype(np.int64), minlength=4).reshape(2,2)
                        hist[condition][h] += count; counts.append(count.tolist())
                        del dense, valid
                    sample_hist[condition] = counts
                if condition != 'normal':
                    del output
            rows.append(dict(sample_token=r['sample_token'], scene_token=r['scene_token'], hist=sample_hist))
            del forecast, tokens, target
        metrics = {}
        for condition, count in hist.items():
            tp = count[:,1,1]; fp = count[:,0,1]; fn = count[:,1,0]
            iou = np.divide(tp, tp+fp+fn, out=np.zeros(5,dtype=float), where=tp+fp+fn>0)
            recall = np.divide(tp, tp+fn, out=np.zeros(5,dtype=float), where=tp+fn>0)
            metrics[condition] = dict(hist=count.tolist(), iou_by_horizon=iou.tolist(),
                recall_by_horizon=recall.tolist(), future_mean_iou=float(iou[1:].mean()))
        phys = []
        for h in range(4):
            row = dict(horizon_seconds=(h+1)*.5)
            for g, name in enumerate(('stationary','ambiguous','moving')):
                values, zeros = physical[h][g], zero_physical[h][g]
                row[name] = dict(objects=len(values), epe_xyz_m=float(np.mean(values)) if values else None,
                    zero_motion_epe_xyz_m=float(np.mean(zeros)) if zeros else None)
            phys.append(row)
        result['arms'][arm] = dict(occupancy=metrics, object_equal_physical=phys, rows=rows)
        del model; torch.cuda.empty_cache()
    # All interventions and arms must score exactly the same GT population.
    reference = np.asarray(result['arms']['transport']['occupancy']['normal']['hist']).sum(axis=2)
    for arm in result['arms'].values():
        for values in arm['occupancy'].values():
            assert np.array_equal(np.asarray(values['hist']).sum(axis=2), reference)
    result.update(status='COMPLETE_TRAIN_ONLY_READOUT_AUDIT', same_gt_denominators=True,
        seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    out = Path(a.out)
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'arms'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('fit','package','repo','train-cache','sparse-labels','out'):
        parser.add_argument('--'+name, required=True)
    main(parser.parse_args())
