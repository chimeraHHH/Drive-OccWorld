"""Full-shape synthetic T/J/D autograd probe; never trains or reads data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def run(a):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_material_state_v1 import DenseMaterialState
    native._prepare_repo(a.repo)
    os.chdir(a.repo)
    from projects.mmdet3d_plugin.bevformer.dense_heads import world_head_v1 as losses
    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    assert free >= 24 * 2**30
    torch.cuda.set_per_process_memory_fraction(12 * 2**30 / total)
    x = torch.randn(1, 40000, 256, device='cuda')
    target = torch.zeros(5, 512, 512, 40, dtype=torch.long)
    for h in range(5):
        target[h, 100+2*h:140+2*h, 200+h:220+h, 5:12] = 1
    target[:, :2] = 255
    coarse = losses._downsample_occ_target(target)
    label = dict(source_flat_indices=np.arange(0, 640000, 160, dtype=np.int64),
                 object_index=np.repeat(np.arange(4, dtype=np.int64), 1000),
                 valid=np.ones((4, 4000), dtype=bool),
                 object_speed_group=np.tile(np.array([0, 1, 2, 2], dtype=np.int8), (4, 1)),
                 target_displacement_m=np.zeros((4, 4000, 3), dtype=np.float32))
    for h in range(4):
        label['target_displacement_m'][h, :, 0] = np.repeat([0., .1, 1., 3.], 1000) * (h+1)
    result = dict(schema='material512-determinism-probe-v2', synthetic=True,
                  optimizer_updates=0, dataset_samples_read=0, arms={})
    for arm in ('T', 'J', 'D'):
        torch.manual_seed(11)
        model = DenseMaterialState(material_mode='fixed' if arm == 'T' else 'evolving').cuda()
        model.encoder.requires_grad_(False)
        model.decoder.requires_grad_(False)
        # Nonzero recurrence probes interpolation collisions beyond identity.
        with torch.no_grad():
            model.velocity.weight.normal_(0, .003)
            model.content_increment.weight.normal_(0, .003)
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = native._parameter_digest(model)
        repetitions = []
        for repeat in range(2):
            model.zero_grad(set_to_none=True)
            started = time.monotonic()
            forecast = model(x, 'direct' if arm == 'D' else 'transport')
            resized = F.interpolate(forecast['logits'][0].cpu(), size=(256, 256, 20),
                                    mode='trilinear', align_corners=False)
            occupancy = losses.CE_ssc_loss(resized, coarse, torch.tensor([1., 5.]), ignore_index=255)
            occupancy = occupancy + losses.lovasz_softmax(torch.softmax(resized, 1), coarse, ignore=255)
            physical, _ = helper.object_group_loss(helper.gather_sparse(forecast['displacement'], label), label)
            loss = occupancy + .1 * physical.cpu()
            loss.backward()
            gradients = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
            assert torch.isfinite(gradients).all() and float(gradients.norm()) > 0
            repetitions.append(dict(loss=float(loss.detach()), occupancy=float(occupancy.detach()),
                physical=float(physical.detach()), gradient_sha256=native._tensor_digest(gradients),
                gradient_norm=float(gradients.norm()), seconds=time.monotonic()-started))
            del forecast, resized, loss, occupancy, physical, gradients
        assert repetitions[0]['loss'] == repetitions[1]['loss']
        assert repetitions[0]['gradient_sha256'] == repetitions[1]['gradient_sha256']
        assert native._parameter_digest(model) == initial
        result['arms'][arm] = dict(initial_parameters_sha256=initial, repetitions=repetitions)
        del model, parameters
        torch.cuda.empty_cache()
    assert len({v['initial_parameters_sha256'] for v in result['arms'].values()}) == 1
    result.update(status='PASS_FULL_TJD_GRAPH_REPEATED',
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  limitation='Synthetic within-process repeated graph only; no training/generalization evidence',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
