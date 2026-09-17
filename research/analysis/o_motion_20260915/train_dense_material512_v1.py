"""Train one authenticated DenseMaterialState arm on original train512.

The shared current-only codec is loaded as an exact, frozen starting point.
Only the recurrent dynamics, content increment, and velocity head are trained.
T renders fixed source content with transport, J renders evolving content with
transport, and D renders evolving content directly.  This is a training
artifact; development evaluation and model selection are deliberately absent.
"""
import argparse
from collections import OrderedDict
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np


SCHEMA = 'dense-material512-training-v1'
TRAINING = dict(seed=11, samples=512, epochs=4, updates=2048,
                lr=3e-4, weight_decay=.01, grad_clip_norm=10.,
                optimizer='AdamW', betas=[.9, .999], eps=1e-8,
                order='single np.RandomState(11), four sequential permutation(512) draws',
                target_slice=[0, 2, 7], lru_max_samples=8,
                future_horizons=4, logits_horizons=5,
                physical_weight=.1)
SHAPE_XYZ = (200, 200, 16)
FINE_SHAPE = (512, 512, 40)
COARSE_SHAPE = (256, 256, 20)
EXTENT_XYZ = (-51.2, -51.2, -5., 51.2, 51.2, 3.)
ARMS = {
    'T': dict(material_mode='fixed', readout='transport'),
    'J': dict(material_mode='evolving', readout='transport'),
    'D': dict(material_mode='evolving', readout='direct'),
}
COMPLETE_STATUS = 'COMPLETE_DENSE_MATERIAL512_TRAIN'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False,
                                allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def resources_from(protocol):
    resources = protocol['resources']
    if 'train' in resources:
        resources = resources['train']
    require(set(resources) >= {'max_seconds', 'max_allocated_gib'},
            'Incomplete frozen resource cap')
    return dict(max_seconds=resources['max_seconds'],
                max_allocated_gib=resources['max_allocated_gib'])


def validate_protocol(args, protocol):
    require(protocol['schema'] == SCHEMA, 'Wrong protocol schema')
    require(protocol['status'] == 'FROZEN', 'A frozen protocol is required')
    require(protocol['training'] == TRAINING, 'Frozen training recipe changed')
    require(protocol['model'] == dict(
        shape_xyz=list(SHAPE_XYZ), extent_xyz=list(EXTENT_XYZ),
        class_weights=[1., 5.],
        frozen_modules=['encoder', 'decoder'],
        trainable_modules=['dynamics', 'content_increment', 'velocity'],
        arms={'T': 'fixed transport', 'J': 'evolving transport',
              'D': 'evolving direct'}), 'Frozen model boundary changed')
    policy = protocol['numerical_policy']
    require(policy == dict(dtype='float32', matmul_tf32=False,
                           cudnn_tf32=False, cudnn_benchmark=False,
                           deterministic_algorithms=True,
                           cudnn_deterministic=True, loss_device='cpu',
                           physical_loss_device='cuda',
                           downsample_device='cpu',
                           cublas_workspace_config=':4096:8'),
            'Numerical policy changed')
    resources = resources_from(protocol)
    require(args.max_seconds == resources['max_seconds'] and
            args.max_allocated_gib == resources['max_allocated_gib'],
            'CLI resource cap must equal frozen protocol')
    require(args.max_seconds > 0 and args.max_allocated_gib > 0,
            'Resource caps must be positive')
    require(args.arm in ARMS, 'Unknown arm')
    here = Path(__file__).resolve().parent
    required_sources = {Path(__file__).name,
                        'train_shared_current_codec512_v1.py',
                        'dense_material_state_v1.py',
                        'dense_task_state_v1.py', 'dense_task_state_v2.py', 'transport_ops.py',
                        'train_source_motion_v1.py', 'native_state_cache.py'}
    require(required_sources <= set(protocol['sources_sha256']),
            'Incomplete local source bindings')
    for name, digest in protocol['sources_sha256'].items():
        require(Path(name).name == name and sha(here / name) == digest,
                'Local source SHA mismatch: ' + name)
    for path, digest in protocol['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Runtime source SHA mismatch: ' + path)
    require(sha(args.protocol) == args.protocol_sha256,
            'Protocol SHA mismatch')
    require(protocol['codec_complete_sha256'] == args.codec_complete_sha256,
            'Codec completion binding differs from protocol')
    return resources


def authenticate_codec(args, protocol, native):
    run = Path(args.codec_run).resolve()
    complete_path = run / 'complete.json'
    require(not (run/'failed.json').exists(), 'Codec failed receipt present')
    require(sha(complete_path) == args.codec_complete_sha256,
            'Shared codec complete.json SHA mismatch')
    complete = json.loads(complete_path.read_text())
    require(complete['schema'] == 'shared-current-codec512-v1' and
            complete['status'] == 'COMPLETE_CURRENT_CODEC512' and
            complete['updates'] == 2048, 'Shared codec is incomplete')
    checkpoint = run / 'shared_codec.pth'
    expected_checkpoint = protocol['codec_checkpoint_sha256']
    require(sha(checkpoint) == expected_checkpoint and
            complete['files_sha256']['shared_codec.pth'] == expected_checkpoint,
            'Shared codec checkpoint SHA mismatch')
    for name in ('manifest.json', 'protocol.json', 'samples.jsonl',
                 'training.jsonl', 'evaluation.json'):
        require(name in complete['files_sha256'] and
                sha(run / name) == complete['files_sha256'][name],
                'Shared codec receipt file mismatch: ' + name)
    require(sha(run / 'protocol.json') == complete['protocol_sha256'],
            'Shared codec protocol receipt is not self-consistent')
    import torch
    payload = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    require(payload['schema'] == 'shared-current-codec512-v1' and
            payload['status'] == 'COMPLETE_CURRENT_CODEC512' and
            payload['protocol_sha256'] == complete['protocol_sha256'] and
            payload.get('current_only_supervision') is True,
            'Shared codec payload provenance changed')
    require(payload.get('updates') == 2048, 'Shared codec payload update count changed')
    require(payload['initial_frozen_sha256'] == payload['frozen_final_sha256'],
            'Shared codec recurrent parameters changed')
    require('state_dict' in payload and 'final_model_sha256' in payload,
            'Shared codec payload lacks full state/digest')
    require(payload['final_model_sha256'], 'Empty shared codec final digest')
    return run, complete, payload


def main(args):
    started = time.monotonic()
    # All CLI paths are anchored before changing cwd for the native import.
    # Server launchers often pass relative output/cache paths; resolving here
    # prevents those paths from silently moving under --repo.
    for name in ('protocol', 'codec_run', 'train_cache', 'sparse_labels',
                 'repo', 'out'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    protocol = json.loads(Path(args.protocol).read_text())
    resources = validate_protocol(args, protocol)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', protocol)

    # This must be present before importing torch.  A deterministic fallback
    # would silently change the protocol and invalidate cross-arm comparison.
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    package = Path(__file__).resolve().parent
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_material_state_v1 import DenseMaterialState

    # The native world-head import resolves third_lib extensions relative to
    # the author checkout.  Adding it to sys.path alone is insufficient when
    # the launcher starts from the analysis package directory.
    os.chdir(str(Path(args.repo).resolve()))
    native._prepare_repo(args.repo)
    lossmod = importlib.import_module(
        'projects.mmdet3d_plugin.bevformer.dense_heads.world_head_v1')
    torch.set_num_threads(2)
    torch.manual_seed(11)
    np.random.seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(), 'CUDA is required for this training')
    free, total = torch.cuda.mem_get_info()
    require(free >= 32 * 2**30, 'Less than 32 GiB free')
    torch.cuda.set_per_process_memory_fraction(
        resources['max_allocated_gib'] * 2**30 / total)
    torch.cuda.reset_peak_memory_stats()

    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stopped.append(signum))

    def check():
        require(not stopped, 'Termination requested; no resume')
        require(time.monotonic() - started < resources['max_seconds'],
                'Wall-clock ceiling reached')
        require(torch.cuda.max_memory_allocated() <=
                resources['max_allocated_gib'] * 2**30,
                'CUDA allocation ceiling reached')

    codec_run, codec_complete, codec_payload = authenticate_codec(args, protocol, native)
    trainroot, records, cache_receipt = helper.cache_index(
        args.train_cache, 'train', protocol)
    require(len(records) == 512 and len({r['sample_token'] for r in records}) == 512 and
            len({r['scene_token'] for r in records}) == 256,
            'Original train512 population changed')
    require(cache_receipt['index_sha256'] == protocol['cache_index_sha256']['train'],
            'Train cache receipt mismatch')
    labelroot, label_records = helper.labels_manifest(args.sparse_labels, protocol)

    # The codec payload is the sole initialization source.  The state digest
    # is checked after loading, before any optimizer or forward pass.
    model = DenseMaterialState(material_mode=ARMS[args.arm]['material_mode']).cuda()
    model.load_state_dict(codec_payload['state_dict'], strict=True)
    initial_model_sha = native._parameter_digest(model)
    require(initial_model_sha == codec_payload['final_model_sha256'],
            'Loaded arm does not equal shared codec final state')
    frozen_initial_sha = {
        name: native._parameter_digest(getattr(model, name))
        for name in ('encoder', 'decoder')
    }
    for name in ('encoder', 'decoder'):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(False)
    for name in ('dynamics', 'content_increment', 'velocity'):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
    require(all(not p.requires_grad for n, p in model.named_parameters()
                if n.startswith(('encoder.', 'decoder.'))),
            'Frozen codec became trainable')
    require(all(p.requires_grad for n, p in model.named_parameters()
                if n.startswith(('dynamics.', 'content_increment.', 'velocity.'))),
            'Recurrent trainable boundary changed')
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(sum(p.numel() for p in trainable) > 0, 'No trainable recurrent parameters')

    weights = torch.tensor([1., 5.], device='cpu')
    lru = OrderedDict()
    populations = {}

    def sample(ordinal):
        check()
        if ordinal in lru:
            value = lru.pop(ordinal)
            lru[ordinal] = value
            return value
        record = records[ordinal]
        # CPU storage keeps the LRU bounded in host memory; the detached copy
        # is the only token tensor subsequently moved to CUDA for forward().
        tokens = helper.load_tokens(trainroot, record, native, 'cpu')
        target_path = native._sample_directory(trainroot, record) / \
            record['files']['targets']['file']
        native._verify_file(target_path, record['files']['targets'])
        raw = np.load(target_path, allow_pickle=False, mmap_mode='r')
        require(raw.dtype == np.uint8 and raw.shape == (1, 7) + FINE_SHAPE,
                'Native target shape/dtype changed')
        target = np.asarray(raw[0, 2:]).copy()
        del raw
        require(target.shape == (5,) + FINE_SHAPE and
                np.isin(target, [0, 1, 255]).all(),
                'Full five-frame target class mapping changed')
        coarse = lossmod._downsample_occ_target(
            torch.from_numpy(target).long()).to(device='cpu')
        label_descriptor = label_records.get(record['sample_token'])
        require(label_descriptor is not None,
                'Missing sparse label for cache sample')
        label = helper.load_sparse(labelroot, label_descriptor, record)
        population = dict(ordinal=ordinal, sample_token=record['sample_token'],
                          scene_token=record['scene_token'],
                          inputs_sha256=record['files']['inputs']['sha256'],
                          targets_sha256=record['files']['targets']['sha256'],
                          sparse_label_sha256=label_descriptor['sha256'],
                          target_shape=list(target.shape),
                          target_occupied=[int(np.count_nonzero(target[h] == 1)) for h in range(5)],
                          target_empty=[int(np.count_nonzero(target[h] == 0)) for h in range(5)],
                          target_ignore=[int(np.count_nonzero(target[h] == 255)) for h in range(5)],
                          dt_future_seconds=label['dt_future_seconds'].tolist())
        if ordinal not in populations:
            populations[ordinal] = population
            append(output / 'samples.jsonl', population)
        value = dict(tokens=tokens.detach().cpu(), target=target, coarse=coarse,
                     label=label, sample_token=record['sample_token'],
                     scene_token=record['scene_token'])
        lru[ordinal] = value
        while len(lru) > TRAINING['lru_max_samples']:
            lru.popitem(last=False)
        require(len(lru) <= TRAINING['lru_max_samples'], 'LRU cap exceeded')
        return value

    def occupancy_loss(logits_cpu, target_cpu):
        require(logits_cpu.device.type == 'cpu' and target_cpu.device.type == 'cpu',
                'Occupancy loss must run on CPU')
        resized = F.interpolate(logits_cpu, size=COARSE_SHAPE,
                                mode='trilinear', align_corners=False)
        ce = lossmod.CE_ssc_loss(resized, target_cpu, weights, ignore_index=255)
        lovasz = lossmod.lovasz_softmax(torch.softmax(resized, dim=1),
                                        target_cpu, ignore=255)
        return ce + lovasz, dict(ce=float(ce.detach()), lovasz=float(lovasz.detach()))

    order_rng = np.random.RandomState(11)
    orders = [order_rng.permutation(512).tolist() for _ in range(4)]
    manifest = dict(schema=SCHEMA, status='RUNNING', arm=args.arm,
                    protocol_sha256=args.protocol_sha256,
                    codec_run=str(codec_run),
                    codec_complete_sha256=args.codec_complete_sha256,
                    codec_checkpoint_sha256=protocol['codec_checkpoint_sha256'],
                    cache=cache_receipt, sparse_labels=protocol['labels'],
                    seed=11, training=TRAINING, sample_orders=orders,
                    sample_orders_sha256=hashlib.sha256(
                        json.dumps(orders, separators=(',', ':')).encode()).hexdigest(),
                    lru_max_samples=8, inputs='authenticated prev_bev_input[:, -1] only',
                    target_supervision='full raw target [0,2:] for five logits; CPU loss only',
                    future_target_values_used_in_model=False,
                    development_samples_read=0, official_models_trained=False,
                    no_prior_future_fit_loaded=True, shared_current_codec_loaded=True,
                    initial_model_sha256=initial_model_sha,
                    initial_parameters_sha256=initial_model_sha,
                    initial_frozen_sha256=frozen_initial_sha,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    all_parameters=sum(p.numel() for p in model.parameters()),
                    populations_expected=512)
    write(output / 'manifest.json', manifest)
    (output / 'sample_order.jsonl').touch()
    optimizer = torch.optim.AdamW(trainable, lr=TRAINING['lr'],
                                  betas=tuple(TRAINING['betas']),
                                  eps=TRAINING['eps'],
                                  weight_decay=TRAINING['weight_decay'])
    model.train()
    update = 0
    try:
        for epoch, order in enumerate(orders):
            for position, ordinal in enumerate(order):
                check()
                item = sample(ordinal)
                optimizer.zero_grad(set_to_none=True)
                forecast = model(item['tokens'].cuda(), ARMS[args.arm]['readout'])
                logits_cpu = forecast['logits'][0].cpu()
                displacement = forecast['displacement']
                occupancy, detail = occupancy_loss(logits_cpu, item['coarse'])
                sparse = helper.gather_sparse(displacement, item['label'])
                physical, physical_detail = helper.object_group_loss(sparse, item['label'])
                loss = occupancy + TRAINING['physical_weight'] * physical.cpu()
                require(bool(torch.isfinite(loss)), 'Nonfinite total loss')
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in trainable), 'Nonfinite recurrent gradient')
                preclip = torch.nn.utils.clip_grad_norm_(trainable,
                                                         TRAINING['grad_clip_norm'])
                before = {n: p.detach().clone() for n, p in model.named_parameters()
                          if p.requires_grad}
                optimizer.step()
                update += 1
                update_norm = {
                    prefix: float(sum((p.detach() - before[n]).double().square().sum()
                                      for n, p in model.named_parameters()
                                      if n.startswith(prefix)).sqrt())
                    for prefix in ('dynamics.', 'content_increment.', 'velocity.')
                }
                order_row = dict(arm=args.arm, epoch=epoch, position=position,
                                 update=update, ordinal=ordinal,
                                 sample_token=item['sample_token'],
                                 scene_token=item['scene_token'])
                append(output / 'sample_order.jsonl', order_row)
                append(output / 'training.jsonl', dict(**order_row,
                    total=float(loss.detach()), occupancy=float(occupancy.detach()),
                    physical=float(physical.detach()), **detail,
                    physical_detail=physical_detail,
                    preclip_grad_norm=float(preclip), update_norm=update_norm,
                    elapsed_seconds=time.monotonic() - started))
                del forecast, logits_cpu, displacement, sparse, occupancy, physical, loss
        require(update == TRAINING['updates'], 'Actual update count mismatch')
        require(len(populations) == 512, 'Population receipt is incomplete')
        frozen_final_sha = {name: native._parameter_digest(getattr(model, name))
                            for name in ('encoder', 'decoder')}
        require(frozen_final_sha == frozen_initial_sha,
                'Frozen encoder/decoder changed during recurrent training')
        final_model_sha = native._parameter_digest(model)
        payload = dict(schema=SCHEMA, status=COMPLETE_STATUS,
                       arm=args.arm, readout=ARMS[args.arm]['readout'],
                       material_mode=ARMS[args.arm]['material_mode'],
                       protocol_sha256=args.protocol_sha256,
                       codec_complete_sha256=args.codec_complete_sha256,
                       codec_checkpoint_sha256=protocol['codec_checkpoint_sha256'],
                       state_dict={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()},
                       initial_model_sha256=initial_model_sha,
                       final_model_sha256=final_model_sha,
                       initial_parameters_sha256=initial_model_sha,
                       final_parameters_sha256=final_model_sha,
                       initial_frozen_sha256=frozen_initial_sha,
                       frozen_final_sha256=frozen_final_sha,
                       frozen_sha256=frozen_final_sha,
                       updates=update, development_read=0,
                       future_targets_in_model=False,
                       future_targets_in_loss=True, fine_evaluation=False)
        checkpoint = output / 'model_final.pth'
        temporary = output / 'model_final.pth.tmp'
        torch.save(payload, str(temporary)); temporary.replace(checkpoint)
        manifest.update(status=COMPLETE_STATUS,
                        populations=list(populations.values()),
                        frozen_final_sha256=frozen_final_sha,
                        final_model_sha256=final_model_sha,
                        final_parameters_sha256=final_model_sha,
                        completed_updates=update)
        write(output / 'manifest.json', manifest)
        complete_files = ['protocol.json', 'manifest.json', 'samples.jsonl',
                          'sample_order.jsonl', 'training.jsonl', 'model_final.pth']
        complete = dict(schema=SCHEMA, status=COMPLETE_STATUS,
                        arm=args.arm, protocol_sha256=args.protocol_sha256,
                        codec_complete_sha256=args.codec_complete_sha256,
                        codec_checkpoint_sha256=protocol['codec_checkpoint_sha256'],
                        updates=update, development_read=0,
                        initial_parameters_sha256=initial_model_sha,
                        final_parameters_sha256=final_model_sha,
                        frozen_sha256=frozen_final_sha,
                        fine_evaluation=False,
                        elapsed_seconds=time.monotonic() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        files_sha256={name: sha(output / name) for name in complete_files})
        write(output / 'complete.json', complete)
        print(json.dumps(dict(event='complete', status=complete['status'], arm=args.arm,
                              updates=update, final_model_sha256=final_model_sha),
                         allow_nan=False), flush=True)
    except BaseException as error:
        write(output / 'failed.json', dict(schema=SCHEMA, arm=args.arm,
                                          error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--codec-run', required=True)
    parser.add_argument('--codec-complete-sha256', required=True)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--sparse-labels', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-seconds', required=True, type=int)
    parser.add_argument('--max-allocated-gib', required=True, type=float)
    try:
        main(parser.parse_args())
    except BaseException:
        raise
