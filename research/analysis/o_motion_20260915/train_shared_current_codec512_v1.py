"""Fresh current-only codec fit on the authenticated original train512 cache.

The model is initialized from ``DenseTaskState`` and is never warm-started
from an earlier fit.  Only its encoder and decoder are optimized; dynamics,
content_increment, and velocity remain frozen at their fresh initialization.
Future target slices are never passed to the model or loss.  This is a
bounded preparation artifact for a later shared-codec comparison, not an
evaluation of O or a generalization result.
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


SCHEMA = 'shared-current-codec512-v1'
TRAINING = dict(seed=11, samples=512, epochs=4, updates=2048,
                lr=3e-4, weight_decay=.01, grad_clip_norm=10.,
                optimizer='AdamW', betas=[.9, .999], eps=1e-8,
                order='single np.RandomState(11), four sequential permutation(512) draws',
                target_slice=[0, 2, 3], lru_max_samples=8)
SHAPE_XYZ = (200, 200, 16)
FINE_SHAPE = (512, 512, 40)
COARSE_SHAPE = (256, 256, 20)


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
    # Training records are the durable progress receipt; fsync each line so a
    # killed process cannot report a completed update that was never recorded.
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False,
                                 allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def protocol_resources(protocol):
    resources = protocol['resources']
    if 'train' in resources:
        resources = resources['train']
    require(set(resources) >= {'max_seconds', 'max_allocated_gib'},
            'Protocol resource cap is incomplete')
    return dict(max_seconds=resources['max_seconds'],
                max_allocated_gib=resources['max_allocated_gib'])


def validate_protocol(args, protocol):
    require(protocol['schema'] == SCHEMA, 'Wrong protocol schema')
    require(protocol['status'] == 'FROZEN', 'A frozen protocol is required')
    for key, value in TRAINING.items():
        require(protocol['training'][key] == value,
                'Frozen training recipe changed: ' + key)
    require(protocol.get('model') == dict(shape_xyz=list(SHAPE_XYZ),
            extent_xyz=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
            class_weights=[1., 5.],
            frozen_modules=['dynamics', 'content_increment', 'velocity'],
            trainable_modules=['encoder', 'decoder']),
            'Frozen model/optimizer boundary changed')
    policy = protocol['numerical_policy']
    require(policy['dtype'] == 'float32' and
            policy['matmul_tf32'] is False and
            policy['cudnn_tf32'] is False and
            policy['cudnn_benchmark'] is False and
            policy['deterministic_algorithms'] is True and
            policy['cudnn_deterministic'] is True and
            policy['loss_device'] == 'cpu' and
            policy['downsample_device'] == 'cpu' and
            policy['cublas_workspace_config'] == ':4096:8',
            'Numerical policy changed')
    resources = protocol_resources(protocol)
    require(args.max_seconds == resources['max_seconds'] and
            args.max_allocated_gib == resources['max_allocated_gib'],
            'CLI resource cap must equal frozen protocol')
    require(args.max_seconds > 0 and args.max_allocated_gib > 0,
            'Resource ceilings must be positive')
    here = Path(__file__).resolve().parent
    for name, digest in protocol['sources_sha256'].items():
        require(Path(name).name == name and sha(here / name) == digest,
                'Local source SHA mismatch: ' + name)
    for path, digest in protocol['runtime_source_sha256'].items():
        require(sha(path) == digest, 'Runtime source SHA mismatch: ' + path)
    require(sha(args.protocol) == args.protocol_sha256,
            'Protocol SHA mismatch')
    return resources


def main(args):
    started = time.monotonic()
    protocol = json.loads(Path(args.protocol).read_text())
    resources = validate_protocol(args, protocol)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', protocol)

    package = Path(__file__).resolve().parent
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Set deterministic CUBLAS workspace before starting Python')
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper

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
    require(torch.cuda.is_available(), 'CUDA is required for this fit')
    free, total = torch.cuda.mem_get_info()
    require(free >= 32 * 2**30, 'Less than 32 GiB free; do not touch another task')
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

    trainroot, records, cache_receipt = helper.cache_index(
        args.train_cache, 'train', protocol)
    # official_index indexes the original dataset, not this 512-row subset.
    # Its exact row order is bound by the authenticated cache index bytes.
    require(len(records) == 512 and len({r['sample_token'] for r in records}) == 512
            and len({r['scene_token'] for r in records}) == 256,
            'Original train512 population changed')
    require(protocol['cache_index_sha256']['train'] ==
            cache_receipt['index_sha256'], 'Train cache receipt mismatch')

    # The native cache module validates input file bytes before deserializing
    # its importable mmdet3d objects.  The target file is verified separately;
    # only [0, 2:3] is copied into the training item.
    model = None
    from dense_task_state_v2 import DenseTaskState
    model = DenseTaskState().cuda()
    for module_name in ('dynamics', 'content_increment', 'velocity'):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for module_name in ('encoder', 'decoder'):
        for parameter in getattr(model, module_name).parameters():
            parameter.requires_grad_(True)
    require(all(not parameter.requires_grad for name, parameter in
                model.named_parameters()
                if name.startswith(('dynamics.', 'content_increment.', 'velocity.'))),
            'Frozen recurrent module became trainable')
    require(all(parameter.requires_grad for name, parameter in model.named_parameters()
                if name.startswith(('encoder.', 'decoder.'))),
            'Codec parameter unexpectedly frozen')
    initial_model_sha = native._parameter_digest(model)
    initial_frozen_sha = {
        name: native._parameter_digest(getattr(model, name))
        for name in ('dynamics', 'content_increment', 'velocity')}
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    require(sum(parameter.numel() for parameter in trainable) > 0,
            'No trainable codec parameters')

    weights = torch.tensor([1., 5.], device='cpu')

    def occ_loss(logits, target):
        # This PyTorch runtime rejects deterministic CUDA trilinear backward.
        # The same mathematical resize and original loss run on CPU; the
        # differentiable copy sends the gradient back to the GPU codec.
        resized = F.interpolate(logits.cpu(), size=COARSE_SHAPE,
                                mode='trilinear', align_corners=False)
        ce = lossmod.CE_ssc_loss(resized, target, weights, ignore_index=255)
        lovasz = lossmod.lovasz_softmax(torch.softmax(resized, dim=1),
                                        target, ignore=255)
        return ce + lovasz, dict(ce=float(ce.detach()),
                                 lovasz=float(lovasz.detach()))

    # Each cache value contains one CPU token tensor and one current-only fine
    # target.  The LRU is deliberately capped at the protocol's eight samples.
    lru = OrderedDict()
    populations = {}

    def sample(ordinal):
        check()
        if ordinal in lru:
            value = lru.pop(ordinal)
            lru[ordinal] = value
            return value
        record = records[ordinal]
        tokens = helper.load_tokens(trainroot, record, native, 'cpu')
        target_path = native._sample_directory(trainroot, record) / \
            record['files']['targets']['file']
        native._verify_file(target_path, record['files']['targets'])
        raw = np.load(target_path, allow_pickle=False, mmap_mode='r')
        require(raw.dtype == np.uint8 and raw.shape == (1, 7) + FINE_SHAPE,
                'Native target shape/dtype changed')
        current = np.asarray(raw[0, 2:3]).copy()
        del raw
        require(np.isin(current, [0, 1, 255]).all(),
                'Current target class mapping changed')
        valid = current != 255
        population = dict(ordinal=ordinal, sample_token=record['sample_token'],
                          scene_token=record['scene_token'],
                          inputs_sha256=record['files']['inputs']['sha256'],
                          targets_sha256=record['files']['targets']['sha256'],
                          current_shape=list(current.shape),
                          current_occupied=int(np.count_nonzero(current == 1)),
                          current_empty=int(np.count_nonzero(current == 0)),
                          current_ignore=int(np.count_nonzero(~valid)))
        if ordinal not in populations:
            populations[ordinal] = population
            append(output / 'samples.jsonl', population)
        coarse = lossmod._downsample_occ_target(
            torch.from_numpy(current).long())
        value = dict(tokens=tokens.detach().cpu(), fine=current,
                     coarse=coarse, sample_token=record['sample_token'])
        lru[ordinal] = value
        while len(lru) > TRAINING['lru_max_samples']:
            lru.popitem(last=False)
        return value

    order_rng = np.random.RandomState(11)
    orders = [order_rng.permutation(512).tolist() for _ in range(4)]
    manifest = dict(schema=SCHEMA, status='RUNNING', protocol_sha256=args.protocol_sha256,
                    seed=11, training=TRAINING, sample_orders=orders,
                    sample_orders_sha256=hashlib.sha256(
                        json.dumps(orders, separators=(',', ':')).encode()).hexdigest(),
                    cache=cache_receipt, selected_samples=512,
                    lru_max_samples=8, inputs='authenticated prev_bev_input[:, -1] only',
                    target_supervision='current raw target [0,2:3] only; full target file SHA verified',
                    future_target_values_used=False, development_samples_read=0,
                    official_models_trained=False, no_prior_fit_loaded=True,
                    initial_model_sha256=initial_model_sha,
                    initial_frozen_sha256=initial_frozen_sha,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    all_parameters=sum(p.numel() for p in model.parameters()),
                    original_population_rows=512)
    write(output / 'manifest.json', manifest)
    print(json.dumps(dict(event='initialized', schema=SCHEMA,
                          initial_model_sha256=initial_model_sha,
                          initial_frozen_sha256=initial_frozen_sha),
                     allow_nan=False), flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=TRAINING['lr'],
                                  betas=tuple(TRAINING['betas']),
                                  eps=TRAINING['eps'],
                                  weight_decay=TRAINING['weight_decay'])
    optimizer.zero_grad(set_to_none=True)
    model.train()
    try:
        update = 0
        for epoch, order in enumerate(orders):
            for position, ordinal in enumerate(order):
                check()
                item = sample(ordinal)
                prediction = model.current(item['tokens'].cuda())
                loss, detail = occ_loss(prediction, item['coarse'])
                require(bool(torch.isfinite(loss)), 'Nonfinite codec loss')
                loss.backward()
                require(all(parameter.grad is None or
                            bool(torch.isfinite(parameter.grad).all())
                            for parameter in trainable),
                        'Nonfinite codec gradient')
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable,
                                                           TRAINING['grad_clip_norm'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                append(output / 'training.jsonl', dict(stage='codec', epoch=epoch,
                    position=position, update=update, ordinal=ordinal,
                    sample_token=item['sample_token'], total=float(loss.detach()),
                    grad_norm=float(grad_norm), **detail,
                    elapsed_seconds=time.monotonic() - started))
                if update % 64 == 0:
                    print(json.dumps(dict(event='progress', update=update,
                                          epoch=epoch, ordinal=ordinal),
                                     allow_nan=False), flush=True)
                del prediction, loss
        require(update == TRAINING['updates'], 'Actual update count mismatch')

        frozen_final_sha = {
            name: native._parameter_digest(getattr(model, name))
            for name in ('dynamics', 'content_increment', 'velocity')}
        require(frozen_final_sha == initial_frozen_sha,
                'Frozen recurrent parameters changed')

        # Evaluate every original train sample at native fine XYZ resolution.
        model.eval()
        histogram = np.zeros((2, 2), dtype=np.int64)
        per_sample = []
        with torch.no_grad():
            for ordinal in range(512):
                item = sample(ordinal)
                logits = model.current(item['tokens'].cuda())
                fine_logits = F.interpolate(logits, size=FINE_SHAPE,
                                            mode='trilinear', align_corners=False)
                prediction = fine_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                target = item['fine'][0]
                valid = target != 255
                counts = np.bincount((2 * target[valid] + prediction[valid]).astype(np.int64),
                                     minlength=4).reshape(2, 2)
                histogram += counts
                per_sample.append(dict(ordinal=ordinal,
                                       sample_token=item['sample_token'],
                                       hist=counts.tolist()))
        require(len(populations) == 512,
                'Original population receipt is incomplete')
        tp, fp, fn = histogram[1, 1], histogram[0, 1], histogram[1, 0]
        require(tp + fp + fn > 0, 'Empty current occupancy evaluation union')
        iou = float(tp / (tp + fp + fn))
        final_model_sha = native._parameter_digest(model)
        payload = dict(schema=SCHEMA, status='COMPLETE_CURRENT_CODEC512',
                       protocol_sha256=args.protocol_sha256,
                       state_dict={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()},
                       initial_model_sha256=initial_model_sha,
                       final_model_sha256=final_model_sha,
                       initial_frozen_sha256=initial_frozen_sha,
                       frozen_final_sha256=frozen_final_sha,
                       updates=update, current_only_supervision=True)
        temporary = output / 'shared_codec.pth.tmp'
        torch.save(payload, str(temporary)); temporary.replace(output / 'shared_codec.pth')
        manifest.update(status='COMPLETE_CURRENT_CODEC512', populations=list(populations.values()),
                        frozen_final_sha256=frozen_final_sha,
                        final_model_sha256=final_model_sha,
                        final_current_hist=histogram.tolist(),
                        final_current_iou=iou,
                        completed_updates=update)
        write(output / 'manifest.json', manifest)
        write(output / 'evaluation.json', dict(schema=SCHEMA, resolution='native fine XYZ',
            current_only=True, histogram=histogram.tolist(), iou=iou,
            samples=per_sample))
        complete = dict(schema=SCHEMA, status='COMPLETE_CURRENT_CODEC512',
                        protocol_sha256=args.protocol_sha256,
                        updates=update,
                        elapsed_seconds=time.monotonic()-started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        files_sha256={name: sha(output / name) for name in
                                      ['protocol.json', 'manifest.json', 'samples.jsonl',
                                       'training.jsonl', 'evaluation.json',
                                       'shared_codec.pth']})
        write(output / 'complete.json', complete)
        print(json.dumps(dict(event='complete', status=complete['status'],
                              updates=update, current_iou=iou,
                              elapsed_seconds=time.monotonic() - started),
                         allow_nan=False), flush=True)
    except BaseException as error:
        if output.is_dir():
            write(output / 'failed.json', dict(schema=SCHEMA, error=repr(error),
                                              traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'protocol-sha256', 'train-cache', 'repo', 'out',
                 'max-seconds', 'max-allocated-gib'):
        parser.add_argument('--' + name, required=True,
                            type=float if name == 'max-allocated-gib' else
                            int if name == 'max-seconds' else str)
    main(parser.parse_args())
