"""Bounded train-only fit of the own-state/identity-readout pair.

This is an engineering learning-curve experiment, not a dev-set comparison
with O. No existing model/checkpoint is modified. Final weights are retained
for inspection only; any larger experiment needs an explicit fresh protocol.
"""
import argparse
import copy
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


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def run(a):
    start = time.monotonic()
    p = json.loads(Path(a.protocol).read_text())
    assert sha(a.protocol) == a.protocol_sha256 and p['status'] == 'FROZEN'
    assert p['schema'] == 'dense-task-state-fit-v2'
    root = Path(__file__).parent
    for name, digest in p['sources_sha256'].items():
        assert Path(name).name == name and sha(root / name) == digest, name
    for path, digest in p['runtime_source_sha256'].items():
        assert sha(path) == digest, path
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', p)
    import torch
    import torch.nn.functional as F
    import native_state_cache as native
    import train_source_motion_v1 as helper
    from dense_task_state_v2 import DenseTaskState, geometry_checks
    native._prepare_repo(a.repo)
    lossmod = importlib.import_module('projects.mmdet3d_plugin.bevformer.dense_heads.world_head_v1')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    assert free >= 32 * 2**30, 'Less than 32 GiB free; do not touch another task'
    torch.cuda.set_per_process_memory_fraction(p['resources']['max_allocated_gib'] * 2**30 / total)
    torch.cuda.reset_peak_memory_stats()
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stopped.append(signum))

    def check():
        assert not stopped, 'Termination requested'
        assert time.monotonic() - start < p['resources']['max_seconds'], 'Bounded fit timed out'
        assert torch.cuda.max_memory_allocated() / 2**30 <= p['resources']['max_allocated_gib'], 'Memory cap'

    torch.manual_seed(11); np.random.seed(11)
    geometry = geometry_checks('cuda')
    write(out / 'geometry_checks.json', geometry)
    trainroot, records, receipt = helper.cache_index(a.train_cache, 'train', p)
    labelroot, labels = helper.labels_manifest(a.sparse_labels, p)
    chosen = np.random.RandomState(11).permutation(512)[:p['fit_samples']].tolist()
    manifest = dict(protocol_sha256=a.protocol_sha256, selected_ordinals=chosen,
        records=[{k: records[i][k] for k in ('sample_token', 'scene_token', 'official_index', 'split')} for i in chosen],
        cache_receipt=receipt, inputs='only prev_bev_input[:,-1]; no metadata, future ego, O logits or labels',
        target='legacy occupied-mode 512x512x40 -> 256x256x20, original class mapping',
        development_samples_read=0, official_models_trained=False, seed=11,
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, initial_free_gib=free/2**30))
    write(out / 'manifest.json', manifest)
    # Inputs authenticate independently and the first model call precedes any
    # target read. After this check, labels are cached on CPU for training only.
    torch.manual_seed(11)
    model = DenseTaskState().cuda()
    dataset = []
    for i in chosen:
        check(); r = records[i]
        tokens = helper.load_tokens(trainroot, r, native, 'cuda')
        with torch.no_grad():
            initial = model.current(tokens)
            assert initial.shape == (1, 2, 200, 200, 16) and torch.isfinite(initial).all()
        directory = native._sample_directory(trainroot, r)
        descriptor = r['files']['targets']; path = directory / descriptor['file']
        native._verify_file(path, descriptor)
        raw = np.load(path, allow_pickle=False)
        assert raw.dtype == np.uint8 and raw.shape == (1, 7, 512, 512, 40)
        assert np.isin(raw, [0, 1, 255]).all()
        target = torch.from_numpy(raw[0, 2:].copy()).cuda().long()
        target = lossmod._downsample_occ_target(target).cpu()
        del raw
        label = helper.load_sparse(labelroot, labels[r['sample_token']], r)
        dataset.append(dict(tokens=tokens.cpu(), target=target, label=label))
        del initial, tokens
    del target
    manifest['sample_dt_seconds'] = [d['label']['dt_future_seconds'].tolist() for d in dataset]
    manifest['nominal_dt_seconds'] = [.5, 1., 1.5, 2.]
    manifest['sample_dt_offset_seconds'] = [(d['label']['dt_future_seconds']-np.array([.5,1.,1.5,2.])).tolist() for d in dataset]
    manifest['decoder_coverage_input'] = False
    write(out / 'manifest.json', manifest)
    torch.cuda.empty_cache()

    weights = torch.tensor([1., 5.], device='cuda')

    def occ_loss(logits, target):
        # One decoder. Same original CE+Lovasz terms, not three replicated
        # intermediate heads and not a claim of identical O training budget.
        resized = F.interpolate(logits, size=(256, 256, 20), mode='trilinear', align_corners=False)
        ce = lossmod.CE_ssc_loss(resized, target, weights, ignore_index=255)
        lovasz = lossmod.lovasz_softmax(torch.softmax(resized, dim=1), target, ignore=255)
        return ce + lovasz, dict(ce=float(ce.detach()), lovasz=float(lovasz.detach()))

    def record(row):
        check(); row.update(seconds=time.monotonic()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n'); f.flush()
        print(json.dumps(row, allow_nan=False), flush=True)

    def update(opt, loss, parameters):
        assert torch.isfinite(loss), 'Nonfinite loss'
        loss.backward()
        params = list(parameters)
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params), 'Nonfinite gradient'
        norm = torch.nn.utils.clip_grad_norm_(params, 10.)
        opt.step(); opt.zero_grad(set_to_none=True)
        return float(norm)

    def codec_scores():
        scores = []
        with torch.no_grad():
            for d in dataset:
                check()
                loss, detail = occ_loss(model.current(d['tokens'].cuda()), d['target'][:1].cuda())
                scores.append(dict(total=float(loss), **detail))
        return scores

    before_codec = codec_scores()
    codec_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.AdamW(codec_parameters, lr=p['lr'], weight_decay=.01)
    for step in range(p['codec_updates']):
        check(); d = dataset[step % len(dataset)]
        loss, detail = occ_loss(model.current(d['tokens'].cuda()), d['target'][:1].cuda())
        norm = update(opt, loss, codec_parameters)
        record(dict(stage='codec', step=step+1, sample_index=step%len(dataset), total=float(loss.detach()), grad_norm=norm, **detail))
        del loss
    after_codec = codec_scores()
    shared = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=shared, protocol_sha256=a.protocol_sha256), out / 'shared_codec.pth')
    del opt, model
    summary = dict(scope='train-only engineering fit; no O comparison or generalization claim',
                   before_codec=before_codec, after_codec=after_codec, arms={})
    write(out / 'summary.json', summary)

    def arm_scores(model, arm):
        result = []
        with torch.no_grad():
            for d in dataset:
                check(); tokens = d['tokens'].cuda()
                forecast = model(tokens, arm)
                occ, detail = occ_loss(forecast['logits'][0], d['target'].cuda())
                physical, _ = helper.object_group_loss(helper.gather_sparse(forecast['displacement'], d['label']), d['label'])
                zero = model(tokens, arm, 'zero')['logits']
                reverse = model(tokens, arm, 'reverse')['logits']
                # Decoder input is the only content path. Also measure sensitivity
                # to a zero-content destination with neutral coverage.
                absent = model.decode(torch.zeros(1, 8, 200, 200, 16, device='cuda'),
                                      torch.ones(1, 1, 200, 200, 16, device='cuda'))
                sensitivity = dict(zero_address_mean_logit_change=float((forecast['logits'][:, 1:]-zero[:, 1:]).abs().mean()),
                    reverse_address_mean_logit_change=float((forecast['logits'][:, 1:]-reverse[:, 1:]).abs().mean()),
                    removed_content_mean_logit_change=float((forecast['logits'][:, 1:]-absent[:, None]).abs().mean()))
                zero_occ, _ = occ_loss(zero[0], d['target'].cuda())
                reverse_occ, _ = occ_loss(reverse[0], d['target'].cuda())
                result.append(dict(occupancy=float(occ), physical=float(physical), zero_address_occupancy=float(zero_occ),
                    reverse_address_occupancy=float(reverse_occ), **detail, **sensitivity))
        return result

    for arm in ('transport', 'direct'):
        torch.manual_seed(11)
        model = DenseTaskState().cuda(); model.load_state_dict(shared, strict=True)
        initial_digest = native._parameter_digest(model)
        before = arm_scores(model, arm)
        opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=.01)
        connectivity = None
        for step in range(p['rollout_updates_per_arm']):
            check(); d = dataset[step % len(dataset)]
            forecast = model(d['tokens'].cuda(), arm)
            occ, detail = occ_loss(forecast['logits'][0], d['target'].cuda())
            physical, _ = helper.object_group_loss(helper.gather_sparse(forecast['displacement'], d['label']), d['label'])
            if step == 0:
                grad = torch.autograd.grad(occ, model.velocity.weight, retain_graph=True, allow_unused=True)[0]
                connectivity = None if grad is None else float(grad.norm())
                assert (grad is None) if arm == 'direct' else (grad is not None and torch.isfinite(grad).all() and grad.norm() > 0)
                del grad
            loss = occ + p['physical_weight'] * physical
            norm = update(opt, loss, model.parameters())
            record(dict(stage=arm, step=step+1, sample_index=step%len(dataset), total=float(loss.detach()),
                occupancy=float(occ.detach()), physical=float(physical.detach()), grad_norm=norm, **detail))
            del forecast, loss, occ, physical
        after = arm_scores(model, arm)
        path = out / (arm + '_final.pth')
        torch.save(dict(state_dict=model.state_dict(), protocol_sha256=a.protocol_sha256, seed=11,
                        fixed_final_update=p['rollout_updates_per_arm'], engineering_only=True), path)
        summary['arms'][arm] = dict(before=before, after=after, initial_parameters_sha256=initial_digest,
            parameters=sum(v.numel() for v in model.parameters()), initial_occupancy_velocity_gradient=connectivity,
            checkpoint_sha256=sha(path))
        write(out / 'summary.json', summary)
        del model, opt; torch.cuda.empty_cache()
    assert summary['arms']['transport']['initial_parameters_sha256'] == summary['arms']['direct']['initial_parameters_sha256']
    assert summary['arms']['transport']['parameters'] == summary['arms']['direct']['parameters']
    summary.update(seconds=time.monotonic()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write(out / 'summary.json', summary)
    write(out / 'complete.json', dict(status='COMPLETE_TRAIN_ONLY_ENGINEERING_FIT', protocol_sha256=a.protocol_sha256,
        files_sha256={name: sha(out/name) for name in ['manifest.json', 'geometry_checks.json', 'summary.json', 'training.jsonl',
            'shared_codec.pth', 'transport_final.pth', 'direct_final.pth']},
        evaluated_development_samples=0, seed=11, updates=p['codec_updates']+2*p['rollout_updates_per_arm']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'protocol-sha256', 'train-cache', 'sparse-labels', 'repo', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        if Path(args.out).is_dir():
            write(Path(args.out) / 'failed.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
