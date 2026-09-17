"""Fixed two-anchor, four-AdamW-update engineering profile for F and O.

Probe weights are discarded. This is neither formal training nor a performance
comparison. No development data, coordinate adapter, threshold selection, or
checkpoint output is permitted. Frozen native sources and data are read-only.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import signal
import sys
import time

FULL_SOURCE_SHA = '12698cb2bb071dd6daf79ccc64ba8f506d0a773a676dd57296f8410fe747d0f7'


def imports():
    import hashlib
    full = Path(__file__).with_name('full_resolution_supervision_preflight.py')
    if hashlib.sha256(full.read_bytes()).hexdigest() != FULL_SOURCE_SHA:
        raise ValueError('Frozen full-supervision engineering source changed')
    import full_resolution_supervision_preflight as base
    return base


def selection(base, a, parent):
    from memory_experiment import cached_records
    rows = cached_records(a.cache)
    base.require(len(rows) == 512 and all(r['split'] == 'train' for r in rows), 'Only original train512 cache is permitted')
    path = Path(a.protocol).with_name('selection_v1.json')
    base.require(base.sha(path) == parent['selection_sha256'], 'Selection SHA differs')
    keys = ('sample_token', 'scene_token', 'official_index', 'split')
    expected = [r for r in base.read(path)['records'] if r['split'] == 'train']
    base.require([[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in expected], 'Frozen cache identity/order differs')
    chosen = []; scenes = set()
    for row in rows:
        if row['scene_token'] not in scenes:
            chosen.append(row); scenes.add(row['scene_token'])
        if len(chosen) == 2:
            break
    base.require(tuple(r['sample_token'] for r in chosen) == base.PAIR_TOKENS and len(scenes) == 2,
                 'Fixed first two distinct training anchors changed')
    return chosen


def initial_parity(base, model, cache, records, expected=None):
    import torch
    from memory_experiment import seed_all
    from native_state_cache import load_sample, replay
    output = []
    for record in records:
        sample = load_sample(cache, record, device='cuda')
        seed_all(11)
        with torch.no_grad():
            prediction = replay(model, sample, training=False)[0]
        base.require(torch.isfinite(prediction).all().item(), 'Nonfinite initial prediction')
        row = dict(sample_token=record['sample_token'], scene_token=record['scene_token'],
            prediction_sha256=base.tensor_digest(prediction),
            native_evaluator_record=base.metric_record(model, prediction, sample))
        if expected is not None:
            base.require(row == expected[len(output)], 'Initial five-horizon/all-layer predictions or full GT evaluator differ')
        output.append(row)
        del sample, prediction
    return output


def gradients(base, params):
    """Inspect whole-head accumulated gradients before the unchanged clip."""
    import torch
    missing = []; zero = []; nonzero = []
    for name, value in params.items():
        grad = value.grad
        if grad is None:
            missing.append(name)
        else:
            base.require(torch.isfinite(grad).all().item(), 'Nonfinite accumulated head gradient: '+name)
            (nonzero if torch.count_nonzero(grad).item() else zero).append(name)
    base.require(bool(nonzero), 'No nonzero future-head gradient')
    return dict(None_parameter_names=missing, zero_gradient_parameter_names=zero,
                nonzero_gradient_parameter_names=nonzero, all_finite=True)


def train_probe(base, model, params, records, a, arm, out, reports):
    import torch
    from memory_experiment import seed_all, state_digest
    from native_state_cache import load_sample, replay
    seed_all(11)
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-5, weight_decay=.01)
    monitored = next(name for name in params if '.transformer.' in name and params[name].numel() > 256)
    before = params[monitored].detach().clone()
    updates = []; micros = []; peak_allocated = 0; peak_reserved = 0
    for update in range(4):
        torch.cuda.synchronize(); started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        losses_in_update = []
        for micro in range(4):
            ordinal = update*4+micro; record = records[ordinal % 2]
            begin = time.monotonic(); sample = load_sample(a.cache, record, device='cuda')
            torch.cuda.synchronize(); load_seconds = time.monotonic()-begin
            begin = time.monotonic(); prediction = replay(model, sample, training=True)[0]
            torch.cuda.synchronize(); forward_seconds = time.monotonic()-begin
            base.require(model.training and prediction.requires_grad, 'Expected actual training-mode gradient path')
            begin = time.monotonic(); losses = model.compute_occ_loss(prediction, sample['targets'])
            total = sum(losses.values())
            base.require(torch.isfinite(total).item(), 'Nonfinite selected training objective')
            audit = dict(model._objective_last_loss_audit)
            torch.cuda.synchronize(); loss_seconds = time.monotonic()-begin
            begin = time.monotonic(); (total/4).backward()
            torch.cuda.synchronize(); backward_seconds = time.monotonic()-begin
            value = float(total.detach().item()); losses_in_update.append(value)
            micros.append(dict(arm=arm, ordinal=ordinal, update=update+1, sample_token=record['sample_token'],
                scene_token=record['scene_token'], input_sha256=record['files']['inputs']['sha256'],
                target_sha256=record['files']['targets']['sha256'],
                temperature='first_training_micro_after_eval_imports' if ordinal == 0 else ('warmup' if update < 2 else 'warm'),
                load_device_seconds=load_seconds, forward_device_seconds=forward_seconds,
                loss_device_seconds_including_scalar_audit=loss_seconds, backward_device_seconds=backward_seconds,
                selected_loss=value, original_twelve_loss_audit=audit))
            del sample, prediction, losses, total
        gradient = gradients(base, params)
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 35, error_if_nonfinite=True)
        base.require(math.isfinite(float(norm)) and float(norm) > 0, 'Invalid complete-head gradient norm')
        monitored_gradient = params[monitored].grad
        base.require(monitored_gradient is not None and torch.isfinite(monitored_gradient).all().item(), 'No transition gradient')
        monitored_norm = float(monitored_gradient.norm().item())
        base.require(monitored_norm > 0, 'Zero transition gradient')
        begin = time.monotonic(); optimizer.step(); torch.cuda.synchronize()
        optimizer_seconds = time.monotonic()-begin
        base.require(all(torch.isfinite(value).all().item() for value in params.values()), 'Nonfinite head weight after AdamW')
        allocated = int(torch.cuda.max_memory_allocated()); reserved = int(torch.cuda.max_memory_reserved())
        peak_allocated = max(peak_allocated, allocated); peak_reserved = max(peak_reserved, reserved)
        row = dict(update=update+1, examples_cumulative=(update+1)*4, warmup=update < 2,
            mean_selected_loss=sum(losses_in_update)/4, preclip_head_gradient_L2=float(norm),
            monitored_postclip_gradient_L2=monitored_norm, whole_head_gradient_check=gradient,
            optimizer_step_device_seconds=optimizer_seconds, update_wall_seconds=time.monotonic()-started,
            peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, lr=1e-5, weight_decay=.01, clip=35)
        updates.append(row)
        base.event('OBJECTIVE_PROFILE_UPDATE', arm=arm, update=update+1, seconds=row['update_wall_seconds'], peak_allocated_bytes=allocated)
        base.write(out/'progress.json', dict(status='ENGINEERING_OPTIMIZER_PROBE', completed_arms=reports,
            current_arm=arm, optimizer_updates=updates, microsteps=micros, probe_weights_saved=False))
    difference = float((params[monitored]-before).abs().max().item())
    base.require(math.isfinite(difference) and difference > 0, 'AdamW did not update the transition')
    final_sha = state_digest(model.future_pred_head)
    base.require(final_sha != base.INITIAL_HEAD_SHA, 'Probe head did not change')
    result = dict(optimizer_updates=updates, microsteps=micros, all_gradients_finite=True,
        examples=16, actual_updates=4, accumulation=4, parameter_max_abs_change=difference,
        monitored_parameter=monitored, final_discarded_head_sha256=final_sha,
        warm_update_wall_seconds_mean=sum(r['update_wall_seconds'] for r in updates[2:])/2,
        maximum_cuda_allocated_bytes=peak_allocated, maximum_cuda_reserved_bytes=peak_reserved,
        first_training_micro=micros[0], warm_micro_device_seconds_mean={key:sum(r[key] for r in micros[8:])/8 for key in
            ('forward_device_seconds','loss_device_seconds_including_scalar_audit','backward_device_seconds')})
    optimizer.zero_grad(set_to_none=True)
    del before, optimizer
    return result


def render(out, summary):
    lines = ['# F/O 实际 AdamW 工程预检', '',
        '仅使用固定两条 train anchor；每臂 4 次更新、累积 4 微步，所有更新后的权重丢弃。完整开发集未读取，本次不产生性能结果。', '',
        '| 臂 | 更新/样本次 | 热更新均秒 | 实际训练 peak allocated GiB | 初始预测/完整 GT 评测一致 |',
        '|---|---:|---:|---:|---|']
    for r in summary['reports']:
        lines.append('| {} | 4 / 16 | {:.3f} | {:.3f} | {} |'.format(r['arm'],r['warm_update_wall_seconds_mean'],r['maximum_cuda_allocated_bytes']/2**30,r['initial_prediction_and_fullGT_hist_exact']))
    lines += ['', 'F 只把原 12 项损失的监督网格改为完整 GT。O 仍计算原 coarse 12 项，然后只让 CE[1,5] 与 Lovász 六项进入总目标；保留的 sem/geo 前向开销及 detached 标量检查已计入耗时。两种目标的 loss 数值不能直接比较。', '',
        '前两次更新标为 warmup，后两次标为 warm。首个 training 微步发生在依赖导入与初值评测之后，不是进程完全冷启动；其详细时间与后续微步均保留。峰值包含真实 backward、累积梯度、clip 与 AdamW 状态，和上一轮 eval VJP 探针不同。', '',
        '两条样本不能保证全 train512 的最坏显存/吞吐。PASS 仅通过工程检查，不自动启动正式训练或证明优于 M0。', '']
    (out/'report.md').write_text('\n'.join(lines))


def run(a, out, started):
    base = imports(); parent, frozen_sources, runtime = base.verify_contract(a)
    sys.path.insert(0, str(Path(a.repo).resolve()))
    before_import = time.monotonic()
    import torch
    from memory_experiment import prepare_model, seed_all, state_digest
    from objective_supervision_adapters import install_objective_supervision
    base.require(torch.cuda.is_available() and a.device.startswith('cuda'), 'GPU required')
    torch.cuda.set_device(torch.device(a.device)); seed_all(11)
    total_memory = torch.cuda.get_device_properties(torch.device(a.device)).total_memory
    cap = int(a.max_allocated_gib*2**30)
    base.require(0 < cap <= total_memory, 'Invalid allocator cap')
    torch.cuda.set_per_process_memory_fraction(cap/total_memory, device=torch.device(a.device))
    dependencies_seconds = time.monotonic()-before_import
    records = selection(base,a,parent)
    hp = parent['training']
    base.require(hp['seed'] == 11 and hp['accumulate'] == 4 and hp['lr'] == 1e-5 and hp['weight_decay'] == .01 and hp['grad_clip'] == 35, 'Parent optimization rule changed')
    sources = {name:base.sha(Path(__file__).with_name(name)) for name in
               ('objective_supervision_adapters.py','objective_supervision_preflight.py','full_resolution_supervision_preflight.py')}
    manifest = dict(schema='m0-objective-supervision-preflight-manifest-v1',
        parent_protocol_sha256=base.PARENT_SHA, config_sha256=base.sha(a.config), m0_sha256=base.sha(a.checkpoint),
        train_cache_index_sha256=base.TRAIN_INDEX_SHA, source_sha256=sources,
        parent_source_sha256=frozen_sources, runtime_source_sha256=runtime,
        identities=[{k:r[k] for k in ('sample_token','scene_token','official_index','split')} for r in records],
        input_target_sources=[dict(sample_token=r['sample_token'],inputs_sha256=r['files']['inputs']['sha256'],targets_sha256=r['files']['targets']['sha256']) for r in records],
        source_selection_sha256=parent['selection_sha256'], seed=11, precision=parent['numerical_policy'],
        arms=['F','O'], updates_per_arm=4, accumulation=4, examples_per_arm=16,
        optimizer=dict(name='AdamW',lr=1e-5,weight_decay=.01,grad_clip=35,preflight_constant_lr=True),
        maximum_seconds=a.max_seconds, allocator_cap_bytes=cap, training_mode=True,
        dependency_import_cuda_seconds=dependencies_seconds, development_data_read=False,
        checkpoint_saved=False, automatic_formal_training=False)
    base.write(out/'manifest.json',manifest)
    reports=[]; expected=None; baseline_load_seconds=None
    for arm in ('native_fp32','F','O'):
        seed_all(11); begin=time.monotonic(); model,migration,params=prepare_model(a.config,a.checkpoint,'native1')
        torch.cuda.synchronize(); model_seconds=time.monotonic()-begin
        initial_sha=state_digest(model.future_pred_head)
        base.require(initial_sha == base.INITIAL_HEAD_SHA and sum(p.numel() for p in params.values()) == 13274016,
                     'Wrong M0 initialization/whole-head parameter count')
        base.require(list(params) == [n for n,p in model.named_parameters() if p.requires_grad] and
                     all(n.startswith('future_pred_head.') for n in params), 'Gradient scope differs')
        adapter = None if arm == 'native_fp32' else install_objective_supervision(model,arm)
        base.require(state_digest(model.future_pred_head) == initial_sha, 'Loss installation changed weights')
        parity=initial_parity(base,model,a.cache,records,expected)
        base.require(state_digest(model.future_pred_head) == initial_sha, 'Initial evaluation changed weights')
        if arm == 'native_fp32':
            expected=parity; baseline_load_seconds=model_seconds
        else:
            result=train_probe(base,model,params,records,a,arm,out,reports)
            row=dict(arm=arm,adapter=adapter,initial_head_sha256=initial_sha,
                trainable_parameters=sum(p.numel() for p in params.values()),trainable_parameter_names=list(params),
                initial_parity=parity,initial_prediction_and_fullGT_hist_exact=True,
                model_load_seconds=model_seconds,**result)
            reports.append(row)
        del model,params;gc.collect();torch.cuda.empty_cache()
    base.require(all(base.sha(Path(__file__).with_name(name)) == digest for name,digest in sources.items()), 'New source changed while running')
    base.require(base.sha(a.checkpoint) == manifest['m0_sha256'], 'Original checkpoint changed')
    summary=dict(schema='m0-objective-supervision-preflight-v1',status='PASS_OBJECTIVE_SUPERVISION_ENGINEERING_PREFLIGHT',
        manifest_sha256=base.sha(out/'manifest.json'),native_fp32_initial=expected,native_model_load_seconds=baseline_load_seconds,
        reports=reports,checkpoint_saved=False,development_data_read=False,automatic_formal_training=False,
        seconds=time.monotonic()-started)
    base.write(out/'summary.json',summary);render(out,summary)
    base.write(out/'complete.json',dict(status=summary['status'],summary_sha256=base.sha(out/'summary.json'),
        manifest_sha256=base.sha(out/'manifest.json'),report_sha256=base.sha(out/'report.md'),
        updates_per_arm=4,examples_per_arm=16,checkpoint_saved=False,performance_claim=False,
        automatic_formal_training=False,seconds=time.monotonic()-started))
    base.event('OBJECTIVE_PREFLIGHT_COMPLETE',seconds=time.monotonic()-started)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','config','checkpoint','protocol','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--train-cache','--cache',dest='cache',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--max-allocated-gib',type=float,default=64.)
    a=p.parse_args(argv)
    if not (0<a.max_seconds<=600 and math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=64):
        raise ValueError('Preflight limited to 600 seconds and 64 GiB allocator')
    base=imports();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def stop(number,_frame):raise InterruptedError('Preflight signal/deadline '+str(number))
    for number in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(number,stop)
    signal.alarm(a.max_seconds)
    try:
        run(a,out,started)
    except BaseException as exc:
        base.write(out/'failed.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=repr(exc),
            checkpoint_saved=False,automatic_formal_training=False,seconds=time.monotonic()-started))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
