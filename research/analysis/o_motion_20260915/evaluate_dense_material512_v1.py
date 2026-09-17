"""Fixed final T/J/D dev200 predictions scored in the original O domain.

Each invocation evaluates one completed arm and authenticated saved O counts.
No fit, model selection, prediction threshold, crop, or O flow-EPE surrogate.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def physical_statistics(displacement, label, helper):
    prediction = helper.gather_sparse(displacement, label).cpu().numpy()
    errors = np.linalg.norm(prediction-label['target_displacement_m'], axis=-1)
    magnitudes = np.linalg.norm(prediction, axis=-1)
    zero_errors = np.linalg.norm(label['target_displacement_m'], axis=-1)
    results = []
    for h in range(4):
        row = dict(horizon_seconds=(h+1)*.5, groups={})
        for group, name in enumerate(('stationary', 'ambiguous', 'moving')):
            values = []
            for obj, g in enumerate(label['object_speed_group'][h]):
                if g != group:
                    continue
                mask = label['valid'][h] & (label['object_index'] == obj)
                assert mask.any()
                values.append(dict(points=int(mask.sum()),
                    epe=float(errors[h, mask].mean(dtype=np.float64)),
                    magnitude=float(magnitudes[h, mask].mean(dtype=np.float64)),
                    zero_epe=float(zero_errors[h, mask].mean(dtype=np.float64))))
            row['groups'][name] = dict(objects=len(values), points=sum(v['points'] for v in values),
                **{key+'_object_sum':sum(v[key] for v in values)
                   for key in ('epe', 'magnitude', 'zero_epe')})
        results.append(row)
    return results


def pooled_physical(rows):
    output = []
    for h in range(4):
        row = dict(horizon_seconds=(h+1)*.5, groups={})
        for group in ('stationary', 'ambiguous', 'moving'):
            pieces = [r['physical'][h]['groups'][group] for r in rows]
            count = sum(p['objects'] for p in pieces)
            item = dict(objects=count, points=sum(p['points'] for p in pieces))
            for key in ('epe', 'magnitude', 'zero_epe'):
                total = sum(p[key+'_object_sum'] for p in pieces)
                item[key+'_object_sum'] = total
                item[key] = total/count if count else None
            row['groups'][group] = item
        output.append(row)
    return output


def run(a):
    started = time.monotonic()
    for name in ('protocol', 'fit', 'dev_cache', 'raw_labels', 'sparse_labels',
                 'o_records', 'repo', 'out'):
        setattr(a, name, str(Path(getattr(a, name)).resolve()))
    p = read(a.protocol)
    assert p['schema'] == 'dense-material512-evaluation-v1' and p['status'] == 'FROZEN'
    assert sha(a.protocol) == a.protocol_sha256
    for name, digest in p['sources_sha256'].items():
        assert Path(name).name == name and sha(Path(__file__).parent/name) == digest, name
    for name, digest in p['runtime_source_sha256'].items():
        assert sha(name) == digest, name
    assert a.max_seconds == p['resources']['max_seconds']
    assert a.max_allocated_gib == p['resources']['max_allocated_gib']
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', p)
    import torch
    import native_state_cache as native
    import train_source_motion_v1 as helper
    import common_change_evaluation_v2 as common
    import common_connected_motion_evaluation_v2 as connected
    import common_occupancy_change_metrics_v1 as metric
    from dense_material_state_v1 import DenseMaterialState
    native._prepare_repo(a.repo)
    os.chdir(a.repo)
    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    free, total = torch.cuda.mem_get_info()
    assert free >= 24*2**30
    torch.cuda.set_per_process_memory_fraction(a.max_allocated_gib*2**30/total)
    stop = []
    signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))
    signal.signal(signal.SIGINT, lambda s, f: stop.append(s))

    def check():
        assert not stop, 'Stop requested'
        assert time.monotonic()-started < a.max_seconds, 'Time cap reached'
        assert torch.cuda.max_memory_allocated() <= a.max_allocated_gib*2**30

    fit = Path(a.fit)
    assert not (fit/'failed.json').exists()
    assert sha(fit/'complete.json') == a.fit_complete_sha256
    done = read(fit/'complete.json')
    assert done['schema'] == 'dense-material512-training-v1'
    assert done['status'] == 'COMPLETE_DENSE_MATERIAL512_TRAIN'
    assert done['arm'] == a.arm and done['updates'] == 2048
    assert done['protocol_sha256'] == p['training_protocol_sha256']
    assert sha(fit/'protocol.json') == p['training_protocol_sha256']
    for name, digest in done['files_sha256'].items():
        assert Path(name).name == name and sha(fit/name) == digest, name
    assert {'model_final.pth', 'manifest.json', 'training.jsonl'} <= set(done['files_sha256'])
    manifest_fit = read(fit/'manifest.json')
    assert manifest_fit['status'] == done['status'] and manifest_fit['completed_updates'] == 2048
    assert manifest_fit['arm'] == a.arm and manifest_fit['development_samples_read'] == 0
    assert manifest_fit['initial_parameters_sha256'] == p['codec_final_parameters_sha256']
    assert manifest_fit['initial_frozen_sha256'] == manifest_fit['frozen_final_sha256'] == p['codec_encoder_decoder_sha256']
    payload = torch.load(str(fit/'model_final.pth'), map_location='cpu', weights_only=False)
    assert payload['schema'] == done['schema'] and payload['status'] == done['status']
    assert payload['arm'] == a.arm and payload['updates'] == 2048
    assert payload['protocol_sha256'] == p['training_protocol_sha256']
    assert payload['material_mode'] == ('fixed' if a.arm == 'T' else 'evolving')
    assert payload['readout'] == ('direct' if a.arm == 'D' else 'transport')
    assert payload['initial_parameters_sha256'] == done['initial_parameters_sha256'] == p['codec_final_parameters_sha256']
    assert payload['initial_frozen_sha256'] == payload['frozen_final_sha256'] == payload['frozen_sha256'] == done['frozen_sha256'] == p['codec_encoder_decoder_sha256']
    model = DenseMaterialState(material_mode='fixed' if a.arm == 'T' else 'evolving').cuda()
    model.load_state_dict(payload['state_dict'], strict=True)
    initial = native._parameter_digest(model)
    assert initial == payload['final_parameters_sha256'] == done['final_parameters_sha256'] == manifest_fit['final_parameters_sha256']
    assert {name:native._parameter_digest(getattr(model, name)) for name in ('encoder', 'decoder')} == p['codec_encoder_decoder_sha256']
    model.requires_grad_(False)
    model.eval()
    del payload

    cache, records, receipt = helper.cache_index(a.dev_cache, 'development', p)
    assert len({r['scene_token'] for r in records}) == 100
    rawroot, descriptors = connected.label_descriptors(a.raw_labels, records, common)
    labelroot, labels = helper.labels_manifest(a.sparse_labels, p)
    assert sha(a.o_records) == p['o_records_sha256']
    reference = [json.loads(line) for line in Path(a.o_records).read_text().splitlines()]
    assert len(reference) == 200
    refs = {r['sample_token']: r for r in reference}
    assert len(refs) == 200
    # Freeze the evaluated population and all identities before predictions.
    for r, d in zip(records, descriptors):
        old = refs[r['sample_token']]
        assert common.identity(r) == common.identity(old)
        assert old['input_sha256'] == r['files']['inputs']['sha256']
        assert old['GT_cache_sha256'] == r['files']['targets']['sha256']
        assert old['raw_label_sha256'] == d['sha256']
        assert old['O_reference_hist_exact'] and old['native_CPU_hist_exact']
    manifest = dict(schema=p['schema'], arm=a.arm, anchors=[common.identity(r) for r in records],
        training_complete_sha256=sha(fit/'complete.json'),
        checkpoint_sha256=done['files_sha256']['model_final.pth'],
        parameters_sha256=initial, protocol_sha256=a.protocol_sha256,
        cache=receipt, o_records_sha256=p['o_records_sha256'],
        optimizer_updates=0, historical_development_exposure=True,
        t0_source='candidate own current codec; O retains its own current output',
        physical_scope='rigid raw-box displacement proxy at original GT source locations; O has no matched flow head')
    write(out/'manifest.json', manifest)
    rows = []
    for ordinal, (record, descriptor) in enumerate(zip(records, descriptors)):
        check()
        tokens = helper.load_tokens(cache, record, native, 'cuda')
        with torch.no_grad():
            forecast = model(tokens, 'direct' if a.arm == 'D' else 'transport')
            assert tuple(forecast['logits'].shape) == (1, 5, 2, 200, 200, 16)
            assert bool(torch.isfinite(forecast['displacement']).all())
            # The common converter is reused with an explicit already-XYZ adapter.
            pred = common.fine_binary(forecast['logits'][0],
                                     SimpleNamespace(predictions_to_xyz=lambda x: x), check)
            # Label content is first opened after complete fine predictions.
            temporary = {}
            gt = common.load_targets_after_predictions(native, cache, record, temporary, 'cpu')
            del temporary
            rawpath = rawroot/descriptor['file']
            assert sha(rawpath) == descriptor['sha256']
            with gzip.open(rawpath, 'rt') as f:
                raw = json.load(f)
            assert raw['identity'] == common.identity(record)
            scores = metric.evaluate_common_occupancy_change(pred, gt, model.extent, raw)
            label = helper.load_sparse(labelroot, labels[record['sample_token']], record)
            phys = physical_statistics(forecast['displacement'], label, helper)
        old = refs[record['sample_token']]
        om = old['metrics_by_arm']['O']
        for h in range(5):
            ah = np.array(scores['horizons'][h]['occupancy']['confusion'], dtype=np.int64)
            oh = np.array(om['horizons'][h]['occupancy']['confusion'], dtype=np.int64)
            assert np.array_equal(ah.sum(1), oh.sum(1)), 'Original GT domain changed'
            assert scores['horizons'][h]['actual_dt_seconds'] == om['horizons'][h]['actual_dt_seconds']
            if h:
                for group in metric.POSITIVE_GROUPS:
                    assert scores['horizons'][h]['motion_positive_attribution']['groups'][group]['GT'] == om['horizons'][h]['motion_positive_attribution']['groups'][group]['GT']
        for item, olditem in zip(scores['transitions'], om['transitions']):
            assert item['domain'] == olditem['domain']
        row = dict(ordinal=ordinal, **common.identity(record),
            metrics_by_arm={a.arm:scores, 'O':om}, physical=phys,
            input_sha256=record['files']['inputs']['sha256'],
            GT_cache_sha256=record['files']['targets']['sha256'], raw_label_sha256=descriptor['sha256'],
            original_O_GT_domains_exact=True, predictions_completed_before_target_read=True,
            t0_equals_O=scores['t0_boundary']['prediction_full_binary_sha256'] == om['t0_boundary']['prediction_full_binary_sha256'])
        append(out/'records.jsonl', row)
        rows.append(row)
        write(out/'progress.json', dict(arm=a.arm, completed=len(rows), total=200,
                                        elapsed_seconds=time.monotonic()-started))
        del forecast, tokens, pred, gt, raw, label, scores
    assert native._parameter_digest(model) == initial
    summary = dict(common=common.summarize(rows, (a.arm, 'O'), metric),
                   physical=pooled_physical(rows),
                   anchors=200, scenes=100, optimizer_updates=0,
                   no_O_physical_EPE_claim=True,
                   t0_all_equal_O=all(r['t0_equals_O'] for r in rows),
                   caveat='Exposed dev200 only; not full5119 or independent confirmation; common-input T/J/D mechanism comparison and own-head rigid-proxy errors')
    write(out/'summary.json', summary)
    write(out/'complete.json', dict(status='COMPLETE_MATERIAL512_DEV200', arm=a.arm,
        protocol_sha256=a.protocol_sha256, anchors=200, scenes=100, optimizer_updates=0,
        elapsed_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        files_sha256={name:sha(out/name) for name in ('protocol.json', 'manifest.json', 'records.jsonl', 'summary.json')}))
    print(json.dumps(dict(event='complete', arm=a.arm, future_iou=summary['common'][a.arm]['future_macro_GMO_percent'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol', 'protocol-sha256', 'fit', 'dev-cache', 'raw-labels',
                 'sparse-labels', 'o-records', 'repo', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--arm', choices=('T', 'J', 'D'), required=True)
    parser.add_argument('--fit-complete-sha256', required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--max-allocated-gib', type=int, required=True)
    run(parser.parse_args())
