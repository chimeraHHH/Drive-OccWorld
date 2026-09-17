"""Reanalyze completed real soft-count receipts; no model, gradient, or training.

The ideal binary loss identity is algebra, not exact FP32/CUDA equivalence.
Serialized ratios are float64 calculations from saved float32 reductions.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
SUMMARY_SHA = 'f76c849cfd28fb74ec8c3ad38856d56b8ec47628813268970ae0d61ca6d08dd6'
LOSS_SHA = 'e6a4b9accef1f47cf031f84e18fac925d2764b4105b8010486c8229e589abaa4'
CONFIG_SHA = 'c92faf7c2a4697ae7faa0c69ca844068dd8f5a6d28463e7ad45307c4ab374303'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def loss_one(p):
    if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid observed ratio; do not clamp/repair the receipt')
    return min(-math.log(p), 100.) if p else 100.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE/'server_results/supervision_granularity_v1')
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    done = read(a.source/'complete.json')
    assert done['status'] == 'COMPLETE' and done['optimizer_updates'] == 0 and done['gradients'] is False
    assert done['summary_sha256'] == sha(a.source/'summary.json') == SUMMARY_SHA
    for name in ('manifest', 'report'):
        assert sha(a.source/(name+('.json' if name == 'manifest' else '.md'))) == done[name+'_sha256']
    summary = read(a.source/'summary.json')
    assert summary['samples'] == summary['scenes'] == 16
    workspace = BASE.parent.parent
    loss_path = workspace/'code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer/losses/semkitti_loss.py'
    config_path = workspace/'analysis/sota_p2_20260911/configs/S0.py'
    assert sha(loss_path) == LOSS_SHA and sha(config_path) == CONFIG_SHA
    assert set(done['records_sha256']) == {'M0_fp32','native1','persistent2','rolling2'}
    rows = []; identities = None
    for model, digest in done['records_sha256'].items():
        path = a.source/(model+'_records.jsonl'); assert sha(path) == digest
        records = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [(r['sample_token'],r['scene_token'],r['official_index']) for r in records]
        assert len(records) == len(set(ids)) == 16 and len({x[1] for x in ids}) == 16
        if identities is None: identities = ids
        assert ids == identities
        for record in records:
            assert record['full_hist_exact_previous_posthoc'] is True and len(record['layers']) == 3
            for layer in record['layers']:
                z = layer['pooled_current_plus_four_future']; c0,c1 = z['semantic_softmax_classes']
                geo = z['geometry_GMO_one_minus_p0']; index = layer['layer_index']
                assert c0['GT_mass'] > 0 and c1['GT_mass'] > 0
                assert c0['class_present'] and c1['class_present'] and geo['denominator_epsilon'] == 1e-5
                sem_approx = .5*sum(loss_one(c[k]) for c in (c0,c1) for k in ('precision','recall','specificity'))
                geo_approx = sum(loss_one(geo[k]) for k in ('precision','recall','specificity'))
                ideal = 1.5*loss_one(c1['precision'])+.5*loss_one(c0['precision'])+2*loss_one(c1['recall'])+2*loss_one(c1['specificity'])
                eps_correction = sum(math.log1p(1e-5/x) for x in (c1['prediction_mass'],c1['GT_mass'],c0['GT_mass']))
                actual = record['original_loss_components']
                sem = actual['loss_voxel_sem_scal_inter_'+str(index)]
                geometry = actual['loss_voxel_geo_scal_inter_'+str(index)]
                ratios = [c[k] for c in (c0,c1,geo) for k in ('precision','recall','specificity')]
                rows.append(dict(model=model,sample_token=record['sample_token'],scene_token=record['scene_token'],
                    official_index=record['official_index'],layer=index,GT0=c0['GT_mass'],GT1=c1['GT_mass'],
                    native_sem=sem,native_geo=geometry,serialized_sem_approx=sem_approx,serialized_geo_approx=geo_approx,
                    sem_approx_minus_native=sem_approx-sem,geo_approx_minus_native=geo_approx-geometry,
                    ideal_binary_combination=ideal,epsilon_correction_ideal=eps_correction,
                    ideal_plus_epsilon_minus_native=ideal+eps_correction-sem-geometry,
                    minimum_saved_ratio=min(ratios),any_forward_clamp_regime=any(x<=math.exp(-100) for x in ratios),
                    any_backward_denominator_floor_regime_from_saved_ratios=any(x*(1-x)<1e-12 for x in ratios),
                    semantic_geo_prediction_mass_difference=z['numerical_difference']['semantic_minus_geometric_prediction_mass']))
    assert len(rows) == 192
    result = dict(status='COMPLETE_REAL_RECEIPT_ALGEBRA_REANALYSIS',source_summary_sha256=SUMMARY_SHA,
        source_complete_sha256=sha(a.source/'complete.json'),source_records_sha256=done['records_sha256'],
        script_sha256=sha(__file__),native_loss_source_sha256=LOSS_SHA,config_sha256=CONFIG_SHA,
        models=list(done['records_sha256']),anchors_per_model=16,layers=3,rows=192,
        current_plus_future_pooled=True,all_rows_GT_both_classes_present=True,
        model_forward_calls=0,gradient_calls=0,optimizer_updates=0,mock_data=False,
        source_has_O_or_F=False,exact_CUDA_replay_claim=False,
        maximum_absolute_errors={key:max(abs(r[key]) for r in rows) for key in (
            'sem_approx_minus_native','geo_approx_minus_native','ideal_plus_epsilon_minus_native')},
        forward_clamp_rows=sum(r['any_forward_clamp_regime'] for r in rows),
        backward_floor_rows_from_saved_ratios=sum(r['any_backward_denominator_floor_regime_from_saved_ratios'] for r in rows),
        caveat='Saved soft counts support approximate scalar reconstruction and conditional algebra only. No gradient equivalence, loss conflict, causal harm, O mechanism, or population claim.')
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'numbers.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.out/'complete.json').write_text(json.dumps(dict(status=result['status'],files_sha256={n:sha(a.out/n) for n in ('summary.json','numbers.csv')}),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
