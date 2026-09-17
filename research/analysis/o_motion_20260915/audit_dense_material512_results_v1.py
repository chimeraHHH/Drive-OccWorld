"""Independent CPU aggregation from completed original-domain records.

Does not import the producer's aggregation/metric functions or access models.
Bootstrap describes paired scene sampling on exposed dev200, not seed variance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def ratio(x, y):
    return np.divide(x, y, out=np.full(np.shape(x), np.nan, dtype=float), where=y != 0)


def classification_check(hist, reported):
    hist = np.asarray(hist, dtype=np.int64)
    assert np.array_equal(hist, reported['confusion']) and (hist >= 0).all()
    assert int(hist.sum()) == reported['valid_voxels']
    checks = 0
    for i, name in enumerate(reported['row_names']):
        tp = int(hist[i, i]); gt = int(hist[i].sum()); pred = int(hist[:, i].sum())
        fp = pred-tp; fn = gt-tp; tn = int(hist.sum())-tp-fp-fn
        item = dict(GT=gt, predicted=pred, TP=tp, FP=fp, FN=fn, TN=tn)
        for key, numerator, denominator in [('IoU', tp, tp+fp+fn), ('precision', tp, pred), ('recall', tp, gt)]:
            item[key] = numerator/denominator if denominator else None
        for key, value in item.items():
            got = reported['classes'][name][key]
            assert (got is None and value is None) or (got is not None and value is not None and abs(got-value) < 1e-12), (name, key, got, value)
            checks += 1
    return checks


def persistence_histograms(row, arm):
    """Exact future-GT versus current-prediction marginal of joint transitions."""
    outputs = []
    data = row['metrics_by_arm'][arm]
    for h, transition in enumerate(data['transitions'], 1):
        assert transition['row_names'] == transition['column_names'] == ['00','01','10','11']
        domain = transition['domain']
        assert domain['both_valid'] == domain['h_valid'] == domain['t0_valid'] == domain['grid_voxels']
        joint = np.asarray(transition['confusion'], dtype=np.int64)
        now = np.zeros((2,2), dtype=np.int64)
        future = np.zeros((2,2), dtype=np.int64)
        persistence = np.zeros((2,2), dtype=np.int64)
        for gt in range(4):
            for prediction in range(4):
                now[gt//2, prediction//2] += joint[gt,prediction]
                future[gt%2, prediction%2] += joint[gt,prediction]
                persistence[gt%2, prediction//2] += joint[gt,prediction]
        assert np.array_equal(now, data['horizons'][0]['occupancy']['confusion'])
        assert np.array_equal(future, data['horizons'][h]['occupancy']['confusion'])
        outputs.append(persistence)
    return np.asarray(outputs)


def run(a):
    n = Path(__file__).parent
    fit = n/'server_results/training/dense_material512_v1'/f'{a.arm}_train'
    evaluation = fit.parent/f'{a.arm}_evaluate'
    protocol = read(n/'dense_material512_evaluation_protocol_v1.json')
    done = read(evaluation/'complete.json'); train_done = read(fit/'complete.json')
    assert done['status'] == 'COMPLETE_MATERIAL512_DEV200' and done['arm'] == a.arm
    assert train_done['status'] == 'COMPLETE_DENSE_MATERIAL512_TRAIN' and train_done['arm'] == a.arm
    verified = {}
    for root, receipt in [(fit, train_done), (evaluation, done)]:
        for name, digest in receipt['files_sha256'].items():
            if name.endswith(('.json', '.jsonl')):
                assert sha(root/name) == digest, str(root/name)
                verified[root.name+'/'+name] = digest
    assert done['protocol_sha256'] == sha(n/'dense_material512_evaluation_protocol_v1.json')
    assert train_done['protocol_sha256'] == protocol['training_protocol_sha256']
    manifest = read(fit/'manifest.json'); eval_manifest = read(evaluation/'manifest.json')
    assert eval_manifest['training_complete_sha256'] == sha(fit/'complete.json')
    assert eval_manifest['parameters_sha256'] == train_done['final_parameters_sha256']
    assert manifest['initial_parameters_sha256'] == protocol['codec_final_parameters_sha256']
    assert manifest['initial_frozen_sha256'] == manifest['frozen_final_sha256'] == protocol['codec_encoder_decoder_sha256']
    assert manifest['development_samples_read'] == 0 and manifest['completed_updates'] == 2048
    rng = np.random.RandomState(11)
    orders = [rng.permutation(512).tolist() for _ in range(4)]
    assert manifest['sample_orders'] == orders
    assert manifest['sample_orders_sha256'] == hashlib.sha256(json.dumps(orders, separators=(',', ':')).encode()).hexdigest()
    trainrows = lines(fit/'training.jsonl'); orderrows = lines(fit/'sample_order.jsonl')
    assert len(trainrows) == len(orderrows) == 2048
    populations = lines(fit/'samples.jsonl')
    assert len(populations) == len({p['ordinal'] for p in populations}) == 512
    assert len({p['scene_token'] for p in populations}) == 256
    by_ord = {p['ordinal']: p for p in populations}
    for i, (row, order) in enumerate(zip(trainrows, orderrows)):
        epoch, position = divmod(i, 512)
        assert row['update'] == i+1 and row['epoch'] == epoch and row['position'] == position
        assert row['ordinal'] == orders[epoch][position]
        assert all(row[k] == v for k, v in order.items())
        assert row['sample_token'] == by_ord[row['ordinal']]['sample_token']
        assert row['scene_token'] == by_ord[row['ordinal']]['scene_token']
        assert all(np.isfinite(row[k]) for k in ('total', 'occupancy', 'physical', 'ce', 'lovasz', 'preclip_grad_norm', 'elapsed_seconds'))
        assert abs(row['occupancy']-row['ce']-row['lovasz']) < 2e-6
        assert abs(row['total']-row['occupancy']-.1*row['physical']) < 2e-6
    refpath = n/'server_results/training/common_change_dev200_v2/records.jsonl'
    assert sha(refpath) == protocol['o_records_sha256']
    reference = {r['sample_token']: r for r in lines(refpath)}
    rows = lines(evaluation/'records.jsonl'); summary = read(evaluation/'summary.json')
    assert len(rows) == len({r['sample_token'] for r in rows}) == len(reference) == 200
    assert len({r['scene_token'] for r in rows}) == 100
    for i, row in enumerate(rows):
        assert row['ordinal'] == i
        old = reference[row['sample_token']]
        for key in ('scene_token', 'split', 'official_index', 'input_sha256', 'GT_cache_sha256', 'raw_label_sha256'):
            assert row[key] == old[key], key
        assert row['metrics_by_arm']['O'] == old['metrics_by_arm']['O']
        for h in range(5):
            own = row['metrics_by_arm'][a.arm]['horizons'][h]
            base = old['metrics_by_arm']['O']['horizons'][h]
            assert np.array_equal(np.array(own['occupancy']['confusion']).sum(1), np.array(base['occupancy']['confusion']).sum(1))
            assert own['actual_dt_seconds'] == base['actual_dt_seconds']
            if h:
                for group, values in own['motion_positive_attribution']['groups'].items():
                    assert values['GT'] == base['motion_positive_attribution']['groups'][group]['GT']
                    assert values['TP']+values['FN'] == values['GT']
        for own, base in zip(row['metrics_by_arm'][a.arm]['transitions'], old['metrics_by_arm']['O']['transitions']):
            assert own['domain'] == base['domain']
    scenes = sorted({r['scene_token'] for r in rows})
    scene_groups = [[r for r in rows if r['scene_token'] == scene] for scene in scenes]
    weights = np.random.RandomState(11).multinomial(100, np.full(100, .01), size=10000)
    stats = {}; samples = {}; checked = 0
    for arm in (a.arm, 'O'):
        per_scene = np.array([np.sum([[h['occupancy']['confusion'] for h in r['metrics_by_arm'][arm]['horizons']] for r in group], axis=0) for group in scene_groups], dtype=np.int64)
        hist = per_scene.sum(0); boot = np.einsum('bs,shij->bhij', weights, per_scene)
        future = 100*ratio(hist[1:, 1, 1], hist[1:, 1, 1]+hist[1:, 0, 1]+hist[1:, 1, 0]).mean()
        boot_future = 100*ratio(boot[:, 1:, 1, 1], boot[:, 1:, 1, 1]+boot[:, 1:, 0, 1]+boot[:, 1:, 1, 0]).mean(1)
        assert abs(future-summary['common'][arm]['future_macro_GMO_percent']) < 1e-12
        for h in range(5):
            checked += classification_check(hist[h], summary['common'][arm]['horizons'][h]['occupancy'])
        movement = {}; movement_pooled = {}
        for label, groups in [('moving', ('speed_gt_0.5_le_5', 'speed_gt_5')), ('stationary', ('speed_le_0.1',))]:
            counts = np.array([[[sum(r['metrics_by_arm'][arm]['horizons'][h]['motion_positive_attribution']['groups'][g][key] for r in sg for g in groups) for key in ('TP', 'GT')] for h in range(1,5)] for sg in scene_groups])
            pooled = counts.sum(0); boot_counts = np.einsum('bs,shk->bhk', weights, counts)
            movement[label] = 100*float(ratio(pooled[:,0], pooled[:,1]).mean())
            movement_pooled[label] = 100*float(ratio(pooled[:,0].sum(),pooled[:,1].sum()))
            samples[arm+'_'+label] = 100*ratio(boot_counts[:, :, 0], boot_counts[:, :, 1]).mean(1)
        for h in range(1,5):
            for group, target in summary['common'][arm]['horizons'][h]['motion_positive_groups'].items():
                for key in ('GT','TP','FN'):
                    total=sum(r['metrics_by_arm'][arm]['horizons'][h]['motion_positive_attribution']['groups'][group][key] for r in rows)
                    assert total==target[key]
                    checked+=1
                expected=target['TP']/target['GT'] if target['GT'] else None
                assert expected==target['recall']
                checked+=1
        transition = {}
        for h in range(4):
            thist = np.sum([r['metrics_by_arm'][arm]['transitions'][h]['confusion'] for r in rows],axis=0,dtype=np.int64)
            checked += classification_check(thist, summary['common'][arm]['transitions'][h])
        for name in ('01', '10'):
            transition[name] = 100*np.mean([h['classes'][name]['IoU'] for h in summary['common'][arm]['transitions']])
        stats[arm] = dict(current_GMO_percent=100*float(ratio(hist[0,1,1],hist[0,1,1]+hist[0,0,1]+hist[0,1,0])), future_GMO_percent=float(future),
            horizon_GMO_percent=(100*ratio(hist[:,1,1],hist[:,1,1]+hist[:,0,1]+hist[:,1,0])).tolist(),
            future_FP=int(hist[1:,0,1].sum()), future_FN=int(hist[1:,1,0].sum()),
            future_macro_moving_recall_percent=movement['moving'],future_macro_stationary_recall_percent=movement['stationary'],
            future_pooled_moving_recall_percent=movement_pooled['moving'],future_pooled_stationary_recall_percent=movement_pooled['stationary'],
            arrival_IoU_percent=float(transition['01']),vacating_IoU_percent=float(transition['10']))
        samples[arm+'_GMO'] = boot_future
        persistence_scenes=np.array([np.sum([persistence_histograms(r,arm) for r in group],axis=0) for group in scene_groups])
        ph=persistence_scenes.sum(0); pb=np.einsum('bs,shij->bhij',weights,persistence_scenes)
        piou=100*ratio(ph[:,1,1],ph[:,1,1]+ph[:,0,1]+ph[:,1,0])
        biou=100*ratio(pb[:,:,1,1],pb[:,:,1,1]+pb[:,:,0,1]+pb[:,:,1,0]).mean(1)
        stats[arm]['own_current_persistence']=dict(
            future_GMO_percent=float(piou.mean()),horizon_GMO_percent=piou.tolist(),
            future_FP=int(ph[:,0,1].sum()),future_FN=int(ph[:,1,0].sum()),
            forecast_minus_persistence_GMO_pp=float(future-piou.mean()),
            forecast_minus_persistence_scene95CI_pp=np.quantile(boot_future-biou,[.025,.975]).tolist(),
            derivation='Exact marginal of saved GT/prediction joint t0-future transitions; all voxels valid at both endpoints, both native marginals rechecked',
            zero_displacement_interpretation='T fixed-material zero transport gives current persistence; this identity does not apply to evolving J/D content')
    physical = []
    for h in range(4):
        item = dict(horizon_seconds=(h+1)*.5, groups={})
        for group in ('stationary','ambiguous','moving'):
            values=[r['physical'][h]['groups'][group] for r in rows]
            count=sum(v['objects'] for v in values); pointcount=sum(v['points'] for v in values)
            target=summary['physical'][h]['groups'][group]
            assert target['objects']==count and target['points']==pointcount
            for key in ('epe','magnitude','zero_epe'):
                total=sum(v[key+'_object_sum'] for v in values)
                assert abs(target[key+'_object_sum']-total)<1e-10
                assert abs(target[key]-total/count)<1e-12
                checked+=2
            item['groups'][group]={k:target[k] for k in ('objects','points','epe','zero_epe')}
        physical.append(item)
    differences={key:dict(delta_pp=stats[a.arm][metric]-stats['O'][metric],
        scene_bootstrap_95CI_pp=np.quantile(samples[a.arm+'_'+key]-samples['O_'+key],[.025,.975]).tolist())
        for key,metric in [('GMO','future_GMO_percent'),('moving','future_macro_moving_recall_percent'),('stationary','future_macro_stationary_recall_percent')]}
    out=dict(status='PASS_COMPLETED_RECORD_AGGREGATION_AUDIT',arm=a.arm,actual_updates=2048,train_samples=512,development_samples=200,development_scenes=100,
        evaluation_complete_sha256=sha(evaluation/'complete.json'),training_complete_sha256=sha(fit/'complete.json'),
        source_sha256=sha(__file__),verified_text_files_sha256=verified,
        original_O_metrics_all_exact=True,original_GT_domains_exact=True,update_orders_exact=True,
        classification_and_physical_scalars_checked=checked,metrics=stats,difference_vs_O=differences,physical=physical,
        bootstrap=dict(replicates=10000,seed=11,unit='scene',sampling='multinomial scene counts; pooled counts before ratios',scope='Exposed dev200 sampling variation; not training-seed uncertainty'),
        limitations=['No independent rerun of raw predictions or checkpoint loading in this CPU audit','Physical EPE is XYZ rigid-box proxy at original source points, not O physical EPE','Not full5119 or independent confirmation','Only T/J/D have matched training; original O budget/input privileges differ'])
    target=n/f'dense_material512_{a.arm}_evidence_v1.json'
    target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=out['status'],arm=a.arm,metrics=stats,difference_vs_O=differences,scalars_checked=checked),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['T','J','D'],required=True)
    run(parser.parse_args())
