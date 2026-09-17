"""Independent integer-count audit of final T train512 fit; no model access."""
import hashlib,json
from pathlib import Path
import numpy as np
from audit_dense_material512_results_v1 import classification_check


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def iou(h):
    h=np.asarray(h,dtype=np.int64)
    d=int(h[1].sum()+h[:,1].sum()-h[1,1]);return float(h[1,1]/d) if d else None


def run():
    n=Path(__file__).resolve().parent
    directory=n/'server_results/diagnostics/material512_train_fit_full_v1'
    done=read(directory/'complete.json')
    assert done['status']=='COMPLETE_MATERIAL512_TRAIN_FIT' and done['mode']=='full'
    assert done['arm']=='T' and done['anchors']==512 and done['scenes']==256 and done['optimizer_updates']==0
    verified={}
    for name,digest in done['files_sha256'].items():
        assert Path(name).name==name and sha(directory/name)==digest,name
        verified[name]=digest
    assert done['protocol_sha256']==sha(n/'material512_train_fit_protocol_v1.json')==sha(directory/'protocol.json')
    manifest=read(directory/'manifest.json');summary=read(directory/'summary.json');rows=lines(directory/'records.jsonl')
    train=n/'server_results/training/dense_material512_v1/T_train';train_done=read(train/'complete.json')
    assert manifest['training_complete_sha256']==sha(train/'complete.json')
    assert manifest['parameters_sha256']==train_done['final_parameters_sha256']
    assert manifest['checkpoint_sha256']==train_done['files_sha256']['model_final.pth']
    assert manifest['optimizer_updates']==0 and manifest['mode']=='full' and manifest['split']=='train'
    assert sha(train/'samples.jsonl')==train_done['files_sha256']['samples.jsonl']
    population={r['sample_token']:r for r in lines(train/'samples.jsonl')}
    assert len(rows)==len({r['sample_token'] for r in rows})==len(population)==512
    assert len({r['scene_token'] for r in rows})==256
    assert summary['anchors']==512 and summary['scenes']==256 and summary['optimizer_updates']==0
    assert summary['final_parameters_unchanged'] and summary['mode']=='full' and summary['split']=='train'
    full_domain=True;persistence=[];joint_forecast=[];checks=0
    for ordinal,row in enumerate(rows):
        assert row['ordinal']==ordinal and row['split']=='train'
        assert set(row['metrics_by_arm'])=={'T'} and row['predictions_completed_before_target_read']
        original=population[row['sample_token']]
        for k,key in [('scene_token','scene_token'),('input_sha256','inputs_sha256'),('GT_cache_sha256','targets_sha256')]:
            assert row[k]==original[key],key
        assert all(row[k]==v for k,v in manifest['anchors'][ordinal].items())
        ph=[];fh=[];m=row['metrics_by_arm']['T']
        for t,h in enumerate(m['horizons']):
            checks+=classification_check(h['occupancy']['confusion'],h['occupancy'])
            if t:
                for v in h['motion_positive_attribution']['groups'].values():assert v['GT']==v['TP']+v['FN']
        for h,t in enumerate(m['transitions'],1):
            checks+=classification_check(t['confusion'],t)
            assert t['row_names']==t['column_names']==['00','01','10','11']
            d=t['domain'];j=np.asarray(t['confusion'],dtype=np.int64)
            assert j.shape==(4,4) and int(j.sum())==d['both_valid']
            pm=np.zeros((2,2),np.int64);fm=pm.copy();cm=pm.copy()
            for gt in range(4):
                for pred in range(4):
                    pm[gt%2,pred//2]+=j[gt,pred]
                    fm[gt%2,pred%2]+=j[gt,pred]
                    cm[gt//2,pred//2]+=j[gt,pred]
            if d['both_valid']==d['h_valid']:
                assert np.array_equal(fm,m['horizons'][h]['occupancy']['confusion'])
            else:full_domain=False
            if d['both_valid']==d['t0_valid']:
                assert np.array_equal(cm,m['horizons'][0]['occupancy']['confusion'])
            assert np.array_equal(pm.sum(1),fm.sum(1))
            ph.append(pm);fh.append(fm)
        persistence.append(ph);joint_forecast.append(fh)
    common=summary['common']['T'];ious=[]
    for h in range(5):
        hist=np.sum([r['metrics_by_arm']['T']['horizons'][h]['occupancy']['confusion'] for r in rows],axis=0,dtype=np.int64)
        checks+=classification_check(hist,common['horizons'][h]['occupancy'])
        ious.append(100*iou(hist))
        if h:
            for name,v in common['horizons'][h]['motion_positive_groups'].items():
                for key in ('GT','TP','FN'):
                    assert v[key]==sum(r['metrics_by_arm']['T']['horizons'][h]['motion_positive_attribution']['groups'][name][key] for r in rows)
    assert abs(np.mean(ious[1:])-common['future_macro_GMO_percent'])<1e-10
    for h in range(4):
        joint=np.sum([r['metrics_by_arm']['T']['transitions'][h]['confusion'] for r in rows],axis=0,dtype=np.int64)
        checks+=classification_check(joint,common['transitions'][h])
        for group,reported in summary['physical'][h]['groups'].items():
            pieces=[r['physical'][h]['groups'][group] for r in rows]
            assert reported['objects']==sum(x['objects'] for x in pieces)
            assert reported['points']==sum(x['points'] for x in pieces)
            for k in ('epe','magnitude','zero_epe'):
                value=sum(x[k+'_object_sum'] for x in pieces)
                assert abs(value-reported[k+'_object_sum'])<1e-9
                assert abs(value/reported['objects']-reported[k])<1e-10
    ph=np.sum(persistence,axis=0,dtype=np.int64);fh=np.sum(joint_forecast,axis=0,dtype=np.int64)
    piou=[100*iou(h) for h in ph];fiou=[100*iou(h) for h in fh]
    result=dict(schema='independent-material512-train-fit-audit-v1',status='COMPLETE',anchors=512,scenes=256,
        verified_files=verified,complete_sha256=sha(directory/'complete.json'),
        checkpoint_sha256=manifest['checkpoint_sha256'],parameter_sha256=manifest['parameters_sha256'],
        recomputed_classification_scalars=checks,current_GMO_percent=ious[0],future_GMO_percent=ious[1:],
        future_macro_GMO_percent=float(np.mean(ious[1:])),
        persistence=dict(matches_full_future_valid_domain=full_domain,confusion=ph.tolist(),
            future_GMO_percent=piou,future_macro_GMO_percent=float(np.mean(piou)),
            forecast_same_domain_GMO_percent=fiou,
            forecast_minus_persistence_pp=float(np.mean(fiou)-np.mean(piou))),
        physical=summary['physical'],optimizer_updates=0,
        caveat='Training-set fit of one fixed T checkpoint; no O training comparison, no generalization or convergence claim. Raw predictions not independently rerun.')
    dest=n/'material512_train_fit_evidence_v1.json';dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','anchors','current_GMO_percent','future_macro_GMO_percent','persistence')}))


if __name__=='__main__':run()
