"""Authenticate completed curves, independently aggregate logs, export figure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    root=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    done=json.loads((root/'complete.json').read_text())
    assert done['status']=='COMPLETE_TRAIN4_MATERIAL_LEARNING_CURVES' and done['updates']==1024
    assert done['development_read']==0
    for name in ['protocol.json','manifest.json','summary.json','training.jsonl','milestones.jsonl','objects.jsonl']:
        assert sha(root/name)==done['files_sha256'][name],name
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    objects={(r['arm'],r['step']):r['objects'] for r in map(json.loads,(root/'objects.jsonl').read_text().splitlines())}
    milestones={(r['arm'],r['step']):r for r in map(json.loads,(root/'milestones.jsonl').read_text().splitlines())}
    training=[json.loads(x) for x in (root/'training.jsonl').read_text().splitlines()]
    assert len(training)==1024
    assert len({v['initial_parameters_sha256'] for v in manifest['arms'].values()})==1
    rows=[];max_error=0.;audited=0
    reference=None; t0=None
    for arm in ['T','J']:
        steps=[r['step'] for r in training if r['arm']==arm];assert steps==list(range(1,513))
        arm_train=[r for r in training if r['arm']==arm]
        assert [r['sample_index'] for r in arm_train]==[i%4 for i in range(512)]
        for point in summary['arms'][arm]:
            step=point['step'];assert step in [0,32,128,512] and point==milestones[(arm,step)]
            for condition,value in point['occupancy'].items():
                hist=sum(np.asarray(r['hist'][condition],dtype=np.int64) for r in point['sample_hists'])
                assert np.array_equal(hist,value['hist'])
                if reference is None:reference=hist.sum(2);t0=hist[0]
                assert np.array_equal(hist.sum(2),reference) and np.array_equal(hist[0],t0)
                iou=hist[:,1,1]/(hist[:,1,1]+hist[:,0,1]+hist[:,1,0])
                err=max(float(np.max(np.abs(iou-value['iou_by_horizon']))),abs(float(iou[1:].mean())-value['future_mean_iou']))
                assert err<1e-14;max_error=max(max_error,err);audited+=6
            for h,item in enumerate(point['physical']):
                for g,name in enumerate(['stationary','ambiguous','moving']):
                    raw=[r for r in objects[(arm,step)] if r['h']==h and r['group']==g]
                    assert item[name]['objects']==len(raw) and item[name]['points']==sum(r['points'] for r in raw)
                    for key in ['epe','magnitude','zero_epe']:
                        value=float(np.mean([r[key] for r in raw])) if raw else None
                        if value is None:assert item[name][key] is None
                        else:
                            err=abs(value-item[name][key]);assert err<1e-12;max_error=max(max_error,err);audited+=1
            moving=point['physical'][-1]['moving'];static=point['physical'][-1]['stationary']
            grads=point['gradients'];occnorm=[g['all_dynamics']['occupancy_norm'] for g in grads]
            phynorm=[g['all_dynamics']['weighted_physical_norm'] for g in grads]
            cos=[g['all_dynamics']['cosine'] for g in grads if g['all_dynamics']['cosine'] is not None]
            velocity_cos=[g['velocity_only']['cosine'] for g in grads if g['velocity_only']['cosine'] is not None]
            rows.append(dict(arm=arm,update=step,train_future_iou_percent=100*point['occupancy']['normal']['future_mean_iou'],
                zero_address_iou_percent=100*point['occupancy']['zero']['future_mean_iou'],
                reverse_address_iou_percent=100*point['occupancy']['reverse']['future_mean_iou'],
                moving2s_epe_m=moving['epe'],moving2s_predicted_magnitude_m=moving['magnitude'],
                zero_moving2s_epe_m=moving['zero_epe'],stationary2s_epe_m=static['epe'],zero_stationary2s_epe_m=static['zero_epe'],
                occupancy_loss_mean=float(np.mean([x['occupancy'] for x in point['losses']])),
                physical_loss_mean=float(np.mean([x['physical'] for x in point['losses']])),
                occupancy_gradient_norm_mean=float(np.mean(occnorm)),weighted_physical_gradient_norm_mean=float(np.mean(phynorm)),
                dynamic_gradient_cosine_mean=float(np.mean(cos)) if cos else None,
                velocity_gradient_cosine_mean=float(np.mean(velocity_cos)) if velocity_cos else None))
    path=out/'learning_curves.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    update_stats={}
    for arm in ['T','J']:
        values=[r for r in training if r['arm']==arm]
        update_stats[arm]={}
        for prefix in ['dynamics.','content_increment.','velocity.']:
            first=next((r['step'] for r in values if r['gradient_stats'][prefix]['norm'] is not None and r['gradient_stats'][prefix]['norm']>0),None)
            update_stats[arm][prefix]=dict(first_nonzero_gradient_step=first,
                update_norm_mean=float(np.mean([r['update_norm'][prefix] for r in values])),
                update_norm_sum=float(sum(r['update_norm'][prefix] for r in values)),
                includes_adamw_weight_decay=True)
    report=dict(status='VERIFIED_TRAIN4_MATERIAL_CURVE_AGGREGATES',source_complete_sha256=sha(root/'complete.json'),
        aggregated_scalars=audited,max_abs_error=max_error,same_gt_denominators=True,frozen_t0_hist=True,
        same_initial_parameters=True,update_order_verified=True,checkpoint_selection=False,
        scope='same four training samples only; no O or held-out comparison',rows=rows,update_statistics=update_stats,
        seconds=summary['seconds'],peak_allocated_gib=summary['peak_allocated_gib'],
        parameter_counts=manifest['arms'],source_sha256=sha(__file__))
    (out/'aggregate_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.7),layout='constrained')
    colors={'T':'#3288bd','J':'#d95f02'}
    labels={'T':'Fixed current material','J':'Evolving content'}
    keys=['moving2s_epe_m','stationary2s_epe_m','train_future_iou_percent']
    titles=['Moving objects: 2 s EPE (m)','Stationary objects: 2 s EPE (m)','Future occupancy IoU (%)']
    for ax,key,title in zip(axes,keys,titles):
        for arm in ['T','J']:
            data=[r for r in rows if r['arm']==arm]
            ax.plot([r['update'] for r in data],[r[key] for r in data],marker='o',lw=2,color=colors[arm],label=labels[arm])
        ax.set_xticks([0,32,128,512]);ax.set_xlabel('Optimizer updates');ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.18)
    axes[0].axhline(rows[0]['zero_moving2s_epe_m'],color='#666666',ls='--',lw=1,label='Zero-motion reference')
    axes[1].axhline(rows[0]['zero_stationary2s_epe_m'],color='#666666',ls='--',lw=1)
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Material-readout test • 4 train samples • frozen codec • seed 11',fontsize=12,fontweight='bold')
    fig.savefig(out/'learning_curves.png',dpi=200);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    print(json.dumps({'audited_scalars':audited,'max_abs_error':max_error,'final':[r for r in rows if r['update']==512]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    main(parser.parse_args())
