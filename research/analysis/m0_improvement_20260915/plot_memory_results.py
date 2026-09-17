"""Render completed real memory results only; never synthesize missing data.

Inputs: aggregate_memory.py summary+complete, frozen protocol, and three full
training logs with matching run manifests/completion receipts. Missing inputs
fail before output-directory creation. Curves use logged loss_mean directly;
each logged point already averages four accumulated microbatches. No further
smoothing, interpolation of missing updates, checkpoint/threshold selection,
or source-result changes are performed.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

MODELS=('M0','M0_fp32','native1','persistent2','rolling2')
ARMS=('native1','persistent2','rolling2')
COLORS={'M0':'#303943','M0_fp32':'#8A949E','native1':'#3975AE',
        'persistent2':'#137F77','rolling2':'#CB7935'}
LABELS={'M0':'M0 · native precision','M0_fp32':'M0_fp32 · precision reference',
        'native1':'native1 · continued','persistent2':'persistent2 · observed memory',
        'rolling2':'rolling2 · capacity control'}
MAIN_COMPARISONS=(('persistent2','rolling2'),('persistent2','native1'),('persistent2','M0'))
FOREST_COMPARISONS=MAIN_COMPARISONS+(('persistent2','M0_fp32'),)


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.is_file(),'Required real input is absent; no figure generated: '+str(path))
    return json.loads(path.read_text())


def close(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)


def validate_gmo_model(model):
    """Recompute foreground IoU from integer confusion; never use binary mIoU."""
    horizons=model['horizons']
    require([h['horizon_seconds'] for h in horizons]==[0.,.5,1.,1.5,2.],
            'Unexpected model horizon contract')
    values=[]
    for row in horizons:
        cm=row['confusion_GT_rows_prediction_columns']
        require(isinstance(cm,list) and len(cm)==2 and all(len(r)==2 for r in cm),
                'Expected binary confusion matrix')
        require(all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for r in cm for x in r),
                'Confusion must contain nonnegative integer counts')
        tp,fp,fn=cm[1][1],cm[0][1],cm[1][0]
        require(tp+fp+fn>0,'Undefined foreground IoU')
        gmo=tp/(tp+fp+fn)
        require(close(gmo,row['gmo_iou']),'Stored GMO differs from foreground confusion; refusing plot')
        values.append(gmo)
    require(close(sum(values[1:])/4,model['metrics']['future_macro_gmo']['estimate']),
            'Future macro is not the mean of four independently pooled GMO IoUs')
    return [100*v for v in values[1:]]


def load_inputs(summary_path,runs_root,protocol_path):
    summary_path=Path(summary_path)
    summary=read(summary_path);complete=read(summary_path.with_name('complete.json'));protocol=read(protocol_path)
    require(complete['status']=='COMPLETE' and complete['summary_sha256']==sha(summary_path),
            'Summary completion/hash binding failed')
    require(summary['schema']=='m0-memory-aggregate-v2' and
            summary['status']=='COMPLETED_DEVELOPMENT_COMPARISON','Expected completed real v2 memory summary')
    require(summary['protocol_sha256']==sha(protocol_path),'Plot protocol differs from evaluated protocol')
    require(set(summary['models'])==set(MODELS),'Expected both frozen M0 references and all three trained arms')
    require(summary['samples']==200 and summary['scenes']==100,
            'This discussion figure is registered for development200/100scenes')
    require(protocol['training']['development_samples']==200 and protocol['training']['seed']==11,
            'Development count/seed mismatch')
    require(summary['semantics']['full_validation_result'] is False and
            summary['semantics']['training_seeds']==1,'Development/single-seed interpretation changed')
    require(summary['bootstrap']['replicates']==10000 and summary['bootstrap']['seed']==11 and
            summary['bootstrap']['paired'] is True and summary['bootstrap']['unit']=='scene',
            'Expected registered paired scene uncertainty')
    require(close(summary['registered_minimum_meaningful_effect_pp'],.5) and
            close(protocol['evaluation']['minimum_meaningful_effect_pp'],.5),
            'Registered meaningful-effect reference is not 0.5 pp')
    require(protocol['evaluation']['primary_mechanism_contrast']=='persistent2 minus rolling2' and
            protocol['evaluation']['utility_contrasts']==[
                'persistent2 minus native1 continuation','persistent2 minus original M0'],
            'Registered key comparisons changed')
    curves={name:validate_gmo_model(summary['models'][name]) for name in MODELS}
    contrasts={(r['left'],r['right']):r for r in summary['comparisons']}
    require(len(contrasts)==len(summary['comparisons']),'Duplicate paired comparison')
    forest=[]
    for left,right in FOREST_COMPARISONS:
        row=contrasts[(left,right)]['metrics']['future_macro_gmo']
        delta=summary['models'][left]['metrics']['future_macro_gmo']['estimate']-summary['models'][right]['metrics']['future_macro_gmo']['estimate']
        require(close(delta,row['delta_ratio']) and close(100*delta,row['delta_pp']),
                'Paired point estimate or ratio-to-pp conversion differs')
        lo,hi=row['ci95_pp'];require(math.isfinite(lo) and math.isfinite(hi) and lo<=hi,'Invalid CI')
        require(all(close(100*r,p) for r,p in zip(row['ci95_ratio'],row['ci95_pp'])),
                'CI ratio/pp conversion differs')
        forest.append(dict(left=left,right=right,delta_pp=row['delta_pp'],lower_pp=lo,upper_pp=hi,
                           registered_key=(left,right) in MAIN_COMPARISONS))
    logs={};sources={}
    hp=protocol['training'];expected_updates=hp['passes']*hp['train_samples']//hp['accumulate']
    for arm in ARMS:
        directory=Path(runs_root)/arm
        manifest=read(directory/'manifest.json');done=read(directory/'complete.json')
        training_done=read(directory/'training_complete.json');path=directory/'training.jsonl'
        require(path.is_file(),'Required real training log absent; no figure generated: '+str(path))
        source=summary['source_receipts'][arm]
        require(sha(directory/'manifest.json')==source['manifest_sha256'] and
                sha(directory/'complete.json')==source['complete_sha256'] and
                sha(directory/'training_complete.json')==source['training_complete_sha256'],
                'Plot run receipts differ from scored run: '+arm)
        require(manifest['protocol_sha256']==summary['protocol_sha256'] and manifest['arm']==arm,
                'Training log run/protocol mismatch')
        require(done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED' and
                done['updates']==expected_updates and training_done['updates']==expected_updates,
                'Training log does not correspond to a complete fixed final run')
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(rows)==expected_updates and [r['update'] for r in rows]==list(range(1,expected_updates+1)),
                'Missing/repeated/out-of-order optimizer updates; no interpolation permitted')
        require([r['examples'] for r in rows]==[hp['accumulate']*i for i in range(1,expected_updates+1)],
                'Training log example counts mismatch')
        require(all(math.isfinite(r['loss_mean']) for r in rows),'Nonfinite logged task loss')
        logs[arm]=rows
        sources[arm]=dict(training_jsonl_sha256=sha(path),manifest_sha256=sha(directory/'manifest.json'),
            complete_sha256=sha(directory/'complete.json'),updates=len(rows),
            loss_field='loss_mean',loss_mean_is_microbatch_average=hp['accumulate'],additional_smoothing=False,
            log_binding='hash captured at plot time; completed run receipts and all update/example identities verified')
    return summary,protocol,curves,forest,logs,sources


def render(summary,protocol,curves,forest,logs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#68737E','axes.linewidth':.7,'xtick.color':'#45505B','ytick.color':'#45505B',
        'text.color':'#25303A','axes.labelcolor':'#25303A','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(15.5,10.0),facecolor='white')
    gs=fig.add_gridspec(2,2,left=.065,right=.965,top=.89,bottom=.205,
                        height_ratios=[1.06,.9],width_ratios=[1.08,1.0],hspace=.67,wspace=.46)
    ax_a=fig.add_subplot(gs[0,0]);ax_b=fig.add_subplot(gs[0,1]);ax_c=fig.add_subplot(gs[1,:])
    fig.text(.065,.959,'Persistent observation memory',fontsize=20,weight='bold',ha='left')
    fig.text(.065,.927,'Fixed-state forecasting  |  Completed development screen',fontsize=11,color='#637180')
    # A: exact real four-horizon foreground IoUs, not binary mIoU.
    styles={'M0':('o',(0,(5,2))),'M0_fp32':('s',(0,(1,2))),
            'native1':('^','-'),'persistent2':('D','-'),'rolling2':('v','-')}
    for name in MODELS:
        marker,linestyle=styles[name]
        ax_a.plot([.5,1.,1.5,2.],curves[name],label=LABELS[name],color=COLORS[name],
                  marker=marker,markersize=5.2,linewidth=2.15 if name=='persistent2' else 1.55,
                  linestyle=linestyle,markerfacecolor='white' if name in ['M0','M0_fp32'] else COLORS[name],
                  zorder=4 if name=='persistent2' else 3)
    ax_a.set_title('(a) Future GMO by horizon',loc='left',weight='bold',pad=14)
    ax_a.set_xlabel('Prediction horizon (s)');ax_a.set_ylabel('GMO IoU (%)')
    ax_a.set_xticks([.5,1.,1.5,2.]);ax_a.set_xlim(.45,2.05)
    all_values=np.asarray(list(curves.values()));span=max(float(np.ptp(all_values)),.8)
    ax_a.set_ylim(max(0,float(all_values.min())-.11*span),min(100,float(all_values.max())+.18*span))
    ax_a.grid(axis='y',color='#E4E9ED',linewidth=.75);ax_a.set_axisbelow(True)
    ax_a.legend(loc='upper center',bbox_to_anchor=(.5,-.235),ncol=2,frameon=False,
                fontsize=8.7,columnspacing=1.3,handlelength=2.5)
    # B: horizontal lines allow a percentile CI to exclude its point estimate;
    # matplotlib errorbar asymmetric widths cannot represent that general case.
    labels=['vs rolling2\nregistered mechanism','vs native1\nregistered continuation',
            'vs M0\nregistered utility','vs M0_fp32\nadditional precision reference']
    ys=np.arange(len(forest))[::-1]
    values=[v for row in forest for v in (row['lower_pp'],row['upper_pp'],row['delta_pp'])]+[0,.5]
    lower,upper=min(values),max(values);span=max(upper-lower,.5)
    ax_b.set_xlim(lower-.09*span,upper+.36*span)
    for y,row,label in zip(ys,forest,labels):
        color=COLORS['persistent2'] if row['registered_key'] else '#687A90'
        ax_b.hlines(y,row['lower_pp'],row['upper_pp'],color=color,linewidth=2.1,zorder=3)
        ax_b.vlines([row['lower_pp'],row['upper_pp']],y-.075,y+.075,color=color,linewidth=1.3)
        ax_b.plot(row['delta_pp'],y,marker='D' if row['registered_key'] else 's',color=color,
                  markersize=6,markerfacecolor=color if row['registered_key'] else 'white',zorder=4)
        ax_b.text(upper+.10*span,y,f"{row['delta_pp']:+.3f}",va='center',fontsize=9.5,
                  family='DejaVu Sans',color=color)
    ax_b.axvline(0,color='#5B6570',linewidth=1.0,zorder=1)
    ax_b.axvline(.5,color='#B77C45',linewidth=1.0,linestyle=(0,(4,3)),zorder=1)
    ax_b.axhline(.5,color='#E0E5EA',linewidth=.8)
    ax_b.set_yticks(ys,labels,fontsize=9);ax_b.set_ylim(-.55,3.55)
    ax_b.set_title('(b) Persistent2 paired effects',loc='left',weight='bold',pad=14)
    ax_b.set_xlabel('Future macro GMO difference (pp)')
    ax_b.grid(axis='x',color='#EEF1F4',linewidth=.65);ax_b.set_axisbelow(True)
    ax_b.text(.0,-.235,'Bars: paired 95% scene CI   |   Dashed: registered +0.5 pp',
              transform=ax_b.transAxes,fontsize=8.8,color='#637180',ha='left')
    # C: no filtering/smoothing/resampling of logged update-level loss_mean.
    for name in ARMS:
        rows=logs[name]
        ax_c.plot([r['update'] for r in rows],[r['loss_mean'] for r in rows],
                  color=COLORS[name],label=name,linewidth=1.1,alpha=.80)
    ax_c.set_title('(c) Original task loss during training',loc='left',weight='bold',pad=14)
    ax_c.set_xlabel('Optimizer update');ax_c.set_ylabel('Composite task loss')
    ax_c.set_xlim(1,len(logs['native1']));ax_c.grid(axis='y',color='#E4E9ED',linewidth=.75)
    ax_c.set_axisbelow(True);ax_c.legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
    ax_c.text(0,-.27,'Each point is the logged mean over 4 accumulated microbatches; no additional smoothing.',
              transform=ax_c.transAxes,fontsize=9,color='#637180')
    fig.text(.065,.115,'Development only: 200 anchors from 100 scenes; fixed training seed 11. This is not the 5,119-anchor full validation result.',
             fontsize=9.2,color='#4F5D69')
    fig.text(.065,.090,'Intervals: 10,000 paired scene bootstrap draws, seed 11; uncertainty of fixed models across scenes, not training-seed robustness.',
             fontsize=9.2,color='#4F5D69')
    fig.text(.065,.065,'M0 retains native TF32 predictions. M0_fp32 and trained heads disable matmul TF32; cuDNN TF32 and the observed t0 cache remain unchanged.',
             fontsize=9.2,color='#4F5D69')
    for extension in ('pdf','svg','png'):
        fig.savefig(out/('memory_results.'+extension),dpi=240 if extension=='png' else None,
                    bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True);p.add_argument('--runs-root',required=True)
    p.add_argument('--protocol',default=str(Path(__file__).with_name('protocol_v2.json')))
    p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    # All input presence/completeness/units/identity checks precede any output.
    summary,protocol,curves,forest,logs,sources=load_inputs(a.summary,a.runs_root,a.protocol)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    render(summary,protocol,curves,forest,logs,out)
    caption=('Completed development comparison on the same 200 anchors from 100 scenes, with one training seed (11). '
        '(a) GMO foreground IoU at 0.5, 1.0, 1.5 and 2.0 seconds; values are percentages, not binary mIoU. '
        '(b) Persistent2 minus rolling2, native1 and original M0 are the registered key comparisons; '
        'M0_fp32 is separately labelled as an additional precision reference. Points and bars show the paired '
        'difference in four-horizon macro GMO and the 95% percentile interval from 10,000 scene-paired draws. '
        'The solid reference is zero and the dashed reference is the registered 0.5 percentage-point minimum meaningful effect. '
        '(c) Original composite task loss against optimizer update, using each logged loss_mean without further smoothing; '
        'each value already averages four accumulated microbatches. M0 retains native TF32 outputs; M0_fp32 and all '
        'trained heads disable matmul TF32 while cuDNN TF32 and the observed t0 cache stay unchanged. These development '
        'data are historically exposed and do not establish full-validation performance or training-seed robustness.\n')
    (out/'caption.txt').write_text(caption)
    receipt=dict(schema='memory-results-plot-v1',status='RENDERED_PENDING_VISUAL_QA',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        summary_sha256=sha(a.summary),protocol_sha256=sha(a.protocol),training_sources=sources,
        plotted_GMO_percent=curves,plotted_forest_pp=forest,
        smoothing='none beyond input log loss_mean over four accumulated microbatches',
        samples=200,scenes=100,training_seed=11,full_validation=False,
        files_sha256={name:sha(out/name) for name in ['memory_results.pdf','memory_results.png','memory_results.svg','caption.txt']})
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],out=str(out))),flush=True)


if __name__=='__main__':
    main()
