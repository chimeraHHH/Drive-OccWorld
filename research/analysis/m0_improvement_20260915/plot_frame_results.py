"""Plot completed real four-model frame results; never synthesize missing data.

Defaults expect local mirrored outputs under server_results/campaign_frame_v1:
summary_v1/{summary,complete}.json and runs/reference/. The latter is needed only
with --with-initialization. All plotted points are derived from integer native
confusion matrices and checked against SHA-bound completed summaries/receipts.
No checkpoint, large cache, remote host, training process or skill is accessed.

Examples:
  python plot_frame_results.py --out results/frame_v1
  python plot_frame_results.py --summary OTHER/summary.json
    --reference-run OTHER/runs/reference --with-initialization --out NEW

No files are produced when required inputs are absent or incomplete. Successful
rendering writes PDF/PNG/SVG, caption and a receipt marked pending visual QA.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

PKG=Path(__file__).resolve().parent
MODELS=('M0','M0_fp32','native1','reference')
HORIZONS=(0.,.5,1.,1.5,2.)
PRIMARY=(('reference','M0_fp32'),('reference','native1'))
PARENT_SHA='071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
COLORS={'M0':'#303943','M0_fp32':'#8A949E','native1':'#3975AE','reference':'#137F77'}
LABELS={'M0':'M0 · native precision','M0_fp32':'M0_fp32 · precision reference',
        'native1':'native1 · continued','reference':'A · reference-frame state, trained'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path,expected=None):
    path=Path(path)
    require(path.is_file(),'Required completed real input is absent; no figure generated: '+str(path))
    data=path.read_bytes()
    if expected is not None:
        require(hashlib.sha256(data).hexdigest()==expected,'Input SHA differs: '+str(path))
    return json.loads(data)


def jsonl(path,expected):
    path=Path(path)
    require(path.is_file(),'Required real initialization/final records are absent: '+str(path))
    require(sha(path)==expected,'Records SHA differs: '+str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(a,b):
    return (isinstance(a,(int,float)) and isinstance(b,(int,float)) and
            math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12))


def confusion(matrix):
    require(isinstance(matrix,list) and len(matrix)==2 and all(isinstance(r,list) and len(r)==2 for r in matrix),
            'Expected 2x2 confusion with GT rows and prediction columns')
    require(all(type(v) is int and v>=0 for row in matrix for v in row),'Confusion counts must be nonnegative integers')
    return matrix


def gmo(matrix):
    matrix=confusion(matrix);denominator=matrix[1][1]+matrix[0][1]+matrix[1][0]
    require(denominator>0,'Undefined GMO foreground IoU')
    return matrix[1][1]/denominator


def validate_model(model,pooled_gt):
    rows=model['horizons']
    require([r['horizon_seconds'] for r in rows]==list(HORIZONS),'Five-horizon contract changed')
    values=[]
    for h,row in enumerate(rows):
        matrix=confusion(row['confusion_GT_rows_prediction_columns'])
        require([sum(r) for r in matrix]==pooled_gt[h],'Model/target GT row totals differ')
        score=gmo(matrix)
        require(close(score,row['gmo_iou']),'GMO differs from foreground confusion; refusing binary mIoU substitution')
        require([matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]]==[row[k] for k in ('TN','FP','FN','TP')],
                'Stored FP/FN/TP/TN differ from confusion')
        values.append(score)
    macro=model['metrics']['future_macro_gmo']
    require(macro['unit']=='ratio' and close(sum(values[1:])/4,macro['estimate']),
            'Future macro must exclude t0 and average the four separately pooled GMO IoUs')
    return [100*x for x in values]


def validate_identities(summary):
    identities=summary['ordered_identities_and_GT']
    require(len(identities)==200 and len({r['sample_token'] for r in identities})==200 and
            len({r['scene_token'] for r in identities})==100,'Expected exact development200/100-scene coverage')
    require(all(r['split']=='development' for r in identities),'Wrong dataset role')
    pooled=[[0,0] for _ in HORIZONS]
    for row in identities:
        counts=row['target_counts_0_1_255_by_frame']
        require(len(counts)==7 and all(len(r)==3 and all(type(v) is int and v>=0 for v in r) for r in counts),
                'Expected seven native GT class-count frames')
        require(all(sum(r)==512*512*40 for r in counts),'GT counts do not exhaust the native grid')
        digest=row['targets_sha256']
        require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
                'Missing target file identity')
        for h in range(5):
            for cls in range(2):pooled[h][cls]+=counts[h+2][cls]
    return identities,pooled


def load_initial(reference_run,summary,identities):
    """A zero-update diagnostic line, never a fifth final model/contrast."""
    directory=Path(reference_run)
    require(not (directory/'failed.json').exists(),'Reference run has a failure receipt')
    bound=summary['source_receipts']['reference']['files_sha256']
    done=read(directory/'complete.json',bound['complete.json'])
    manifest=read(directory/'manifest.json',bound['manifest.json'])
    require(done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED' and done['updates']==512 and done['examples']==2048,
            'Initialization line needs the same completed fixed-final A run')
    require(done['manifest_sha256']==bound['manifest.json'] and
            done['frame_protocol_sha256']==manifest['protocol_sha256']==summary['frame_protocol_sha256'] and
            manifest['arm']=='reference','Initialization/final run identity mismatch')
    initial=manifest['initial_development']
    require(initial['optimizer_updates']==0 and initial['samples']==200 and
            initial['head_state_sha256']==manifest['initial_head_sha256'] and
            initial['sha256']==done['initial_development_sha256']==bound['initial_development_records.jsonl'],
            'Initialization is not the frozen zero-update diagnostic')
    require(done['evaluation']['sha256']==bound['development_records.jsonl'],'Final record hash is not bound')
    totals={label:[[[0,0],[0,0]] for _ in HORIZONS] for label in ('initial','final')}
    for label,filename in [('initial','initial_development_records.jsonl'),('final','development_records.jsonl')]:
        rows=jsonl(directory/filename,bound[filename]);require(len(rows)==200,'Diagnostic/final coverage is incomplete')
        for row,identity in zip(rows,identities):
            require(row['sample_token']==identity['sample_token'] and row['scene_token']==identity['scene_token'] and
                    row['horizon_seconds']==list(HORIZONS),'Diagnostic/final sample order or horizon differs')
            matrices=row['hist_by_horizon'];require(len(matrices)==5,'Diagnostic/final horizon count differs')
            for h,matrix in enumerate(matrices):
                confusion(matrix)
                require([sum(r) for r in matrix]==identity['target_counts_0_1_255_by_frame'][h+2][:2],
                        'Diagnostic/final per-sample GT counts differ')
                for i in range(2):
                    for j in range(2):totals[label][h][i][j]+=matrix[i][j]
    require(totals['final']==[r['confusion_GT_rows_prediction_columns'] for r in summary['models']['reference']['horizons']],
            'Reference final records do not reproduce the completed summary')
    return [100*gmo(matrix) for matrix in totals['initial']],dict(
        status='VERIFIED_ZERO_UPDATE_DIAGNOSTIC',directory=str(directory.resolve()),optimizer_updates=0,
        files_sha256={name:bound[name] for name in ('complete.json','manifest.json','initial_development_records.jsonl','development_records.jsonl')},
        included_in_final_model_count=False,included_in_paired_contrasts=False)


def load_inputs(a):
    path=Path(a.summary);require(not path.with_name('failed.json').exists(),'Aggregation has a failure receipt')
    complete=read(path.with_name('complete.json'))
    require(complete['schema']=='m0-frame-aggregate-complete-v1' and complete['status']=='COMPLETE',
            'Need the completed four-model frame aggregation')
    require(complete['summary_sha256']==complete['files_sha256']['summary.json'],'Summary completion hash fields differ')
    summary=read(path,complete['summary_sha256'])
    for name,digest in complete['files_sha256'].items():
        require(Path(name).name==name,'Unsafe aggregate artifact name')
        require((path.parent/name).is_file() and sha(path.parent/name)==digest,'Incomplete/changed aggregate artifact: '+name)
    require(summary['schema']=='m0-frame-aggregate-v1' and summary['status']=='COMPLETED_DEVELOPMENT_COMPARISON',
            'Unsupported or incomplete frame result schema')
    frame=read(a.protocol,summary['frame_protocol_sha256']);parent=read(a.parent_protocol,PARENT_SHA)
    require(complete['frame_protocol_sha256']==summary['frame_protocol_sha256'] and
            frame['parent_protocol_sha256']==summary['parent_protocol_sha256']==PARENT_SHA,
            'Parent/frame protocol binding differs')
    require(frame['status']=='FROZEN_BEFORE_REFERENCE_TRAINING' and frame['arm']=='reference' and
            frame['training']==parent['training'] and frame['numerical_policy']==parent['numerical_policy'],
            'Reference and control training/precision contracts differ')
    require(summary['script_sha256']==frame['source_sha256']['frame_aggregate.py'] and
            sha(PKG/'frame_aggregate.py')==summary['script_sha256'],'Aggregator source identity differs')
    require(summary['samples']==complete['samples']==200 and summary['scenes']==100 and
            set(summary['models'])==set(MODELS) and complete['models']==list(MODELS),'Four-model development scope differs')
    audit=summary['audit'];semantics=summary['semantics'];bootstrap=summary['bootstrap']
    require(audit['status']=='PASS' and audit['all_four_models_same_GT_row_counts'] is True and
            audit['all_exactly_200_same_order'] is True,'Required producer identity/GT audit did not pass')
    require(semantics['full_validation_result'] is False and semantics['training_seeds']==1 and
            semantics['training_seed_robustness_claim'] is False and semantics['threshold_fitting'] is False and
            semantics['R4_calibration_reused'] is False and semantics['historical_val_exposure'] is True and
            semantics['precision']==frame['numerical_policy'],
            'Development/uncertainty/precision interpretation differs')
    require(frame['training']['seed']==11 and bootstrap['replicates']==10000 and bootstrap['seed']==11 and
            bootstrap['paired'] is True and bootstrap['unit']=='scene' and bootstrap['scene_count']==100 and
            bootstrap['all_anchors_within_scene_kept_together'] is True,
            'Expected 10,000 fixed-seed paired scene draws')
    identities,pooled_gt=validate_identities(summary)
    curves={name:validate_model(summary['models'][name],pooled_gt) for name in MODELS}
    comparisons={(c['left'],c['right']):c for c in summary['comparisons']}
    require(len(comparisons)==len(summary['comparisons']),'Duplicate comparison')
    forest=[]
    for left,right in PRIMARY:
        comparison=comparisons[(left,right)];row=comparison['metrics']['future_macro_gmo']
        delta=summary['models'][left]['metrics']['future_macro_gmo']['estimate']-summary['models'][right]['metrics']['future_macro_gmo']['estimate']
        require(comparison['primary_contrast'] is True and close(delta,row['delta_ratio']) and close(100*delta,row['delta_pp']),
                'Primary effect or ratio-to-percentage-point conversion differs')
        lo,hi=row['ci95_pp'];require(math.isfinite(lo) and math.isfinite(hi) and lo<=hi,'Invalid scene CI')
        require(len(row['ci95_ratio'])==2 and all(close(100*r,p) for r,p in zip(row['ci95_ratio'],row['ci95_pp'])),
                'CI ratio-to-percentage-point conversion differs')
        forest.append(dict(left=left,right=right,delta_pp=row['delta_pp'],lower_pp=lo,upper_pp=hi))
    minimum=summary['registered_minimum_meaningful_effect_pp']
    registered=frame.get('evaluation',{}).get('minimum_meaningful_effect_pp')
    require((minimum is None and registered is None) or (close(minimum,registered) and minimum>=0),
            'Meaningful-effect reference differs from the frame protocol')
    initial,initial_receipt=None,dict(status='OMITTED_NOT_REQUESTED')
    if a.with_initialization:initial,initial_receipt=load_initial(a.reference_run,summary,identities)
    return summary,curves,forest,minimum,initial,initial_receipt


def render(curves,forest,minimum,initial,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#68737E','axes.linewidth':.7,'xtick.color':'#45505B','ytick.color':'#45505B',
        'text.color':'#25303A','axes.labelcolor':'#25303A','pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(13.8,6.8),facecolor='white')
    gs=fig.add_gridspec(1,2,left=.065,right=.96,top=.80,bottom=.38,width_ratios=[1.22,1.],wspace=.43)
    ax=fig.add_subplot(gs[0,0]);effect=fig.add_subplot(gs[0,1])
    fig.text(.065,.948,'Reference-frame state forecasting',fontsize=20,weight='bold')
    fig.text(.065,.905,'Completed development comparison  |  Four fixed final models',fontsize=11,color='#637180')
    styles={'M0':('o',(0,(5,2))),'M0_fp32':('s',(0,(1,2))),
            'native1':('^','-'),'reference':('D','-')}
    if initial is not None:
        ax.plot(HORIZONS,initial,label='A · initialization (0 updates)',color=COLORS['reference'],
            linestyle=(0,(2,3)),marker='.',markersize=4,linewidth=1.15,alpha=.40,zorder=2)
    for name in MODELS:
        marker,style=styles[name]
        ax.plot(HORIZONS,curves[name],label=LABELS[name],color=COLORS[name],linestyle=style,
            marker=marker,markersize=5.0,linewidth=2.2 if name=='reference' else 1.55,
            markerfacecolor='white' if name in MODELS[:2] else COLORS[name],zorder=4 if name=='reference' else 3)
    ax.axvline(.25,color='#D9E0E6',linestyle=(0,(2,3)),linewidth=.8,zorder=1)
    ax.set_title('(a) GMO across evaluated horizons',loc='left',weight='bold',pad=15)
    ax.set_xlabel('Prediction horizon (s)');ax.set_ylabel('GMO foreground IoU (%)')
    ax.set_xticks(HORIZONS,['0\ncurrent','0.5','1.0','1.5','2.0']);ax.set_xlim(-.06,2.06)
    values=np.asarray(list(curves.values())+([] if initial is None else [initial]))
    span=max(float(np.ptp(values)),.8)
    ax.set_ylim(max(0,float(values.min())-.12*span),min(100,float(values.max())+.15*span))
    ax.grid(axis='y',color='#E4E9ED',linewidth=.75);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    # Final models appear first; initialization remains a separately labelled
    # zero-update diagnostic even when its source file is available.
    if initial is not None:handles,labels=handles[1:]+handles[:1],labels[1:]+labels[:1]
    ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.255),ncol=2,
        frameon=False,fontsize=8.8,columnspacing=1.25,handlelength=2.6)
    ys=[1,0];bounds=[v for row in forest for v in (row['lower_pp'],row['upper_pp'],row['delta_pp'])]+[0.]
    if minimum is not None:bounds.append(minimum)
    lower,upper=min(bounds),max(bounds);width=max(upper-lower,.5)
    effect.set_xlim(lower-.12*width,upper+.12*width);effect.set_ylim(-.5,1.58)
    for y,row in zip(ys,forest):
        effect.hlines(y,row['lower_pp'],row['upper_pp'],color=COLORS['reference'],linewidth=2.3,zorder=3)
        effect.vlines([row['lower_pp'],row['upper_pp']],y-.06,y+.06,color=COLORS['reference'],linewidth=1.3)
        effect.plot(row['delta_pp'],y,'D',color=COLORS['reference'],markersize=6.5,zorder=4)
        effect.text(.98,y+.24,f"{row['delta_pp']:+.3f}  [{row['lower_pp']:+.3f}, {row['upper_pp']:+.3f}]",
            transform=effect.get_yaxis_transform(),ha='right',va='center',fontsize=9.2,color=COLORS['reference'])
    effect.axvline(0,color='#596572',linewidth=1,zorder=1)
    if minimum is not None:effect.axvline(minimum,color='#B77C45',linewidth=1,linestyle=(0,(4,3)),zorder=1)
    effect.set_yticks(ys,['A − M0_fp32\nprecision-matched reference','A − native1\nmatched continuation'],fontsize=9.5)
    effect.set_title('(b) Paired effects of trained A',loc='left',weight='bold',pad=15)
    effect.set_xlabel('Future macro GMO difference (pp)')
    effect.grid(axis='x',color='#EEF1F4',linewidth=.65);effect.set_axisbelow(True)
    guide='Bars: 95% paired scene CI; solid line: zero.'
    if minimum is not None:guide+=f'\nDashed line: registered +{minimum:g} pp reference.'
    effect.text(0,-.26,guide,transform=effect.transAxes,fontsize=9,color='#637180',va='top',linespacing=1.5)
    fig.text(.065,.154,'Same 200 development anchors / 100 scenes; one training seed (11). Historically exposed validation data; not full 5,119-anchor validation.',
        fontsize=9,color='#4F5D69')
    fig.text(.065,.116,'Future macro excludes t0: mean of four separately pooled foreground IoUs. CI: 10,000 paired scene draws; no training-seed robustness claim.',
        fontsize=9,color='#4F5D69')
    fig.text(.065,.078,'M0 uses native TF32 predictions. M0_fp32 and trained future heads disable matmul TF32; observed t0 states and cuDNN TF32 are unchanged.',
        fontsize=9,color='#4F5D69')
    for extension in ('pdf','png','svg'):
        fig.savefig(out/('frame_results.'+extension),dpi=240 if extension=='png' else None,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',default=str(PKG/'server_results/campaign_frame_v1/summary_v1/summary.json'))
    parser.add_argument('--reference-run',default=str(PKG/'server_results/campaign_frame_v1/runs/reference'))
    parser.add_argument('--protocol',default=str(PKG/'frame_protocol_v1.json'))
    parser.add_argument('--parent-protocol',default=str(PKG/'protocol_v2.json'))
    parser.add_argument('--with-initialization',action='store_true',help='Require and separately plot SHA-verified zero-update A records')
    parser.add_argument('--out',required=True)
    a=parser.parse_args(argv);out=Path(a.out).resolve();require(not out.exists(),'Output must be a new directory')
    summary,curves,forest,minimum,initial,initial_receipt=load_inputs(a)
    # This is the first output mutation: no mock or partial-input figure exists.
    out.mkdir(parents=True,exist_ok=False);render(curves,forest,minimum,initial,out)
    caption=('Completed development comparison on the same 200 anchors from 100 scenes, with fixed training seed 11. '
        '(a) Foreground GMO IoU at the current time and 0.5, 1.0, 1.5 and 2.0 seconds, shown as percentages. '
        'GMO denotes movable semantic classes; it is not a measured true-motion mask or binary mIoU. '
        'Lines connect evaluated horizons without smoothing. M0 and M0_fp32 are frozen references; native1 and A '
        'use their fixed final checkpoints after 512 optimizer updates (2,048 examples, four passes, accumulation four). '
        '(b) A minus M0_fp32 and A minus native1 are the two primary comparisons. Points show the four-future-horizon '
        'macro GMO differences in percentage points; bars show two-sided 95% percentile intervals from 10,000 paired '
        'scene-bootstrap draws with seed 11. Each horizon pools the native confusion matrices across anchors before '
        'foreground IoU is calculated; the four future IoUs are then averaged. The current-time score is excluded. '
        'All anchors from a resampled scene stay together. Intervals describe fixed-model scene uncertainty, not '
        'training randomness or historical model selection, and are not corrected for multiple comparisons. '
        'The solid vertical reference is zero. '+
        (f'The dashed reference is the registered {minimum:g} percentage-point meaningful-effect threshold. ' if minimum is not None else '')+
        ('The pale dotted A curve is the independently verified zero-update initialization diagnostic; it is excluded '
         'from the four final models and both paired contrasts. ' if initial is not None else '')+
        'M0 retains native TF32 predictions; M0_fp32 and trained future heads disable matmul TF32 while the observed '
        't0 cache and cuDNN TF32 remain unchanged. The real future ego/action conditions are preserved. This is '
        'historically exposed development data, not the full 5,119-anchor validation result or an independent blind test.\n')
    (out/'caption.txt').write_text(caption)
    files=['frame_results.pdf','frame_results.png','frame_results.svg','caption.txt']
    receipt=dict(schema='frame-results-plot-v1',status='RENDERED_PENDING_VISUAL_QA',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        summary_path=str(Path(a.summary).resolve()),summary_sha256=sha(a.summary),
        summary_complete_sha256=sha(Path(a.summary).with_name('complete.json')),
        frame_protocol_sha256=sha(a.protocol),parent_protocol_sha256=sha(a.parent_protocol),
        aggregator_sha256=summary['script_sha256'],plotted_models=list(MODELS),horizon_seconds=list(HORIZONS),
        plotted_GMO_percent=curves,plotted_primary_forest_pp=forest,
        initial_GMO_percent=initial,initialization=initial_receipt,meaningful_effect_reference_pp=minimum,
        metrics_recomputed_from_integer_GT_rows_prediction_columns=True,
        neural_checkpoints_reread=False,raw_large_cache_reread=False,source_results_modified=False,
        additional_smoothing=False,bootstrap_recomputed=False,samples=200,scenes=100,
        training_seed=11,training_seed_count=1,full_validation=False,
        files_sha256={name:sha(out/name) for name in files})
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],out=str(out))),flush=True)


if __name__=='__main__':main()
