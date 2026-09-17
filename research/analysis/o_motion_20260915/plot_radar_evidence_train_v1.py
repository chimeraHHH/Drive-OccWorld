"""Render the completed train512 radar diagnostic, with matched populations.

Two panels: fixed D-speed-bin AUC and three explicitly different coverage
denominators. No CI, significance, fitted gate or new model evaluation.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ANALYSIS_SHA='eee1f40d1c468db32aa434d5dfde6d6c46bf7586d3b9151f9a011ce0b8059e6f'
AUDIT_SHA='629e04ef55ce8cf2f68b0c98e977ff74be7c16576d7f5dc744f44bf94b1eb2d6'
ANALYZER_SHA='75df9873f97ca442e925d260876c124a9115b626edf35af2eb7ab1315f7d80b3'
RULES_SHA='a518b7a0fecaa4e5fb528c20edc69e5f782ae554ff73fcf5ce5e5907dab22d8b'
FEATURES=('radar_relative_zero_margin_mps','lsq_speed_xy_mps')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def close(a,b):require(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),'Numeric source/audit mismatch')


def prepare(analysis_path,audit_path):
    require(sha(analysis_path)==ANALYSIS_SHA and sha(audit_path)==AUDIT_SHA,'Wrong actual analysis or independent audit')
    require(sha(HERE/'analyze_radar_evidence_train_v1.py')==ANALYZER_SHA and
            sha(HERE/'radar_evidence_train_analysis_rules_v1.json')==RULES_SHA,'Analysis source/rules changed')
    x,audit=read(analysis_path),read(audit_path)
    require(x['schema']=='radar-evidence-discrimination-train-analysis-v1' and
            x['status']=='COMPLETE_CPU_RADAR_TRAIN_EVIDENCE_ANALYSIS' and
            (x['samples'],x['scenes'],x['source_points'],x['object_horizon_rows'])==(512,256,1740053,43189),
            'Require complete actual train512 analysis')
    require(x['sources']['analysis_source_sha256']==ANALYZER_SHA and x['sources']['rules_sha256']==RULES_SHA and
            audit['schema']=='root-radar-2s-limited-independent-check-v1' and audit['status']=='PASS' and
            audit['production_sha256']==ANALYSIS_SHA,'Independent audit does not bind this result')
    h=next(r for r in x['horizons'] if r['horizon_seconds']==2.)
    present=h['radar_auc']['radar_present'];bins=present['speed_bins']
    require(len(bins)==7,'Require all seven prespecified speed bins')
    global_auc={f:next(r for r in present['features'] if r['feature']==f) for f in FEATURES}
    for f,row in global_auc.items():
        require(row['direction']==1 and row['positive_points']==6734 and row['negative_points']==9082,
                'Global comparison does not use the same radar-present population')
        close(row['auc'],audit['auc'][f])
    numbers=[];edges=[0.,.1,.5,1.,2.,5.,10.,None]
    for i,b in enumerate(bins):
        require(b['bin_index']==i and b['left_inclusive_mps']==edges[i] and
                b['right_exclusive_mps']==edges[i+1] and b['right_unbounded']==(i==6),'Frozen bin changed')
        rows={f:next(r for r in b['features'] if r['feature']==f) for f in FEATURES}
        for key in ('positive_points','negative_points','positive_weight','negative_weight','zero_benefit_points'):
            require(rows[FEATURES[0]][key]==rows[FEATURES[1]][key],'Unmatched bin population or weighting')
        for f,row in rows.items():
            require(row['direction']==1 and row['auc'] is not None and 0<=row['auc']<=1,'Unexpected score direction/undefined AUC')
            close(row['auc'],audit['bin_auc'][i][f])
        numbers.append(dict(bin_index=i,left_inclusive_mps=edges[i],right_exclusive_mps=edges[i+1],
            margin_auc=rows[FEATURES[0]]['auc'],matched_D_speed_auc=rows[FEATURES[1]]['auc'],
            positive_points=rows[FEATURES[0]]['positive_points'],negative_points=rows[FEATURES[0]]['negative_points'],
            positive_weight=rows[FEATURES[0]]['positive_weight'],negative_weight=rows[FEATURES[0]]['negative_weight'],
            zero_benefit_points=rows[FEATURES[0]]['zero_benefit_points']))
    require([r['negative_points'] for r in numbers[-2:]]==[15,7],'Rare-negative warning no longer matches data')
    strata={s:h['strata'][s]['all'] for s in ('all','radar_present','no_radar')}
    for s,row in strata.items():
        require(row['original_object_count']==10332,'Wrong original object denominator')
        for key in ('points','weight_sum','FULL_denominator_negative_cost_xy_m'):
            close(row[key],audit['checks'][s][key])
    coverage=[]
    definitions=(('Source points','points','number of original valid CRN-uncovered source points'),
                 ('Original-object weight','weight_sum','sum over uncovered points of 1/n_ALL_original_valid_object_points'),
                 ('Negative-benefit cost','FULL_denominator_negative_cost_xy_m',
                  'sum of weighted max(-benefit,0) / original object count; metres per original object'))
    for label,key,definition in definitions:
        all_value=strata['all'][key];yes=strata['radar_present'][key];no=strata['no_radar'][key]
        require(all_value>0 and yes>=0 and no>=0,'Undefined coverage denominator')
        close(yes+no,all_value)
        coverage.append(dict(label=label,field=key,denominator_definition=definition,
            denominator=all_value,radar_present_numerator=yes,no_radar_numerator=no,
            radar_present_percent=100*yes/all_value,no_radar_percent=100*no/all_value))
    close(coverage[0]['radar_present_percent']/100,audit['derived_coverage']['point_fraction'])
    close(coverage[1]['radar_present_percent']/100,audit['derived_coverage']['original_weight_fraction'])
    close(coverage[2]['no_radar_percent']/100,audit['derived_coverage']['damage_without_same_cell_radar_fraction'])
    return numbers,coverage,{f:r['auc'] for f,r in global_auc.items()},x['sources']


def render(numbers,coverage,global_auc,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':11,
        'axes.labelsize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8b95a5',
        'text.color':'#263446','axes.labelcolor':'#263446','xtick.color':'#263446','ytick.color':'#263446'})
    fig=plt.figure(figsize=(12,5.8),facecolor='white')
    gs=fig.add_gridspec(1,2,width_ratios=[1.22,1],left=.063,right=.978,bottom=.295,top=.76,wspace=.48)
    ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1])
    margin_color='#167d9a';speed_color='#526174';warning='#a56716'
    xx=np.arange(7)
    ax.axhline(.5,color='#acb3bf',lw=1,ls=(0,(3,3)),zorder=0)
    ax.axvspan(4.5,6.5,color='#fff0d4',alpha=.65,zorder=-1)
    ax.plot(xx,[r['margin_auc'] for r in numbers],color=margin_color,marker='o',ms=5.5,lw=1.8,label='Radar relative-zero margin')
    ax.plot(xx,[r['matched_D_speed_auc'] for r in numbers],color=speed_color,marker='s',ms=4.8,lw=1.5,ls='--',label='D LSQ speed — same points')
    ax.set_xlim(-.35,6.35);ax.set_ylim(0,1.03);ax.set_yticks(np.arange(0,1.01,.2))
    names=['0–0.1','0.1–0.5','0.5–1','1–2','2–5','5–10','≥10']
    ax.set_xticks(xx,[name+'\n'+f"n−={r['negative_points']:,}" for name,r in zip(names,numbers)])
    for t in ax.get_xticklabels()[-2:]:t.set_color(warning)
    ax.set_ylabel('Weighted AUC');ax.set_xlabel('Prespecified D-speed bins (m/s)',labelpad=9)
    ax.grid(axis='y',color='#e9edf2',lw=.7,zorder=-2)
    ax.set_title('A  |  Comparison within radar-present points',loc='left',pad=37,fontweight='bold')
    ax.text(0,1.055,f"Matched global AUC: margin {global_auc[FEATURES[0]]:.3f}; D speed {global_auc[FEATURES[1]]:.3f}",
        transform=ax.transAxes,fontsize=9,color='#526174')
    ax.legend(loc='lower left',bbox_to_anchor=(.01,.035),frameon=False,fontsize=8.7,handlelength=2.6)
    ax.text(5.52,.12,'Only 15 / 7\nharmful points',ha='center',va='center',color=warning,fontsize=8.5)
    present_color='#248c84';absent_color='#dce3eb'
    y=np.array([2.,1.,0.])
    for yi,row in zip(y,coverage):
        yes,no=row['radar_present_percent'],row['no_radar_percent']
        bx.barh(yi,yes,height=.42,color=present_color,zorder=2)
        bx.barh(yi,no,left=yes,height=.42,color=absent_color,zorder=2)
        bx.text(0,yi+.29,f"Radar present  {yes:.3f}%",fontsize=9,color=present_color,ha='left',va='bottom')
        bx.text(yes+no/2,yi,f"No radar  {no:.3f}%",fontsize=9.2,ha='center',va='center',color='#344357')
    labels=['Source points\n353,431 points','Original-object weight\nΣ point weights = 2,837.01',
            'Negative-benefit cost\n0.030837 m / original object']
    bx.set_yticks(y,labels);bx.set_xlim(0,100);bx.set_ylim(-.5,2.75)
    bx.set_xticks([0,25,50,75,100],[f'{v}%' for v in [0,25,50,75,100]])
    bx.set_xlabel('Share of each separately defined total',labelpad=9)
    bx.tick_params(axis='y',length=0,pad=10,labelsize=8.8)
    bx.grid(axis='x',color='#edf0f4',lw=.7,zorder=0)
    bx.spines['left'].set_visible(False)
    bx.set_title('B  |  Limited same-cell radar coverage',loc='left',pad=37,fontweight='bold')
    bx.text(0,1.055,'Three bars, three different denominators',transform=bx.transAxes,fontsize=9,color='#526174')
    fig.text(.063,.96,'Radar evidence for fixed-D fallback',fontsize=16,fontweight='bold',ha='left')
    fig.text(.063,.915,'Train512 / 256 scenes seen by D  ·  2 s  ·  Source points outside current CRN predicted boxes',fontsize=10.5,color='#526174')
    notes=[
        'A: Both scores use identical radar-present points and original-object weights. n− counts harmful source points; high-speed bins have few negatives.',
        'B: Denominators are point count, sum of 1/n(original object), and total negative-benefit cost. Repeated z points can share one XY radar measurement.',
        'Descriptive training-set diagnostic: no confidence intervals or significance claim; radar was already used by D. No fitted gate or occupancy result.']
    for i,note in enumerate(notes):fig.text(.063,.155-.037*i,note,fontsize=8.6,color='#526174',ha='left')
    fig.savefig(prefix.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(prefix.with_suffix('.pdf'),facecolor='white',metadata={'Title':'Train512 radar evidence — matched AUC and coverage','Author':'Research diagnostic'})
    plt.close(fig)
    return matplotlib.__version__


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=HERE/'radar_evidence_train_analysis_v1.json')
    p.add_argument('--audit',type=Path,default=HERE/'receipts/radar_evidence_train_analysis_v1_root_spot_audit.json')
    p.add_argument('--out-prefix',type=Path,default=HERE/'figures/radar_evidence_train_v1')
    a=p.parse_args();require(a.out_prefix.parent.is_dir(),'Output parent must already exist')
    paths=[a.out_prefix.with_suffix(suffix) for suffix in ('.png','.pdf','.json')]
    require(not any(path.exists() for path in paths),'New figure paths required; never overwrite')
    numbers,coverage,global_auc,sources=prepare(a.analysis,a.audit)
    version=render(numbers,coverage,global_auc,a.out_prefix)
    result=dict(schema='radar-evidence-train-two-panel-figure-v1',status='RENDERED_ACTUAL_RESULTS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs=dict(analysis_path=str(a.analysis.resolve()),analysis_sha256=ANALYSIS_SHA,
            independent_root_audit_sha256=AUDIT_SHA,analysis_source_sha256=ANALYZER_SHA,analysis_rules_sha256=RULES_SHA,
            plot_source_sha256=sha(__file__),upstream_sources=sources),
        scope=dict(train_samples=512,train_scenes=256,D_saw_training_scenes=True,horizon_seconds=2.,
            population='CRN-current-predicted-box-uncovered original source points',
            A='same radar-present population for both fixed-positive-direction scores',
            B='all uncovered points, decomposed by same-cell radar presence with distinct denominators',
            no_CI=True,no_significance_claim=True,no_gate=True,no_new_sensor=True,no_occupancy_result=True),
        panel_A=dict(bins=numbers,matched_global_auc=global_auc,
            margin_formula='norm(vR_xy)-norm(vD_LSQ_xy-vR_xy)',point_weight='1/n_ALL_original_valid_object_points',
            zero_benefit='excluded only from binary AUC',tied_score_credit=.5,highest_bin_negative_counts=[15,7]),
        panel_B=dict(rows=coverage,percentages_computed_from_raw_numerators_and_denominators=True),
        caption_zh='固定D已见过的训练集诊断，2秒，当前CRN预测框未覆盖的原合法源点。A在相同有雷达人口中比较相对零速度的雷达一致性margin与D幅值；匹配全局AUC分别0.755与0.756。最高两速度箱仅15和7个负收益点，未作区间或显著性主张。B的点数、原对象权重和负收益代价采用不同分母；同cell雷达覆盖有限，多数代价位于无雷达处。重复z不代表独立回波；雷达此前已进入D，不是新增独立传感器，也不是门控或占据收益。',
        outputs={path.name:dict(path=str(path.resolve()),sha256=sha(path)) for path in paths[:2]},
        runtime=dict(matplotlib=version,numpy=np.__version__),visual_qa=dict(status='PENDING_MANUAL_IMAGE_INSPECTION'))
    with paths[2].open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(png=str(paths[0]),pdf=str(paths[1]),receipt=str(paths[2]),receipt_sha256=sha(paths[2]))))


if __name__=='__main__':main()
