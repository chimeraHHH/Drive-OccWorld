"""Plot authenticated fixed-D/CV diagnostic; no new scoring or model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE_SHA = 'b91a892b34bf2fea74c226971972e4cee9d53d0b20e11e359beaea6730532b54'
STEM = 'fixed_D_coverage_complementarity_v1'
TITLE = 'Dense motion can fill coverage gaps, with a stationary-motion cost'
CONTRIBUTION = 'FULL_original_object_denominator_contribution_mean_m'
COLORS = dict(covered='#227C8C', uncovered='#E4AD59', benefit='#008C95', cost='#C46835')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def ci_text(row):
    return f"{row['difference_m']:+.3f} [{row['lower95_m']:+.3f}, {row['upper95_m']:+.3f}]"


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=base/'fixed_D_coverage_complementarity_analysis_v1.json')
    parser.add_argument('--out-dir',type=Path,default=base/'figures')
    args=parser.parse_args()
    require(sha(args.source)==SOURCE_SHA,'Completed analysis SHA differs')
    data=json.loads(args.source.read_text())
    require(data['schema']=='fixed-D-coverage-complementarity-analysis-v1' and
            data['status']=='COMPLETE_CPU_RECORD_ANALYSIS' and data['samples']==200 and
            data['scenes']==100 and data['object_horizon_rows']==16074,'Fixed support/status differs')
    require(all(data['verification'][k] for k in ('original_full_support_and_reference_rows_verified',
        'original_D_and_CV_full_means_reproduced','all_partition_counts_means_sums_reconstructed',
        'fixed_blend_uses_only_preassigned_owner')),'Required source verification missing')
    bootstrap=data['rules']['bootstrap']
    require(bootstrap['paired'] and bootstrap['unit']=='scene' and bootstrap['repetitions']==10000
            and not bootstrap['training_seed_uncertainty_included'],'Bootstrap contract differs')
    groups=('all','moving','ambiguous','stationary')
    bins={r['group']:r for r in data['bins'] if r['horizon_seconds']==2.}
    require(set(bins)==set(groups),'Wrong 2 s groups')
    moving=bins['moving'];arms=('CRN_CV','D','fixed_blend')
    stacked={};reconstruction={}
    for arm in arms:
        item=moving['metrics']['xy']['arms'][arm]
        parts={p:item['parts'][p][CONTRIBUTION] for p in ('covered','uncovered')}
        total=sum(parts.values()); full=item['full_object_mean_epe_m']
        require(np.isclose(total,full,rtol=1e-12,atol=1e-10),'Original-denominator reconstruction failed')
        stacked[arm]=dict(full_mean_xy_epe_m=full,contributions_m=parts)
        reconstruction[arm]=dict(sum_of_contributions_m=total,source_full_mean_m=full,difference_m=total-full)
    differences={g:bins[g]['metrics']['xy']['fixed_blend_minus_CRN_CV'] for g in groups}
    for r in differences.values():
        require(r['defined_draws']==10000 and r['undefined_draws']==0 and
                r['lower95_m']<=r['difference_m']<=r['upper95_m'],'Undefined/invalid original interval')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#ADB6BF','xtick.color':'#47515D','ytick.color':'#47515D',
        'text.color':'#1C2A37','axes.labelcolor':'#1C2A37','pdf.fonttype':42,'ps.fonttype':42})
    fig,(left,right)=plt.subplots(1,2,figsize=(12.8,6.5),gridspec_kw={'width_ratios':[1.,1.04]})
    fig.subplots_adjust(left=.074,right=.966,bottom=.275,top=.79,wspace=.39)
    fig.suptitle(TITLE,x=.074,y=.973,ha='left',fontsize=17,fontweight='bold')
    fig.text(.074,.911,'Fixed-weight diagnostic: CRN-CV on covered points, existing D on uncovered points',fontsize=11.2,color='#536170')
    left.set_title('A   Moving at 2 s: full-support XY EPE',loc='left',pad=15,fontweight='bold')
    x=np.arange(3);covered=np.array([stacked[a]['contributions_m']['covered'] for a in arms])
    uncovered=np.array([stacked[a]['contributions_m']['uncovered'] for a in arms])
    left.bar(x,covered,.56,color=COLORS['covered'],label='Covered contribution',zorder=3)
    left.bar(x,uncovered,.56,bottom=covered,color=COLORS['uncovered'],label='Uncovered contribution',zorder=3)
    for i,arm in enumerate(arms):
        left.text(i,covered[i]/2,f'{covered[i]:.3f}',ha='center',va='center',fontsize=11,color='white',fontweight='bold')
        left.text(i,covered[i]+uncovered[i]/2,f'{uncovered[i]:.3f}',ha='center',va='center',fontsize=11,color='#453523',fontweight='bold')
        left.text(i,covered[i]+uncovered[i]+.10,f"{stacked[arm]['full_mean_xy_epe_m']:.3f}",ha='center',va='bottom',fontsize=11.5,fontweight='bold')
    left.set_xticks(x,['CRN-CV','D','Fixed blend']);left.tick_params(axis='x',length=0,pad=8)
    left.set_ylim(0,5.15);left.set_yticks([0,1,2,3,4,5]);left.set_xlim(-.62,2.62)
    left.set_ylabel('Mean XY EPE (m)  ↓',labelpad=9);left.grid(axis='y',color='#E8EDF0',linewidth=.7)
    left.legend(frameon=False,loc='upper left',fontsize=9.2,ncol=1,handlelength=1.4)
    left.text(0,-.17,'Same full denominator: 1,226 moving objects; 132,148 points.',transform=left.transAxes,fontsize=9.2,color='#536170')
    right.set_title('B   Fixed blend − CRN-CV at 2 s',loc='left',pad=15,fontweight='bold')
    right.axvline(0,color='#AAB2BA',linestyle=(0,(3,3)),linewidth=1,zorder=1)
    for group,y in zip(groups,[3,2,1,0]):
        r=differences[group];v=r['difference_m'];color=COLORS['benefit'] if v<0 else COLORS['cost']
        right.errorbar(v,y,xerr=np.array([[v-r['lower95_m']],[r['upper95_m']-v]]),
            fmt='o',color=color,markersize=6.5,capsize=4,elinewidth=1.9,zorder=3)
        right.text(v,y-.22,ci_text(r),ha='center',va='top',fontsize=9.1,color=color)
    right.set_yticks([3,2,1,0],['All','Moving','Ambiguous','Stationary']);right.tick_params(axis='y',length=0,pad=9)
    right.spines['left'].set_visible(False);right.set_xlim(-.88,.34);right.set_ylim(-.52,3.58)
    right.set_xticks([-.8,-.6,-.4,-.2,0,.2]);right.grid(axis='x',color='#E8EDF0',linewidth=.7)
    right.set_xlabel('Δ mean XY EPE (m), 95% CI  ·  lower is better',labelpad=10)
    right.text(.0,-.25,'Moving improves; ambiguous and stationary errors increase.',transform=right.transAxes,fontsize=9.2,color='#536170')
    fig.text(.074,.115,'Stack heights use subset error sums / each original object’s FULL point count, then equal means over all original objects.',fontsize=9.1,color='#62707D')
    fig.text(.074,.083,'Fixed dev200 / 100 scenes; all 16,074 original object–horizon rows retained. No training, no occupancy evaluation. Rigid-box motion proxy.',fontsize=9.1,color='#62707D')
    fig.text(.074,.051,'95% CI: 10,000 paired scene bootstrap draws, unadjusted; not training-seed uncertainty. 2 s objects: all 3,849; moving 1,226; ambiguous 487; stationary 2,136.',fontsize=8.8,color='#62707D')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    png=args.out_dir/(STEM+'.png');pdf=args.out_dir/(STEM+'.pdf')
    fig.savefig(png,dpi=200,facecolor='white',metadata={'Title':TITLE,'Source SHA256':SOURCE_SHA})
    fig.savefig(pdf,facecolor='white',metadata={'Title':TITLE,'Subject':'Fixed dev200 only; analysis SHA256 '+SOURCE_SHA,'Creator':Path(__file__).name})
    plt.close(fig)
    receipt=dict(schema='fixed-D-coverage-complementarity-figure-v1',title=TITLE,
        source_analysis=str(args.source.resolve()),source_analysis_sha256=SOURCE_SHA,plot_source_sha256=sha(__file__),
        source_provenance=data['sources'],source_rules=data['rules'],source_verification=data['verification'],
        samples=200,scenes=100,object_horizon_rows=16074,horizon_seconds=2.,
        moving_full_denominator_stacks=stacked,stack_reconstruction=reconstruction,
        group_fixed_blend_minus_CRN_CV_xy_epe_m=differences,
        group_support={g:{k:bins[g][k] for k in ('original_object_count','original_point_count','covered_point_count','uncovered_point_count','scenes_with_support')} for g in groups},
        data_modified_or_reestimated=False,model_inference=False,training=False,occupancy_evaluation=False,
        software=dict(matplotlib=matplotlib.__version__,numpy=np.__version__),files_sha256={png.name:sha(png),pdf.name:sha(pdf)})
    output=args.out_dir/(STEM+'.json');output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(png=str(png.resolve()),pdf=str(pdf.resolve()),receipt=str(output.resolve()),
        plot_source_sha256=sha(__file__),stack_reconstruction=reconstruction),indent=2))


if __name__=='__main__':main()
