"""Plot only authenticated fixed-dev200 summary values; no metric recomputation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_SHA256 = '1588e6e402ff097cbfc2c88594079b6850cc3f7a5e77a676b72c893b8678337c'
STEM = 'shared_rigid_fixed_dev200_result_v1'
TITLE = 'Occupancy gain with no demonstrated motion benefit beyond CRN-CV'
COLORS = dict(D='#75808C', CRN_CV='#253B58', Cpl='#008C95', Fix='#D47A1F')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def ci_text(row, digits=3, scale=1.):
    return (f"{row['difference']*scale:+.{digits}f} "
            f"[{row['lower95']*scale:+.{digits}f}, {row['upper95']*scale:+.{digits}f}]")


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=base/'server_results/training/shared_rigid_formal_campaign_v1/joint_summary.json')
    parser.add_argument('--out-dir', type=Path, default=base/'figures')
    args = parser.parse_args()
    require(sha(args.summary) == SUMMARY_SHA256, 'Fixed completed joint summary SHA differs')
    data = json.loads(args.summary.read_text())
    require(data['schema'] == 'shared-rigid-joint-summary-v1' and data['samples'] == 200
            and data['scenes'] == len(data['scene_tokens']) == 100
            and data['physical_rows_per_arm'] == 16074, 'Wrong fixed evaluation support')
    require(data['bootstrap']['paired'] and data['bootstrap']['unit'] == 'scene'
            and data['bootstrap']['repetitions'] == 10000
            and not data['bootstrap']['training_seed_uncertainty_included'], 'CI contract differs')
    require(data['interpretation']['no_native_O_flow'] and data['interpretation']['fixed512_single_seed11'], 'Endpoint interpretation differs')
    keys = ('future_GMO_IoU_percent', 'future_speed_gt_5_recall_percent')
    panel_a = {metric:{arm:data['common']['comparisons'][arm+'-minus-O'][metric]
                      for arm in ('Cpl','Fix')} for metric in keys}
    moving = sorted((r for r in data['physical']['physical'] if r['group']=='moving'), key=lambda r:r['horizon_seconds'])
    require([r['horizon_seconds'] for r in moving] == [.5,1.,1.5,2.], 'Moving horizons differ')
    for metric in keys:
        for arm in ('Cpl','Fix'):
            row = panel_a[metric][arm]
            require(row['unit'] == 'percentage_points' and row['undefined_bootstrap_repetitions'] == 0,
                    'Invalid difference unit/undefined bootstrap')
            require(row['lower95'] <= row['difference'] <= row['upper95'], 'Unexpected interval ordering')
    panel_b = [dict(horizon_seconds=r['horizon_seconds'], object_anchor_pairs=r['object_anchor_pairs'],
        source_point_occurrences=r['source_point_occurrences'], scenes_with_support=r['scenes_with_support'],
        mean_xy_epe_m={a:r['metrics']['epe_xy_m']['values'][a]['mean'] for a in ('D','CRN_CV','Cpl','Fix')}) for r in moving]
    endpoint = moving[-1]['metrics']['epe_xy_m']
    vs_cv = {a:endpoint['comparisons'][a+'-minus-CRN_CV'] for a in ('Cpl','Fix')}
    for row in vs_cv.values():
        require(row['unit']=='metres' and row['undefined_draws']==0,'Physical CI unit/support differs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#AAB2BA','xtick.color':'#47515D','ytick.color':'#47515D',
        'text.color':'#1C2A37','axes.labelcolor':'#1C2A37','pdf.fonttype':42,'ps.fonttype':42})
    fig, (left,right) = plt.subplots(1,2,figsize=(13.2,6.6),gridspec_kw={'width_ratios':[1.04,1.]})
    fig.subplots_adjust(left=.072,right=.965,bottom=.31,top=.81,wspace=.31)
    fig.suptitle(TITLE,x=.072,y=.975,ha='left',fontsize=17.3,fontweight='bold')
    fig.text(.072,.912,'Shared rigid state: learned spatial addresses (Cpl) versus fixed CV addresses (Fix)',fontsize=11.5,color='#536170')
    left.set_title('A   Occupancy changes relative to O',loc='left',pad=15,fontweight='bold')
    left.axvline(0,color='#AAB2BA',linewidth=1,linestyle=(0,(3,3)),zorder=1)
    ys = ((3.4,2.75),(1.15,.5))
    for metric, positions in zip(keys,ys):
        for arm,y in zip(('Cpl','Fix'),positions):
            r=panel_a[metric][arm]; value=r['difference']
            left.errorbar(value,y,xerr=np.asarray([[value-r['lower95']],[r['upper95']-value]]),
                fmt='o' if arm=='Cpl' else 's',color=COLORS[arm],markersize=6,capsize=4,elinewidth=1.8,zorder=3)
            left.text(value,y-.23,ci_text(r),ha='center',va='top',fontsize=9,color=COLORS[arm])
    left.text(-1.91,3.98,'Future GMO IoU',fontweight='bold',fontsize=10.5)
    left.text(-1.91,1.73,'High-speed recall (>5 m/s)',fontweight='bold',fontsize=10.5)
    left.set_yticks([3.4,2.75,1.15,.5],['Cpl','Fix','Cpl','Fix'])
    left.tick_params(axis='y',length=0,pad=9)
    left.set_xlim(-2.,.75);left.set_ylim(-.04,4.27)
    left.set_xticks([-2,-1.5,-1,-.5,0,.5])
    left.set_xlabel('Difference from O (percentage points; 95% CI)',labelpad=10)
    left.spines['left'].set_visible(False);left.grid(axis='x',color='#E9EDF0',linewidth=.7)
    left.text(.0,-.27,'Cpl − Fix future GMO: '+ci_text(data['common']['comparisons']['Cpl-minus-Fix'][keys[0]],4)+' pp',
              transform=left.transAxes,fontsize=9.2,color='#536170',va='top')
    right.set_title('B   Moving-object displacement error',loc='left',pad=15,fontweight='bold')
    horizons=np.asarray([r['horizon_seconds'] for r in panel_b])
    styles=dict(D=dict(linestyle='--',marker='x',markersize=6,linewidth=1.8),
                CRN_CV=dict(linestyle='-',marker='o',markersize=9,markerfacecolor='white',linewidth=2.),
                Cpl=dict(linestyle=':',marker='D',markersize=4,linewidth=1.2),
                Fix=dict(linestyle='--',marker='+',markersize=7,linewidth=1.3))
    for a in ('D','CRN_CV','Cpl','Fix'):
        values=[r['mean_xy_epe_m'][a] for r in panel_b]
        right.plot(horizons,values,color=COLORS[a],label='CRN-CV' if a=='CRN_CV' else a,**styles[a])
    right.set_xlim(.42,2.12);right.set_ylim(.5,4.65)
    right.set_xticks(horizons,['0.5','1.0','1.5','2.0'])
    right.set_yticks([1,2,3,4]);right.set_xlabel('Prediction horizon (s)',labelpad=10)
    right.set_ylabel('Mean XY EPE (m)  ↓',labelpad=8);right.grid(axis='y',color='#E9EDF0',linewidth=.7)
    right.legend(frameon=False,ncol=2,loc='upper left',fontsize=9.5,handlelength=2.2,columnspacing=1.4)
    right.annotate('CRN-CV, Cpl and Fix\nnearly overlap',xy=(1.48,panel_b[2]['mean_xy_epe_m']['CRN_CV']),
        xytext=(.62,3.0),fontsize=9.5,color='#536170',arrowprops=dict(arrowstyle='-',color='#909CA8',lw=.9),
        bbox=dict(boxstyle='round,pad=.27',facecolor='white',edgecolor='none',alpha=.9))
    right.text(.0,-.27,'2 s candidate − CRN-CV (mm), 95% CI\n'
        +'Cpl '+ci_text(vs_cv['Cpl'],2,1000.)+';  Fix '+ci_text(vs_cv['Fix'],2,1000.),
        transform=right.transAxes,fontsize=9.2,color='#536170',va='top',linespacing=1.5)
    fig.text(.072,.092,'Fixed dev200 / 100 scenes · final 512 updates · seed 11. Physical evaluation retains all 16,074 original object–horizon rows per arm.',fontsize=9,color='#62707D')
    fig.text(.072,.064,'CIs: 10,000 paired scene bootstrap draws, unadjusted; not training-seed uncertainty. EPE: rigid-box proxy, equal object weighting. O has no native flow.',fontsize=9,color='#62707D')
    fig.text(.072,.036,'Future occupancy metrics: mean of four horizon metrics after pooling within each horizon. Moving support per horizon: '+', '.join(str(r['object_anchor_pairs']) for r in panel_b)+'.',fontsize=9,color='#62707D')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    png=args.out_dir/(STEM+'.png');pdf=args.out_dir/(STEM+'.pdf')
    fig.savefig(png,dpi=200,facecolor='white',metadata={'Title':TITLE,'Source SHA256':SUMMARY_SHA256})
    fig.savefig(pdf,facecolor='white',metadata={'Title':TITLE,'Subject':'Fixed dev200 only; summary SHA256 '+SUMMARY_SHA256,'Creator':Path(__file__).name})
    plt.close(fig)
    receipt=dict(schema='shared-rigid-fixed-dev200-result-figure-v1',title=TITLE,
        source_summary=str(args.summary.resolve()),source_summary_sha256=SUMMARY_SHA256,
        plot_source_sha256=sha(__file__),analysis_source_sha256=data['analysis_source_sha256'],
        endpoint_authentication=data['endpoint_authentication'],samples=200,scenes=100,
        physical_rows_per_arm=16074,bootstrap=data['bootstrap'],interpretation=data['interpretation'],
        plotted_common_differences_pp=panel_a,plotted_moving_xy_epe=panel_b,
        moving_2s_candidate_minus_CV_m=vs_cv,
        common_Cpl_minus_Fix_future_GMO_pp=data['common']['comparisons']['Cpl-minus-Fix'][keys[0]],
        display_only_conversion='2 s difference annotation m×1000 to mm; no other unit conversion or metric recomputation',
        O_physical_values_invented=False,other_sample_sets_used=False,model_inference_or_training=False,
        software=dict(matplotlib=matplotlib.__version__,numpy=np.__version__),
        files_sha256={png.name:sha(png),pdf.name:sha(pdf)})
    output=args.out_dir/(STEM+'.json');output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(png=str(png.resolve()),pdf=str(pdf.resolve()),receipt=str(output.resolve()),plot_source_sha256=sha(__file__)),indent=2))


if __name__=='__main__': main()
