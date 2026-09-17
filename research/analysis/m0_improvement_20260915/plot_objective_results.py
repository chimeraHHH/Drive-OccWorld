"""Plot only complete, SHA-bound real F/O results; no missing-data substitution.

Reads local campaign_objective_v1/summary_v1 plus frozen objective/parent
protocols. Produces five-model future-horizon curves, the four registered paired
effects, exact numeric CSVs, caption and an input/output SHA receipt. No remote
host, neural checkpoint, raw cache, training, threshold fitting or skills.
Nothing is written until every required input has passed validation.
"""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

PKG = Path(__file__).resolve().parent
MODELS = ('M0', 'M0_fp32', 'native1', 'F', 'O')
HORIZONS = (0., .5, 1., 1.5, 2.)
PRIMARY = (('F', 'M0_fp32'), ('F', 'native1'), ('O', 'M0_fp32'), ('O', 'native1'))
CONTRASTS = PRIMARY + (('native1', 'M0_fp32'), ('M0_fp32', 'M0'))
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
COLORS = {'M0': '#263238', 'M0_fp32': '#9099A3', 'native1': '#3775AA', 'F': '#7655A3', 'O': '#137F77'}
LABELS = {'M0': 'M0 · native precision', 'M0_fp32': 'M0_fp32 · frozen precision reference',
          'native1': 'C · native1 continuation', 'F': 'F · full GT, original 12 losses',
          'O': 'O · coarse GT, CE + Lovász'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    path = Path(path)
    require(path.is_file(), 'Required completed real input absent; no figure generated: '+str(path))
    data = path.read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Input SHA differs: '+str(path))
    return json.loads(data)


def close(left, right):
    return (type(left) in (int, float) and type(right) in (int, float) and
            math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12))


def interval(value, ratio=False):
    require(isinstance(value, list) and len(value) == 2 and
            all(type(x) in (int, float) and math.isfinite(x) for x in value) and value[0] <= value[1],
            'Invalid finite ordered interval')
    if ratio:
        require(0 <= value[0] <= value[1] <= 1, 'Score interval is outside [0,1]')
    return value


def confusion(matrix):
    require(isinstance(matrix, list) and len(matrix) == 2 and
            all(isinstance(row, list) and len(row) == 2 for row in matrix) and
            all(type(x) is int and x >= 0 for row in matrix for x in row),
            'Confusion must contain nonnegative integer GT rows/prediction columns')
    return matrix


def scores(matrix):
    (tn, fp), (fn, tp) = confusion(matrix)
    require(tp+fp+fn > 0 and tn+fp+fn > 0, 'Undefined class IoU')
    fg, bg = tp/(tp+fp+fn), tn/(tn+fp+fn)
    return fg, bg, .5*(fg+bg)


def validate_identities(summary):
    rows = summary['ordered_identities_and_GT']
    counts = Counter(row['scene_token'] for row in rows)
    require(len(rows) == len({row['sample_token'] for row in rows}) == 200 and
            len(counts) == 100 and set(counts.values()) == {2}, 'Expected dev200, exactly two anchors in each of 100 scenes')
    pooled = [[0,0] for _ in HORIZONS]
    for row in rows:
        require(row['split'] == 'development', 'Wrong data role')
        frames = row['target_counts_0_1_255_by_frame']
        require(len(frames) == 7 and all(len(frame) == 3 and all(type(x) is int and x >= 0 for x in frame)
                and sum(frame) == 512*512*40 for frame in frames), 'GT class counts do not exhaust the original seven grids')
        digest = row['targets_sha256']
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                'Missing target SHA')
        for h in range(5):
            for cls in range(2):
                pooled[h][cls] += frames[h+2][cls]
    require(summary['bootstrap']['scene_order'] == sorted(counts), 'Bootstrap scene identities differ')
    return pooled


def validate_model(model, pooled_gt):
    rows = model['horizons']
    require([row['horizon_seconds'] for row in rows] == list(HORIZONS), 'Five-horizon contract differs')
    values = []; binary = []; future = [[0,0], [0,0]]
    for h, row in enumerate(rows):
        matrix = confusion(row['confusion_GT_rows_prediction_columns']); fg, bg, miou = scores(matrix)
        require([sum(r) for r in matrix] == pooled_gt[h], 'Model/target GT totals differ')
        require(all(close(got, row[key]) for got, key in [(fg,'gmo_iou'), (bg,'non_gmo_iou'), (miou,'binary_miou')]),
                'GMO/non-GMO/binary mIoU must match original integer confusion')
        require([matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]] == [row[key] for key in ('TN','FP','FN','TP')] and
                [sum(r) for r in matrix] == [row['GT_non_GMO'], row['GT_GMO']], 'Stored counts differ from confusion')
        interval(row['gmo_iou_ci95'], ratio=True); interval(row['binary_miou_ci95'], ratio=True)
        values.append(fg); binary.append(miou)
        if h > 0:
            for i in range(2):
                for j in range(2):
                    future[i][j] += matrix[i][j]
    ffg, _, fmiou = scores(future)
    expected = dict(future_macro_gmo=sum(values[1:])/4, future_macro_binary_miou=sum(binary[1:])/4,
                    future_pooled_gmo=ffg, future_pooled_binary_miou=fmiou)
    for key, value in expected.items():
        metric = model['metrics'][key]
        require(metric['unit'] == 'ratio' and close(value, metric['estimate']), 'Metric definition differs: '+key)
        interval(metric['ci95'], ratio=True)
    require(future[0][1] == model['future_FP'] and future[1][0] == model['future_FN'], 'Future FP/FN pooling differs')
    return [100*x for x in values]


def load_inputs(a):
    path = Path(a.summary)
    require(not path.with_name('failed.json').exists(), 'Aggregation has a failure receipt')
    complete = read(path.with_name('complete.json'))
    require(complete['schema'] == 'm0-objective-supervision-aggregate-complete-v1' and complete['status'] == 'COMPLETE',
            'Need the completed real five-model objective aggregation')
    files = complete['files_sha256']
    require(set(files) == {'summary.json','report.md','metrics.csv','contrasts.csv'} and
            files['summary.json'] == complete['summary_sha256'], 'Incomplete aggregate artifact/hash set')
    summary = read(path, complete['summary_sha256'])
    for name, digest in files.items():
        require((path.parent/name).is_file() and sha(path.parent/name) == digest, 'Changed aggregate artifact: '+name)
    require(summary['schema'] == 'm0-objective-supervision-aggregate-v1' and
            summary['status'] == 'COMPLETED_DEVELOPMENT_COMPARISON', 'Wrong/incomplete result schema')
    protocol = read(a.objective_protocol, summary['objective_protocol_sha256']); parent = read(a.protocol, PARENT_SHA)
    require(protocol['schema'] == 'objective-supervision-training-protocol-v1' and
            protocol['status'] == 'FROZEN_BEFORE_F_O_TRAINING' and protocol['arms'] == ['F','O'] and
            complete['objective_protocol_sha256'] == summary['objective_protocol_sha256'] and
            protocol['parent_protocol_sha256'] == summary['parent_protocol_sha256'] == PARENT_SHA, 'Protocol binding differs')
    for key in ('training','numerical_policy','config_sha256','m0_sha256','selection_sha256'):
        require(protocol[key] == parent[key], 'Candidate/C protocol mismatch: '+key)
    require(summary['config_sha256'] == parent['config_sha256'] and
            summary['m0_checkpoint_sha256'] == parent['m0_sha256'] and summary['selection_sha256'] == parent['selection_sha256'] and
            summary['cache_index_sha256'] == protocol['cache_index_sha256']['development'], 'Result/source/cache identity differs')
    require(summary['script_sha256'] == protocol['source_sha256']['objective_supervision_aggregate.py'] and
            sha(PKG/'objective_supervision_aggregate.py') == summary['script_sha256'], 'Aggregator source identity differs')
    require(summary['samples'] == complete['samples'] == 200 and summary['scenes'] == complete['scenes'] == 100 and
            set(summary['models']) == set(MODELS) and complete['models'] == list(MODELS), 'Five-model development scope differs')
    hp = parent['training']
    require(hp['seed'] == 11 and hp['train_samples'] == 512 and hp['development_samples'] == 200 and
            hp['passes'] == 4 and hp['accumulate'] == 4, 'Wrong registered training exposure')
    audit, semantics, bootstrap = summary['audit'], summary['semantics'], summary['bootstrap']
    require(audit['status'] == 'PASS' and all(audit[key] is True for key in
        ('all_five_models_same_full_GT_row_counts', 'all_exactly_200_same_order',
         'actual_three_final_checkpoints_rehashed_and_payload_checked_on_CPU',
         'actual_512_updates_2048_examples_LR_and_orders_checked', 'same_weight_structure_and_M0_initialization')),
        'Required producer checkpoint/order/GT audit did not pass')
    require(semantics['full_validation_result'] is False and semantics['training_seeds'] == 1 and
            semantics['training_seed_robustness_claim'] is False and semantics['threshold_fitting'] is False and
            semantics['historical_val_exposure'] is True and semantics['development_screen_not_new_blind_test'] is True and
            semantics['R4_calibration_reused'] is False and semantics['multiple_comparison_adjustment'] is False and
            semantics['precision'] == protocol['numerical_policy'] and semantics['F_plus_O_not_run'] is True and
            semantics['factorial_interaction_not_identified'] is True, 'Interpretation/precision contract differs')
    require(bootstrap['replicates'] == 10000 and bootstrap['seed'] == 11 and bootstrap['unit'] == 'scene' and
            bootstrap['paired'] is True and bootstrap['scene_count'] == 100 and bootstrap['model_order'] == list(MODELS) and
            bootstrap['all_anchors_within_scene_kept_together'] is True and bootstrap['multiple_comparison_adjustment'] is False and
            bootstrap['primary_contrasts'] == 4, 'Expected 10k common paired scene draws with four unadjusted contrasts')
    evaluation = protocol['evaluation']
    require(evaluation['primary_contrasts'] == summary['registered_primary_contrasts'] == [list(x) for x in PRIMARY] and
            evaluation['historical_validation_exposure'] is True and evaluation['multiple_comparison_adjustment'] is False,
            'Registered evaluation contract differs')
    pooled_gt = validate_identities(summary)
    curves = {name: validate_model(summary['models'][name], pooled_gt) for name in MODELS}
    rows = summary['comparisons']; require([(row['left'], row['right']) for row in rows] == list(CONTRASTS),
                                        'Missing, repeated or reordered contrasts')
    forest = []
    for row in rows:
        left, right = row['left'], row['right']; primary = (left,right) in PRIMARY
        require(row['primary_contrast'] is primary and row['precision_only_contrast'] is ((left,right) == ('M0_fp32','M0')),
                'Comparison role differs')
        for key, value in row['metrics'].items():
            delta = summary['models'][left]['metrics'][key]['estimate']-summary['models'][right]['metrics'][key]['estimate']
            require(close(delta,value['delta_ratio']) and close(100*delta,value['delta_pp']), 'Effect or percentage-point conversion differs')
            interval(value['ci95_ratio']); interval(value['ci95_pp'])
            require(all(close(100*x,y) for x,y in zip(value['ci95_ratio'],value['ci95_pp'])), 'CI unit conversion differs')
        if primary:
            value = row['metrics']['future_macro_gmo']; lo, hi = value['ci95_pp']
            forest.append(dict(left=left, right=right, delta_pp=value['delta_pp'], lower_pp=lo, upper_pp=hi))
    minimum = summary['registered_minimum_meaningful_effect_pp']; registered = evaluation['minimum_meaningful_effect_pp']
    require((minimum is None and registered is None) or (close(minimum,registered) and minimum >= 0), 'Effect reference differs from protocol')
    return summary, curves, forest, minimum


def write_numbers(out, summary):
    fields = ['model','horizon_seconds','plotted','GMO_ratio','GMO_percent','GMO_CI95_lower_percent','GMO_CI95_upper_percent',
              'binary_mIoU_ratio','binary_mIoU_percent','TN','FP','FN','TP','GT_non_GMO','GT_GMO']
    with (out/'horizon_numbers.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in MODELS:
            for row in summary['models'][name]['horizons']:
                writer.writerow(dict(model=name,horizon_seconds=row['horizon_seconds'],plotted=row['horizon_seconds']>0,
                    GMO_ratio=row['gmo_iou'],GMO_percent=100*row['gmo_iou'],
                    GMO_CI95_lower_percent=100*row['gmo_iou_ci95'][0],GMO_CI95_upper_percent=100*row['gmo_iou_ci95'][1],
                    binary_mIoU_ratio=row['binary_miou'],binary_mIoU_percent=100*row['binary_miou'],
                    **{key:row[key] for key in fields[9:]}))
    fields = ['left','right','primary','plotted','metric','delta_ratio','delta_pp','CI95_lower_ratio','CI95_upper_ratio',
              'CI95_lower_pp','CI95_upper_pp']
    with (out/'contrast_numbers.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in summary['comparisons']:
            for key,value in row['metrics'].items():
                writer.writerow(dict(left=row['left'],right=row['right'],primary=row['primary_contrast'],
                    plotted=row['primary_contrast'] and key=='future_macro_gmo',metric=key,
                    delta_ratio=value['delta_ratio'],delta_pp=value['delta_pp'],
                    CI95_lower_ratio=value['ci95_ratio'][0],CI95_upper_ratio=value['ci95_ratio'][1],
                    CI95_lower_pp=value['ci95_pp'][0],CI95_upper_pp=value['ci95_pp'][1]))


def render(curves, forest, minimum, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#68737E','axes.linewidth':.7,
        'xtick.color':'#45505B','ytick.color':'#45505B','text.color':'#25303A','axes.labelcolor':'#25303A',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig = plt.figure(figsize=(14.6,7.3),facecolor='white')
    gs = fig.add_gridspec(1,2,left=.07,right=.97,top=.80,bottom=.35,width_ratios=[1.12,1.],wspace=.39)
    ax = fig.add_subplot(gs[0,0]); effect = fig.add_subplot(gs[0,1])
    fig.text(.07,.947,'Supervision resolution and objective composition',fontsize=20,weight='bold')
    fig.text(.07,.904,'Completed development comparison  |  Five fixed models · four registered primary contrasts',fontsize=11,color='#637180')
    styles = {'M0':('o',(0,(5,2))),'M0_fp32':('s',(0,(1,2))),'native1':('^','-'),'F':('D','-'),'O':('v','-')}
    for name in MODELS:
        marker, style = styles[name]
        ax.plot(HORIZONS[1:],curves[name][1:],label=LABELS[name],color=COLORS[name],linestyle=style,
            marker=marker,markersize=5,linewidth=2.15 if name in ('F','O') else 1.55,
            markerfacecolor='white' if name in MODELS[:2] else COLORS[name])
    ax.set_title('(a) Foreground occupancy forecasting',loc='left',weight='bold',pad=16)
    ax.set_xlabel('Future horizon (s)'); ax.set_ylabel('GMO foreground IoU (%)')
    ax.set_xticks(HORIZONS[1:]); ax.set_xlim(.44,2.06)
    values = [x for name in MODELS for x in curves[name][1:]]; span = max(max(values)-min(values),.8)
    ax.set_ylim(max(0,min(values)-.12*span),min(100,max(values)+.15*span))
    ax.grid(axis='y',color='#E4E9ED',linewidth=.75); ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.50,-.23),ncol=2,frameon=False,fontsize=8.9,
              columnspacing=1.1,handlelength=2.7,labelspacing=.85)
    ys = [3,2,1,0]; bounds = [x for row in forest for x in (row['lower_pp'],row['upper_pp'],row['delta_pp'])]+[0.]
    if minimum is not None:
        bounds.append(minimum)
    lower, upper = min(bounds),max(bounds); width = max(upper-lower,.5)
    effect.set_xlim(lower-.12*width,upper+.12*width); effect.set_ylim(-.55,3.65)
    for y,row in zip(ys,forest):
        color = COLORS[row['left']]
        effect.hlines(y,row['lower_pp'],row['upper_pp'],color=color,linewidth=2.3)
        effect.vlines([row['lower_pp'],row['upper_pp']],y-.055,y+.055,color=color,linewidth=1.3)
        effect.plot(row['delta_pp'],y,'D' if row['left']=='F' else 'v',color=color,markersize=6)
        effect.text(.98,y+.23,f"{row['delta_pp']:+.3f}  [{row['lower_pp']:+.3f}, {row['upper_pp']:+.3f}]",
            transform=effect.get_yaxis_transform(),ha='right',va='center',fontsize=9.2,color=color)
    effect.axvline(0,color='#596572',linewidth=1,zorder=1)
    if minimum is not None:
        effect.axvline(minimum,color='#B77C45',linewidth=1,linestyle=(0,(4,3)),zorder=1)
    effect.set_yticks(ys,[f"{row['left']} − {'C (native1)' if row['right']=='native1' else row['right']}" for row in forest])
    effect.set_title('(b) Paired effects of F and O',loc='left',weight='bold',pad=16)
    effect.set_xlabel('Future macro GMO difference (pp)')
    effect.grid(axis='x',color='#EEF1F4',linewidth=.65); effect.set_axisbelow(True)
    guide = 'Bars: unadjusted 95% paired scene CI; solid line: zero.'
    if minimum is not None:
        guide += f'\nDashed line: registered +{minimum:g} pp reference.'
    effect.text(0,-.24,guide,transform=effect.transAxes,fontsize=9,color='#637180',va='top',linespacing=1.5)
    fig.text(.07,.132,'Same 200 development anchors / 100 scenes; one training seed (11). Historically exposed validation data; not full 5,119-anchor validation.',fontsize=9,color='#4F5D69')
    fig.text(.07,.095,'Future macro: mean of four separately pooled foreground IoUs. CI: 10,000 common scene draws; not training-seed or model-selection uncertainty.',fontsize=9,color='#4F5D69')
    fig.text(.07,.058,'F and O are separate interventions; F + O is not run. M0_fp32 and all trained heads share the same matmul precision; native observed states are unchanged.',fontsize=9,color='#4F5D69')
    for extension in ('pdf','png','svg'):
        fig.savefig(out/('objective_results.'+extension),dpi=240 if extension=='png' else None,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',default=str(PKG/'server_results/campaign_objective_v1/summary_v1/summary.json'))
    parser.add_argument('--objective-protocol',default=str(PKG/'objective_supervision_protocol_v1.json'))
    parser.add_argument('--protocol',default=str(PKG/'protocol_v2.json'))
    parser.add_argument('--out',required=True)
    a = parser.parse_args(argv); out = Path(a.out).resolve(); require(not out.exists(),'Output must be a new directory')
    summary,curves,forest,minimum = load_inputs(a)
    # All validation precedes the first filesystem mutation; absent F/O results
    # cannot produce empty charts, mock points or apparently complete figures.
    out.mkdir(parents=True,exist_ok=False)
    render(curves,forest,minimum,out); write_numbers(out,summary)
    caption = ('Completed development comparison on the same 200 anchors from 100 scenes, with one training seed (11). '
        '(a) Foreground GMO IoU at 0.5, 1.0, 1.5 and 2.0 seconds. GMO denotes movable semantic classes, not a true-motion mask or binary mIoU. '
        'Lines connect measured horizons without smoothing. M0 and M0_fp32 are frozen references; C/native1, F and O use fixed final '
        'checkpoints after 512 optimizer updates, 2,048 example exposures, four passes and accumulation four. F uses full GT with the original '
        '12 loss terms; O uses the original coarse GT with CE[1,5] and Lovász, six optimized terms. O retains sem/geo forward computation. '
        '(b) Points show the four registered primary differences F−M0_fp32, F−C, O−M0_fp32 and O−C in future macro GMO, in percentage points. '
        'Each future horizon pools confusion matrices across anchors before IoU is computed; the resulting four IoUs are averaged. '
        'Current-time IoU is excluded from this mean and panel (a), and remains available in the numeric CSV. Bars show unadjusted two-sided '
        '95% percentile intervals from 10,000 paired scene-bootstrap draws, seed 11; both anchors within each scene remain together. '
        'Intervals describe fixed-model scene sampling, not neural training randomness or historical selection uncertainty. The solid line is zero. '+
        (f'The dashed line denotes the registered +{minimum:g} pp meaningful-effect reference. ' if minimum is not None else '')+
        'M0 retains native TF32 cache predictions. M0_fp32 and all trained future heads disable matmul TF32; the observed t0 cache and cuDNN '
        'TF32 remain unchanged. F/O are separate interventions without a combined F+O arm, so no factorial interaction is identified. '
        'This is historically exposed development data, not a new blind test or full 5,119-anchor validation.\n')
    (out/'caption.txt').write_text(caption)
    files = ['objective_results.pdf','objective_results.png','objective_results.svg','horizon_numbers.csv','contrast_numbers.csv','caption.txt']
    receipt = dict(schema='objective-results-plot-v1',status='RENDERED_PENDING_VISUAL_QA',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_files_sha256={str(Path(path).resolve()):sha(path) for path in
            (a.summary,Path(a.summary).with_name('complete.json'),a.objective_protocol,a.protocol)},
        aggregator_sha256=summary['script_sha256'],plotted_models=list(MODELS),horizon_seconds=list(HORIZONS[1:]),
        plotted_GMO_percent={name:curves[name][1:] for name in MODELS},plotted_primary_forest_pp=forest,
        meaningful_effect_reference_pp=minimum,metrics_recomputed_from_integer_confusion=True,
        bootstrap_recomputed=False,neural_checkpoints_reread=False,raw_large_cache_reread=False,
        source_results_modified=False,additional_smoothing=False,missing_data_substitution=False,
        samples=200,scenes=100,training_seed=11,training_seed_count=1,full_validation=False,
        multiple_comparison_adjustment=False,files_sha256={name:sha(out/name) for name in files})
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],out=str(out))),flush=True)


if __name__ == '__main__':
    main()
