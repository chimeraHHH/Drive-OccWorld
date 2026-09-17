"""Plot only completed, frozen full native objective-model validation.

Requires real objective-joint-full-summary-v2/complete, raw confusion arrays,
the frozen evaluation/training protocols and post-pilot full authorization.
Four/five models follow candidate_names; dev200 and pilot results are rejected.
All metrics and paired intervals are recomputed with the frozen pure helpers.
No output is written until validation succeeds; no mock or missing-data fill.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
OBJECTIVE_SHA = '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d'
FULL_SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
MERGER_SHA = '0394ff12140891f70f38527f290b7f71e3c6b52bfe76bde80002620c20ac003b'
AGGREGATE_SHA = '9e2327d51dd34e3f207d216b0966177531ffdc4e6ecdddddf0f06004ace5e256'
DEV_PLOT_SHA = '182af003ccebc3f830d96c7b8a96bd59365bff7488c3dbfe16b2af0dceaca03b'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
HORIZONS = [0.,.5,1.,1.5,2.]
VOXELS = 512*512*40


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
    require(path.is_file(), 'Required completed real full input absent; no figure generated: '+str(path))
    data = path.read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Input SHA differs: '+str(path))
    return json.loads(data)


def frozen_module(filename, expected):
    require(Path(filename).name == filename and sha(PKG/filename) == expected, 'Frozen plotting/math/source helper differs: '+filename)
    spec = importlib.util.spec_from_file_location('_objective_full_plot_'+Path(filename).stem,PKG/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def is_sha(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def model_plan(protocol):
    candidates = protocol['candidate_names']
    require(candidates in (['F'],['O'],['F','O']), 'Require a nonempty ordered frozen F/O subset')
    names = tuple(['M0','M0_fp32','native1']+candidates)
    primary = tuple((name,reference) for name in candidates for reference in ('M0_fp32','native1'))
    return names,primary,primary+(('native1','M0_fp32'),('M0_fp32','M0'))


def load_inputs(a):
    path = Path(a.summary); directory = path.parent
    require(not (directory/'failed.json').exists(), 'Full merger has a failure receipt')
    complete = read(directory/'complete.json')
    require(complete['schema'] == 'objective-joint-merge-complete-v2' and complete['status'] == 'COMPLETE' and complete.get('mode') == 'full' and
            complete['samples'] == 5119 and complete['scenes'] == 150 and complete['source_result_files_modified'] is False,
            'Only complete full5119/150-scene results are accepted; no dev/pilot substitution')
    files = complete['files_sha256']
    require(set(files) == {'summary.json','report.md','sample_hash_ledger.json','development_disclosure.json',
                          'raw_confusions.npz','metrics.csv','contrasts.csv'} and
            files['summary.json'] == complete['summary_sha256'], 'Incomplete full-result artifact/hash set')
    summary = read(path,complete['summary_sha256'])
    for filename,expected in files.items():
        require((directory/filename).is_file() and sha(directory/filename) == expected, 'Changed full-result artifact: '+filename)
    require(summary['schema'] == 'objective-joint-full-summary-v2' and summary['status'] == 'COMPLETE_FULL_NATIVE_COMPARISON' and
            summary['samples'] == 5119 and summary['scenes'] == 150 and summary['full_validation'] is True and
            summary['optimizer_steps'] == 0, 'Wrong full-result schema/scope')
    source = summary['sources']; evaluation = read(a.evaluation_protocol,source['evaluation_protocol_sha256'])
    parent = read(a.protocol,PARENT_SHA); objective = read(a.objective_protocol,OBJECTIVE_SHA)
    require(evaluation['schema'] == 'objective-joint-evaluation-protocol-v2' and
            evaluation['status'] == 'FROZEN_BEFORE_OBJECTIVE_JOINT_PILOT' and
            evaluation['parent_protocol_sha256'] == source['parent_protocol_sha256'] == PARENT_SHA and
            evaluation['objective_train_protocol_sha256'] == source['objective_train_protocol_sha256'] == OBJECTIVE_SHA and
            complete['evaluation_protocol_sha256'] == source['evaluation_protocol_sha256'], 'Wrong frozen evaluation/training protocol')
    for key in ('training','numerical_policy','config_sha256','m0_sha256','selection_sha256'):
        require(objective[key] == parent[key], 'Parent/objective contract differs: '+key)
    for key in ('config_sha256','m0_sha256','numerical_policy'):
        require(evaluation[key] == parent[key], 'Evaluation source/precision differs: '+key)
    require(source['evaluation_source_sha256'] == evaluation['source_sha256'] and
            source['parent_source_sha256'] == parent['source_sha256'] and source['objective_source_sha256'] == objective['source_sha256'] and
            source['runtime_source_sha256'] == evaluation['runtime_source_sha256'] == objective['runtime_source_sha256'],
            'Result source dictionary differs from protocols')
    require(summary['merger_sha256'] == evaluation['source_sha256']['objective_joint_merge_v2.py'] == MERGER_SHA and
            sha(PKG/'objective_joint_merge_v2.py') == MERGER_SHA and
            source['producer_sha256'] == evaluation['source_sha256']['objective_joint_evaluation_v2.py'] == sha(PKG/'objective_joint_evaluation_v2.py'),
            'Joint producer/merger source identity differs')
    source_union = {}
    for collection in (parent['source_sha256'],objective['source_sha256'],evaluation['source_sha256']):
        for filename,expected in collection.items():
            require(filename not in source_union or source_union[filename] == expected, 'Conflicting frozen helper source: '+filename)
            require(Path(filename).name == filename and sha(PKG/filename) == expected, 'Frozen package source changed: '+filename)
            source_union[filename] = expected
    require(objective['source_sha256']['objective_supervision_aggregate.py'] == AGGREGATE_SHA, 'Full metric helper is not the frozen parameterized revision')
    for key in ('historical_validation_exposure','unqualified_candidates_not_evaluated_not_zero_scored','F_plus_O_not_run','complete_fresh_input_hash_chain_retained'):
        require(summary[key] is True, 'Required full-validation interpretation missing: '+key)
    for key in ('new_blind_test','threshold_fitting','R4_calibration_reused','multiple_comparison_adjustment','training_seed_robustness_claim','historical_cache_projection_applied'):
        require(summary[key] is False, 'Unsupported full-validation claim: '+key)
    require(summary['fixed_training_seed'] == 11 and summary['training_seeds_added'] == 0 and
            evaluation['historical_validation_exposure'] is True and evaluation['new_blind_test'] is False and
            evaluation['multiple_comparison_adjustment'] is False and evaluation['training_seeds_added'] == 0,
            'Historical exposure/single-seed scope differs')
    names,primary,contrasts = model_plan(evaluation)
    require(summary['candidate_names'] == evaluation['candidate_names'] and summary['model_names'] == list(names) and
            set(summary['models']) == set(names) and summary['registered_primary_contrasts'] == [list(pair) for pair in primary] and
            [(row['left'],row['right']) for row in summary['comparisons']] == list(contrasts), 'Frozen full model/comparison set differs')
    merger = frozen_module('objective_joint_merge_v2.py',MERGER_SHA)
    producer = frozen_module('objective_joint_evaluation_v2.py',evaluation['source_sha256']['objective_joint_evaluation_v2.py'])
    native = frozen_module('joint_native_evaluation.py',evaluation['source_sha256']['joint_native_evaluation.py'])
    _,recipe = producer.derive_run(native,dict(native.__dict__))
    require(source['data_recipe_receipt'] == recipe and len(recipe['edits']) == 9 and
            recipe['inverse_ast_restores_original_run'] is True, 'Frozen v2 native data recipe differs')
    policy = evaluation['inactive_boundary_policy']
    expected_policy = dict(path='inputs.plan_dict.sample_traj',shape=[1,1800,5,3],dtype='float64',
        planning_must_be_disabled=True,comparison_only_projection=True,
        source_function_ast_sha256=merger.CONSUMER_AST_SHA,previous_cache_source_coverage_claim=False)
    require(summary['inactive_boundary_policy'] == policy and set(policy) == set(expected_policy)|{'sources','diagnostic'} and
            all(policy[key] == value for key,value in expected_policy.items()), 'V2 inactive-leaf exception changed')
    evidence = source['inactive_policy_evidence']
    require(evidence['difference_sha256'] == policy['diagnostic']['difference_sha256'] == merger.DIFFERENCE_SHA and
            evidence['complete_sha256'] == policy['diagnostic']['complete_sha256'] == merger.DIAGNOSTIC_COMPLETE_SHA and
            evidence['diagnostic_source_sha256'] == evaluation['source_sha256']['native_boundary_diff.py'] == merger.DIAGNOSTIC_SOURCE_SHA and
            evidence['source_function_ast_sha256'] == policy['source_function_ast_sha256'] and
            evidence['additional_runtime_source_sha256'] == policy['sources'] and
            evidence['previous_cache_source_coverage_claim'] is False and
            evidence['merger_executed_neural_or_runtime_source_inspection'] is False,
            'Missing real diagnostic/source proof or retroactive cache-coverage claim')
    require(len(policy['sources']) == len(producer.ADDITIONAL_SOURCE_SHA), 'Wrong supplementary source set')
    for suffix,expected in producer.ADDITIONAL_SOURCE_SHA.items():
        matches = [path for path in policy['sources'] if Path(path).is_absolute() and path.endswith('/'+suffix)]
        require(len(matches) == 1 and policy['sources'][matches[0]] == expected, 'Supplementary source ledger differs: '+suffix)
    require(summary['nomination_rule'] == evaluation['nomination_rule'] == merger.NOMINATION_RULE, 'Nomination rule differs')
    minimum = summary['registered_minimum_meaningful_effect_pp']
    require(minimum == .5, 'Full plot meaningful-effect reference differs from the fixed protocol')
    require(set(evaluation['final_sources']) == set(names[2:]), 'Wrong trained final source set')
    for key,expected in [('final_checkpoints_sha256',{name:evaluation['final_sources'][name]['files_sha256']['latest.pth'] for name in names[2:]}),
                         ('final_head_state_sha256',{name:evaluation['final_sources'][name]['final_head_state_sha256'] for name in names[2:]})]:
        require(source[key] == expected and all(is_sha(value) for value in expected.values()), 'Wrong frozen final weight digest: '+key)
    require(source['actual_loaded_head_sha256'] == dict(M0=INITIAL_HEAD_SHA,M0_fp32=INITIAL_HEAD_SHA,**source['final_head_state_sha256']),
            'Actually loaded model tensors differ')
    authorization = read(a.full_authorization,source['full_authorization_sha256'])
    require(authorization['schema'] == 'objective-joint-full-authorization-v2' and
            authorization['status'] == 'AUTHORIZED_AFTER_PILOT_PASS' and
            authorization['evaluation_protocol_sha256'] == source['evaluation_protocol_sha256'], 'Missing valid post-pilot full authorization')
    for key in ('complete_sha256','summary_sha256'):
        require(source['pilot_gate'][key] == authorization['pilot_merge'][key] and is_sha(source['pilot_gate'][key]), 'Pilot authorization receipt differs')
    resources,ceiling = authorization['resources'],evaluation['resources']['full']
    require(set(resources) == {'max_seconds','max_allocated_gib','shard_count'} and resources['shard_count'] == 2 and
            type(resources['max_seconds']) is int and 0 < resources['max_seconds'] <= ceiling['max_seconds'] and
            math.isfinite(resources['max_allocated_gib']) and 0 < resources['max_allocated_gib'] <= ceiling['max_allocated_gib'], 'Full authorization exceeds frozen resources')
    require([row['shard_index'] for row in source['shards']] == [0,1] and
            [row['samples'] for row in source['shards']] == [2560,2559] and
            all(row['pilot_reference_sources_sha256'] is None for row in source['shards']), 'Full result does not comprise two complete native shards')
    require(all(is_sha(row[key]) for row in source['shards'] for key in
                ('complete_sha256','index_sha256','manifest_sha256','loaded_models_sha256')), 'Incomplete shard hash ledger')
    for row in source['shards']:
        merger.validate_planning(row['planning_disabled_all_models'],names)
    require(source['selection_sha256'] == evaluation['selections']['full']['sha256'] == FULL_SELECTION_SHA, 'Wrong full selection SHA')
    selection = read(a.selection,FULL_SELECTION_SHA); planned = merger.validate_selection(selection,'full')
    require(selection['config_sha256'] == parent['config_sha256'] and source['annotation_sha256'] == selection['ann_sha256'], 'Full annotation/config identity differs')
    identities = summary['ordered_identities']
    require(len(identities) == 5119 and [row['official_index'] for row in identities] == list(range(5119)) and
            len({row['sample_token'] for row in identities}) == 5119 and len({row['scene_token'] for row in identities}) == 150,
            'Incomplete/repeated native full identities')
    for ordinal,(row,expected) in enumerate(zip(identities,planned)):
        require(all(row.get(key) == value for key,value in expected.items()) and row['selection_ordinal'] == ordinal and
                type(row['data_info_index']) is int and row['data_info_index'] >= 0, 'Full sample identity/ordinal differs')
    ledger = read(directory/'sample_hash_ledger.json',files['sample_hash_ledger.json'])
    require(len(ledger) == 5119, 'Missing sample hash ledger entries')
    for row,identity in zip(ledger,identities):
        require(row['identity'] == identity and row['directory'] == 'samples/'+identity['sample_token'] and
                set(row['logits_array_sha256']) == set(names) and all(is_sha(value) for value in row['logits_array_sha256'].values()) and
                all(is_sha(row[key]) for key in ('records_sha256','complete_sha256','input_tree_sha256','targets_array_sha256',
                                              'observed_bev_tensor_sha256','native_radar_tensor_sha256')) and row['engineering_parity'] is None,
                'Invalid full sample/GT/source/path ledger or forbidden historical-cache parity')
    disclosure = read(directory/'development_disclosure.json',files['development_disclosure.json'])
    require(disclosure['all_five_models_disclosed'] is True and set(disclosure['models']) == {'M0','M0_fp32','native1','F','O'} and
            disclosure['candidate_names'] == evaluation['candidate_names'] and disclosure['samples'] == 200 and disclosure['scenes'] == 100 and
            disclosure['summary_sha256'] == source['development_summary_sha256'] == evaluation['development_summary']['summary_sha256'] and
            disclosure['complete_sha256'] == source['development_complete_sha256'] == evaluation['development_summary']['complete_sha256'],
            'Development selection evidence was not completely retained')
    with np.load(directory/'raw_confusions.npz',allow_pickle=False) as archive:
        require(set(archive.files) == {'models','hist_by_model_sample_horizon','target_counts_0_1_255_by_sample_frame',
                                      'sample_tokens','scene_tokens','official_indices','horizon_seconds'}, 'Wrong raw confusion archive')
        require(archive['models'].tolist() == list(names) and archive['sample_tokens'].tolist() == [row['sample_token'] for row in identities] and
                archive['scene_tokens'].tolist() == [row['scene_token'] for row in identities] and
                archive['official_indices'].tolist() == list(range(5119)) and archive['horizon_seconds'].tolist() == HORIZONS,
                'Raw array model/sample/time axes differ')
        counts = merger.integer_array(archive['target_counts_0_1_255_by_sample_frame'],(5119,7,3),'raw full GT')
        raw = merger.integer_array(archive['hist_by_model_sample_horizon'],(len(names),5119,5,2,2),'raw full confusion')
    require(np.all(counts <= VOXELS) and np.all(counts.sum(-1) == VOXELS), 'Raw GT counts do not exhaust every native grid')
    require(np.all(raw <= VOXELS) and np.array_equal(raw.sum(-1),np.broadcast_to(counts[None,:,2:,:2],raw.sum(-1).shape)),
            'Full raw confusion GT rows differ from original labels')
    metric = frozen_module('aggregate_memory.py',parent['source_sha256']['aggregate_memory.py'])
    aggregate = frozen_module('objective_supervision_aggregate.py',AGGREGATE_SHA)
    histograms = {name:raw[i] for i,name in enumerate(names)}
    models,comparisons,bootstrap = aggregate.aggregate(histograms,[row['scene_token'] for row in identities],metric.metric_arrays,
        minimum,names=names,contrasts=contrasts,primary=primary)
    require(summary['models'] == models and summary['comparisons'] == comparisons and summary['bootstrap'] == bootstrap,
            'Stored full scores/intervals differ from recomputed integer-confusion/10k scene arithmetic')
    require(bootstrap['scene_count'] == 150 and bootstrap['replicates'] == 10000 and bootstrap['seed'] == 11 and
            bootstrap['all_anchors_within_scene_kept_together'] is True and bootstrap['multiple_comparison_adjustment'] is False,
            'Wrong full scene-bootstrap semantics')
    curves = {name:[100*row['gmo_iou'] for row in models[name]['horizons']] for name in names}
    forest = [dict(left=row['left'],right=row['right'],delta_pp=row['metrics']['future_macro_gmo']['delta_pp'],
        lower_pp=row['metrics']['future_macro_gmo']['ci95_pp'][0],upper_pp=row['metrics']['future_macro_gmo']['ci95_pp'][1])
        for row in comparisons if row['primary_contrast']]
    return summary,names,curves,forest,minimum


def write_numbers(out, summary, names):
    fields = ['model','horizon_seconds','plotted','GMO_ratio','GMO_percent','GMO_CI95_lower_percent','GMO_CI95_upper_percent',
              'binary_mIoU_ratio','binary_mIoU_percent','TN','FP','FN','TP','GT_non_GMO','GT_GMO']
    with (out/'horizon_numbers.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for name in names:
            for row in summary['models'][name]['horizons']:
                writer.writerow(dict(model=name,horizon_seconds=row['horizon_seconds'],plotted=row['horizon_seconds']>0,
                    GMO_ratio=row['gmo_iou'],GMO_percent=100*row['gmo_iou'],
                    GMO_CI95_lower_percent=100*row['gmo_iou_ci95'][0],GMO_CI95_upper_percent=100*row['gmo_iou_ci95'][1],
                    binary_mIoU_ratio=row['binary_miou'],binary_mIoU_percent=100*row['binary_miou'],
                    **{key:row[key] for key in fields[9:]}))
    fields = ['left','right','primary','plotted','metric','delta_ratio','delta_pp','CI95_lower_ratio','CI95_upper_ratio','CI95_lower_pp','CI95_upper_pp']
    with (out/'contrast_numbers.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in summary['comparisons']:
            for key,value in row['metrics'].items():
                writer.writerow(dict(left=row['left'],right=row['right'],primary=row['primary_contrast'],
                    plotted=row['primary_contrast'] and key=='future_macro_gmo',metric=key,
                    delta_ratio=value['delta_ratio'],delta_pp=value['delta_pp'],
                    CI95_lower_ratio=value['ci95_ratio'][0],CI95_upper_ratio=value['ci95_ratio'][1],
                    CI95_lower_pp=value['ci95_pp'][0],CI95_upper_pp=value['ci95_pp'][1]))


def render(names, curves, forest, minimum, out):
    style = frozen_module('plot_objective_results.py',DEV_PLOT_SHA)
    colors,labels = style.COLORS,style.LABELS
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
    fig.text(.07,.947,'Full native validation of fixed forecasting models',fontsize=20,weight='bold')
    fig.text(.07,.904,f"5,119 anchors / 150 scenes  |  {len(names)} fixed models · {len(forest)} registered primary contrasts",fontsize=11,color='#637180')
    styles = {'M0':('o',(0,(5,2))),'M0_fp32':('s',(0,(1,2))),'native1':('^','-'),'F':('D','-'),'O':('v','-')}
    for name in names:
        marker,line = styles[name]
        ax.plot(HORIZONS[1:],curves[name][1:],label=labels[name],color=colors[name],linestyle=line,marker=marker,
            markersize=5,linewidth=2.15 if name in ('F','O') else 1.55,
            markerfacecolor='white' if name in ('M0','M0_fp32') else colors[name])
    ax.set_title('(a) Foreground occupancy forecasting',loc='left',weight='bold',pad=16)
    ax.set_xlabel('Future horizon (s)'); ax.set_ylabel('GMO foreground IoU (%)')
    ax.set_xticks(HORIZONS[1:]); ax.set_xlim(.44,2.06)
    values = [value for name in names for value in curves[name][1:]]; span = max(max(values)-min(values),.8)
    ax.set_ylim(max(0,min(values)-.12*span),min(100,max(values)+.15*span))
    ax.grid(axis='y',color='#E4E9ED',linewidth=.75); ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.50,-.23),ncol=2,frameon=False,fontsize=8.9,
              columnspacing=1.1,handlelength=2.7,labelspacing=.85)
    ys = list(reversed(range(len(forest))))
    bounds = [value for row in forest for value in (row['lower_pp'],row['upper_pp'],row['delta_pp'])]+[0.,minimum]
    lower,upper = min(bounds),max(bounds); width = max(upper-lower,.5)
    effect.set_xlim(lower-.12*width,upper+.12*width); effect.set_ylim(-.55,len(forest)-.35)
    for y,row in zip(ys,forest):
        color = colors[row['left']]
        effect.hlines(y,row['lower_pp'],row['upper_pp'],color=color,linewidth=2.3)
        effect.vlines([row['lower_pp'],row['upper_pp']],y-.055,y+.055,color=color,linewidth=1.3)
        effect.plot(row['delta_pp'],y,'D' if row['left']=='F' else 'v',color=color,markersize=6)
        effect.text(.98,y+.23,f"{row['delta_pp']:+.3f}  [{row['lower_pp']:+.3f}, {row['upper_pp']:+.3f}]",
            transform=effect.get_yaxis_transform(),ha='right',va='center',fontsize=9.2,color=color)
    effect.axvline(0,color='#596572',linewidth=1,zorder=1)
    effect.axvline(minimum,color='#B77C45',linewidth=1,linestyle=(0,(4,3)),zorder=1)
    effect.set_yticks(ys,[f"{row['left']} − {'C (native1)' if row['right']=='native1' else row['right']}" for row in forest])
    effect.set_title('(b) Paired effects of frozen candidates',loc='left',weight='bold',pad=16)
    effect.set_xlabel('Future macro GMO difference (pp)')
    effect.grid(axis='x',color='#EEF1F4',linewidth=.65); effect.set_axisbelow(True)
    effect.text(0,-.24,f'Bars: unadjusted 95% paired scene CI; solid line: zero.\nDashed line: registered +{minimum:g} pp reference.',
        transform=effect.transAxes,fontsize=9,color='#637180',va='top',linespacing=1.5)
    t0_text = '; '.join(f'{name} − M0_fp32: {curves[name][0]-curves["M0_fp32"][0]:+.3f} pp' for name in names if name in ('F','O'))
    fig.text(.07,.157,'Current frame (t0), outside the plotted future metric: '+t0_text+'. See CSV for all five horizons.',fontsize=9,color='#4F5D69')
    fig.text(.07,.120,'Full native validation: all 5,119 anchors / 150 scenes. Historical validation exposure remains; this is not a new blind test.',fontsize=9,color='#4F5D69')
    fig.text(.07,.083,'Future macro: mean of four separately pooled foreground IoUs. CI: 10,000 common scene draws, keeping every anchor of each sampled scene.',fontsize=9,color='#4F5D69')
    fig.text(.07,.046,'One training seed (11); no seed-robustness claim. Candidates were frozen by the complete five-model development gate; no selection on full results.',fontsize=9,color='#4F5D69')
    for extension in ('pdf','png','svg'):
        fig.savefig(out/('objective_full_results.'+extension),dpi=240 if extension=='png' else None,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True)
    parser.add_argument('--full-authorization',required=True)
    parser.add_argument('--evaluation-protocol',default=str(PKG/'objective_joint_evaluation_protocol_v2.json'))
    parser.add_argument('--objective-protocol',default=str(PKG/'objective_supervision_protocol_v1.json'))
    parser.add_argument('--protocol',default=str(PKG/'protocol_v2.json'))
    parser.add_argument('--selection',default=str(PKG/'full_validation_selection_v1.json'))
    parser.add_argument('--out',required=True)
    a = parser.parse_args(argv); out = Path(a.out).resolve(); require(not out.exists(),'Output must be a new directory')
    summary,names,curves,forest,minimum = load_inputs(a)
    # Do not create even an empty output until all full-result checks succeed.
    out.mkdir(parents=True,exist_ok=False); render(names,curves,forest,minimum,out); write_numbers(out,summary,names)
    caption = ('Complete native validation on all 5,119 official anchors in 150 scenes. '
        '(a) Foreground GMO IoU at 0.5, 1.0, 1.5 and 2.0 seconds, as percentages; lines connect actual horizons without smoothing. '
        'GMO denotes movable semantic classes, not measured true motion or binary mIoU. Models are '+', '.join(names)+'. '
        'The candidate set was frozen by the preregistered complete five-model development gate before full evaluation. '
        'Unqualified F/O models were not evaluated on full validation, and are not assigned zero scores; their development results remain fully disclosed in the source evidence. '
        '(b) Each frozen candidate is compared with M0_fp32 and C/native1. Points show future macro GMO differences in percentage points. '
        'Each future horizon first pools all anchor confusion matrices before foreground IoU; the four future IoUs are then averaged, excluding t0. '
        'Bars show unadjusted two-sided percentile 95% intervals from 10,000 common scene-bootstrap draws with seed 11, retaining all naturally occurring anchors within a resampled scene. '
        'These intervals reflect fixed-model scene sampling, not training randomness or development/model-selection uncertainty. '
        f'The solid vertical reference is zero; the dashed reference is the registered +{minimum:g} pp meaningful-effect threshold. '
        'M0 retains native TF32 predictions. M0_fp32 and all trained future heads disable matmul TF32; the observed states and cuDNN TF32 remain unchanged. '
        'C/F/O retain the same original M0 initialization and fixed-final 512-update training budget. F uses full GT with the original twelve loss terms; '
        'O uses original coarse GT with CE[1,5] and Lovász, six optimized terms. These are separate interventions, with no combined F+O arm. '
        'The complete validation set remains historically exposed, including development scenes: this is not a new blind test or evidence of training-seed robustness.\n')
    t0_rows = [f'{name}: {curves[name][0]:.6f}% GMO IoU; {curves[name][0]-curves["M0_fp32"][0]:+.6f} pp versus M0_fp32' for name in names if name in ('F','O')]
    caption += ('Current-frame (t0) values are retained in horizon_numbers.csv and excluded from the future macro metric: '+
        '; '.join(t0_rows)+'. These are point differences; this figure does not report a paired t0 interval. '
        'Future improvement does not imply improvement at every horizon. '
        'V2 only projected the source-proven inactive planning sample_traj during pilot/history-cache comparison; '
        'the full evaluation preserved complete fresh inputs and applied no historical-cache projection.\n')
    (out/'caption.txt').write_text(caption)
    filenames = ['objective_full_results.pdf','objective_full_results.png','objective_full_results.svg',
                 'horizon_numbers.csv','contrast_numbers.csv','caption.txt']
    inputs = [a.summary,Path(a.summary).with_name('complete.json'),Path(a.summary).with_name('raw_confusions.npz'),
              Path(a.summary).with_name('sample_hash_ledger.json'),Path(a.summary).with_name('development_disclosure.json'),
              a.evaluation_protocol,a.full_authorization,a.objective_protocol,a.protocol,a.selection]
    receipt = dict(schema='objective-full-results-plot-v2',status='RENDERED_PENDING_VISUAL_QA',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),
        input_files_sha256={str(Path(path).resolve()):sha(path) for path in inputs},
        helper_sha256=dict(development_plot_style=DEV_PLOT_SHA,merger=MERGER_SHA,parameterized_aggregate=AGGREGATE_SHA),
        model_names=list(names),candidate_names=summary['candidate_names'],horizon_seconds=HORIZONS[1:],
        plotted_GMO_percent={name:curves[name][1:] for name in names},plotted_primary_forest_pp=forest,
        unplotted_current_frame_GMO_percent={name:curves[name][0] for name in names},
        current_frame_candidate_delta_vs_M0_fp32_pp={name:curves[name][0]-curves['M0_fp32'][0] for name in names if name in ('F','O')},
        current_frame_paired_interval_computed=False,historical_cache_projection_applied=False,
        metrics_recomputed_from_all_integer_confusions=True,bootstrap_recomputed=True,scene_resamples=10000,
        scene_unit_retains_all_natural_anchors=True,samples=5119,scenes=150,full_validation=True,new_blind_test=False,
        training_seed=11,training_seed_count=1,multiple_comparison_adjustment=False,
        source_results_modified=False,neural_checkpoints_reread=False,raw_image_radar_GT_assets_reread=False,
        additional_smoothing=False,missing_data_substitution=False,
        files_sha256={filename:sha(out/filename) for filename in filenames})
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],out=str(out),models=list(names))),flush=True)


if __name__ == '__main__':
    main()
