"""V2 hash-checked merging with the one source-proven inactive pilot leaf.

Actual model inputs remain complete. Only the pilot/history-cache equality
projects inputs.plan_dict.sample_traj; the full-tree equality is reported honestly.

Pilot: exactly two selected anchors in two scenes, engineering evidence only.
Full: all 5119 official anchors/150 scenes, with fixed models M0/M0_fp32/C plus
the ordered qualifying F/O subset registered before the pilot. Only the frozen
objective aggregate's parameterized mathematics is used; no dev200 main/render,
inference, optimization, threshold fitting or source-result mutation occurs.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np

PKG = Path(__file__).resolve().parent
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
OBJECTIVE_SHA = '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d'
OBJECTIVE_AGGREGATE_SHA = '9e2327d51dd34e3f207d216b0966177531ffdc4e6ecdddddf0f06004ace5e256'
SELECTION_SHA = {'pilot':'e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d',
                 'full':'60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'}
M0_SHA = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
SOURCE_MODEL_SHA = '67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
NOMINATION_RULE = dict(metric='future_macro_gmo',comparators=['M0_fp32','native1'],minimum_point_delta_pp=.5,
    minimum_ci_lower_pp=0.,ci_lower_strict=True,include_all_qualifying_in_F_O_order=True)
HORIZONS = [0.,.5,1.,1.5,2.]
VOXELS = 512*512*40
DIFFERENCE_SHA = 'aecf29305e3d4b086e834df4b391580e0964a330a22f4d91c09566b4036b95a5'
DIAGNOSTIC_COMPLETE_SHA = '9577fa4cadd0c1be8c8c189dba6f85e8ed330ea912f7f7047967ced717f635fb'
DIAGNOSTIC_SOURCE_SHA = 'b15ba346382245c2736552fdb651aa6b719d05caa9b999fea8db89b008962d74'
CONSUMER_AST_SHA = '2593f6549742ec99c7c0b9877d63361c57f3126012626f51b7e96db7d494cbba'
V1_PRODUCER_SHA = 'ab0b80aa14ecd8587077e20cf9fbc06c6c8df724f7e0a3c8c3abece2285ee9c7'
V1_MERGER_SHA = '0f41248104fe26e1660d672f0bd2cf7b88b94f61f795ab2836dd2969fef77330'


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
    require(path.is_file(), 'Missing complete real input: '+str(path))
    data = path.read_bytes(); observed = hashlib.sha256(data).hexdigest()
    require(expected is None or observed == expected, 'Hash mismatch: '+str(path))
    return json.loads(data), observed


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def is_sha(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def integer_array(value, shape, label):
    result = np.asarray(value)
    require(result.shape == shape and np.issubdtype(result.dtype,np.integer) and
            np.all(result >= 0) and np.all(result <= np.iinfo(np.int64).max), 'Invalid integer counts: '+label)
    return result.astype(np.int64)


def frozen_module(filename, expected):
    require(Path(filename).name == filename and sha(PKG/filename) == expected, 'Frozen local source differs: '+filename)
    spec = importlib.util.spec_from_file_location('_objective_joint_merge_'+Path(filename).stem,PKG/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_selection(selection, mode):
    require(selection['schema'] == 'native-joint-validation-selection-v1','Wrong selection schema')
    rows = selection['records']; indices = [row['official_index'] for row in rows]
    require(len(rows) == selection['samples'] and all(type(i) is int for i in indices) and
            indices == sorted(set(indices)), 'Selection count/order differs')
    require(len({row['sample_token'] for row in rows}) == len(rows) and
            len({row['scene_token'] for row in rows}) == selection['scenes'], 'Selection identity coverage differs')
    require(all(row['split'] == 'validation' and isinstance(row['sample_token'],str) and
                len(row['sample_token']) == 32 and all(c in '0123456789abcdef' for c in row['sample_token']) for row in rows),
            'Wrong selection split or unsafe sample token')
    require(selection['historical_validation_exposure'] is True and selection['training_seed'] == 11 and
            selection['training_repeats_added'] == 0, 'Selection scientific scope differs')
    if mode == 'full':
        require(indices == list(range(5119)) and selection['scenes'] == 150 and
                selection['scope'] == 'full_native_validation', 'Full must cover all official5119/150 scenes')
    else:
        require(len(rows) == selection['scenes'] == 2 and
                selection['scope'] == 'engineering_only_two_cached_validation_anchors', 'Pilot must cover exactly two frozen anchors/scenes')
    return rows


def validate_planning(audit, models):
    require(isinstance(audit,dict) and set(audit) == set(models), 'Incomplete runtime planning audit')
    for name in models:
        value = audit[name]
        require(isinstance(value,dict) and set(value) == {'turn_on_plan','all_modules_eval'} and
                value['turn_on_plan'] is False and value['all_modules_eval'] is True,
                'Inactive exception requires every model planner disabled and every module eval: '+name)


def validate_boundary_parity(result, models):
    """Validate truthful full/projected hashes; never interpret projection as input."""
    parity = result['engineering_parity']
    flags = {'active_boundary_hash_equal','exact_gt_equal','native_all_5h_3layer_logits_bitwise',
             'all_models_5_horizons_full_GT_confusion_equal'}
    require(isinstance(parity,dict) and set(parity) == flags|{'full_boundary_hash_equal','inactive_random_sample_traj_audit'} and
            all(parity[k] is True for k in flags), 'Pilot active/GT/prediction parity is incomplete')
    audit = parity['inactive_random_sample_traj_audit']
    fields = {'path','comparison_only_projection','actual_inputs_modified','fresh_full_input_sha256','cache_full_input_sha256',
        'fresh_projected_input_sha256','cache_projected_input_sha256','fresh_field_sha256','cache_field_sha256',
        'fresh_shape','cache_shape','fresh_dtype','cache_dtype','fresh_type','cache_type','elements','unequal_count',
        'max_absolute_difference','mean_absolute_difference','field_bitwise_equal','planning_disabled_all_models'}
    require(isinstance(audit,dict) and set(audit) == fields and audit['path'] == 'inputs.plan_dict.sample_traj' and
            audit['comparison_only_projection'] is True and audit['actual_inputs_modified'] is False,
            'Exception must name only the inactive sample_traj and preserve actual inputs')
    for key in fields:
        if key.endswith('_sha256'):
            require(is_sha(audit[key]), 'Invalid boundary digest: '+key)
    require(audit['fresh_full_input_sha256'] == result['input_tree_sha256'] and
            audit['fresh_projected_input_sha256'] == audit['cache_projected_input_sha256'],
            'Fresh full input or active projection digest disagrees')
    require(audit['fresh_shape'] == audit['cache_shape'] == [1,1800,5,3] and
            audit['fresh_dtype'] == audit['cache_dtype'] == 'float64' and
            audit['fresh_type'] == audit['cache_type'] and audit['fresh_type'] in ('numpy.ndarray','torch.Tensor'),
            'Inactive field exact type/dtype/shape contract differs')
    full_equal = audit['fresh_full_input_sha256'] == audit['cache_full_input_sha256']
    field_equal = audit['fresh_field_sha256'] == audit['cache_field_sha256']
    require(type(parity['full_boundary_hash_equal']) is bool and parity['full_boundary_hash_equal'] is full_equal and
            type(audit['field_bitwise_equal']) is bool and audit['field_bitwise_equal'] is field_equal and
            full_equal is field_equal, 'Full/field equality is not reported truthfully')
    require(type(audit['elements']) is int and audit['elements'] == 27000 and
            type(audit['unequal_count']) is int and 0 <= audit['unequal_count'] <= audit['elements'],
            'Invalid inactive-field difference count')
    maximum, mean = audit['max_absolute_difference'],audit['mean_absolute_difference']
    require(type(maximum) in (int,float) and type(mean) in (int,float) and math.isfinite(maximum) and
            math.isfinite(mean) and 0 <= mean <= maximum and
            ((audit['unequal_count'] == 0 and maximum == mean == 0) or
             (audit['unequal_count'] > 0 and maximum > 0)), 'Invalid inactive-field numeric difference statistics')
    # Signed-zero differences may be byte-unequal with zero numeric difference.
    require(not field_equal or (audit['unequal_count'] == 0 and maximum == mean == 0),
            'Byte-equal field cannot have nonzero numeric differences')
    validate_planning(audit['planning_disabled_all_models'],models)


def validate_policy_evidence(evaluation, evaluation_path, producer):
    """Bind the real rejected v1 observation; do not retroactively cover cache sources."""
    policy = evaluation['inactive_boundary_policy']; evidence = policy['diagnostic']
    require(set(evidence) == {'directory','difference_sha256','complete_sha256'} and
            evidence['difference_sha256'] == DIFFERENCE_SHA and evidence['complete_sha256'] == DIAGNOSTIC_COMPLETE_SHA,
            'Different failure receipt cannot authorize this exception')
    require(policy['source_function_ast_sha256'] == producer.CONSUMER_AST_SHA == CONSUMER_AST_SHA and
            policy['previous_cache_source_coverage_claim'] is False and
            evaluation['source_sha256']['native_boundary_diff.py'] == DIAGNOSTIC_SOURCE_SHA and
            evaluation['source_sha256']['objective_joint_evaluation.py'] == V1_PRODUCER_SHA and
            evaluation['source_sha256']['objective_joint_merge.py'] == V1_MERGER_SHA,
            'Original consumer/diagnostic/v1 source binding differs')
    directory = resolve(evidence['directory'],evaluation_path)
    difference, _ = read(directory/'difference.json',DIFFERENCE_SHA)
    complete, _ = read(directory/'complete.json',DIAGNOSTIC_COMPLETE_SHA)
    require(complete['difference_sha256'] == DIFFERENCE_SHA and
            difference['boundary_equal'] is False and len(difference['differences']) == 1 and
            difference['fresh_boundary_sha256'] != difference['cached_boundary_sha256'],
            'Original rejected boundary mismatch was not retained')
    return dict(difference_sha256=DIFFERENCE_SHA,complete_sha256=DIAGNOSTIC_COMPLETE_SHA,
        diagnostic_source_sha256=DIAGNOSTIC_SOURCE_SHA,diagnostic_evaluation_protocol_sha256=difference['evaluation_protocol_sha256'],
        diagnostic_sample_token=difference['identity']['sample_token'],source_function_ast_sha256=CONSUMER_AST_SHA,
        additional_runtime_source_sha256=policy['sources'],previous_cache_source_coverage_claim=False,
        independent_consumer_runtime_assertion='producer checks original source/AST, sole guarded sample_traj read, and all models planning=False/eval',
        merger_executed_neural_or_runtime_source_inspection=False)


def validate_sample(result, identity, contract_sha, mode, models):
    require(result['schema'] == 'objective-joint-sample-v2' and result['contract_sha256'] == contract_sha and
            result['optimizer_steps'] == 0 and result['identity'] == identity, 'Sample schema/identity/contract differs')
    require(set(result['models']) == set(result['logits_array_sha256']) == set(models), 'Missing or extra sample model')
    for key in ('input_tree_sha256','targets_array_sha256','observed_bev_tensor_sha256','native_radar_tensor_sha256'):
        require(is_sha(result[key]), 'Missing typed source hash: '+key)
    require(all(is_sha(value) for value in result['logits_array_sha256'].values()), 'Missing full prediction array hash')
    counts = integer_array(result['target_counts_0_1_255_by_frame'],(7,3),'GT')
    require(np.all(counts.sum(1) == VOXELS), 'GT counts do not exhaust the original seven grids')
    histograms = {}
    for name in models:
        row = result['models'][name]
        require(row['sample_token'] == identity['sample_token'] and row['scene_token'] == identity['scene_token'] and
                row['horizon_seconds'] == HORIZONS, 'Model identity/horizon differs: '+name)
        hist = integer_array(row['hist_by_horizon'],(5,2,2),'confusion '+name)
        require(np.array_equal(hist.sum(2),counts[2:,:2]), 'Original GT rows/prediction columns differ: '+name)
        histograms[name] = hist
    require(set(result['model_seconds']) == set(models) and
            all(math.isfinite(value) and value >= 0 for value in result['model_seconds'].values()), 'Invalid per-model timing')
    require(math.isfinite(result['total_seconds']) and result['total_seconds'] >= 0 and
            type(result['peak_cuda_allocated_bytes']) is int and result['peak_cuda_allocated_bytes'] >= 0, 'Invalid sample resources')
    if mode == 'pilot':
        flags = ('active_boundary_hash_equal','exact_gt_equal','native_all_5h_3layer_logits_bitwise',
                 'all_models_5_horizons_full_GT_confusion_equal')
        require(isinstance(result['engineering_parity'],dict) and
                all(result['engineering_parity'].get(key) is True for key in flags), 'Missing/failed engineering parity')
        validate_boundary_parity(result,models)
    else:
        require(result['engineering_parity'] is None, 'Full must use the fresh native loader, not pilot-cache scoring')
    return histograms, counts


def resolve(path, protocol_path):
    value = Path(path)
    return value if value.is_absolute() else Path(protocol_path).resolve().parent/value


def development_contract(evaluation, evaluation_path, objective):
    """Recompute all registered promotions; never choose the strongest survivor."""
    binding = evaluation['development_summary']; directory = resolve(binding['directory'],evaluation_path)
    require(not (directory/'failed.json').exists(), 'Development aggregation has a failure receipt')
    done, _ = read(directory/'complete.json',binding['complete_sha256'])
    summary, _ = read(directory/'summary.json',binding['summary_sha256'])
    require(done['schema'] == 'm0-objective-supervision-aggregate-complete-v1' and done['status'] == 'COMPLETE' and
            done['summary_sha256'] == binding['summary_sha256'] == done['files_sha256']['summary.json'], 'Broken development hash chain')
    require(set(done['files_sha256']) == {'summary.json','report.md','metrics.csv','contrasts.csv'}, 'Incomplete development artifact set')
    for filename, expected in done['files_sha256'].items():
        require(sha(directory/filename) == expected, 'Changed development artifact: '+filename)
    require(summary['schema'] == 'm0-objective-supervision-aggregate-v1' and
            summary['status'] == 'COMPLETED_DEVELOPMENT_COMPARISON' and
            summary['objective_protocol_sha256'] == done['objective_protocol_sha256'] == OBJECTIVE_SHA and
            summary['parent_protocol_sha256'] == PARENT_SHA and summary['script_sha256'] == OBJECTIVE_AGGREGATE_SHA,
            'Development provenance differs')
    dev_models = ['M0','M0_fp32','native1','F','O']
    require(done['models'] == dev_models and set(summary['models']) == set(dev_models) and
            summary['samples'] == done['samples'] == 200 and summary['scenes'] == done['scenes'] == 100,
            'All five models must remain disclosed on complete dev200')
    require(summary['audit']['status'] == 'PASS' and
            summary['audit']['actual_three_final_checkpoints_rehashed_and_payload_checked_on_CPU'] is True and
            summary['audit']['actual_512_updates_2048_examples_LR_and_orders_checked'] is True and
            summary['audit']['all_five_models_same_full_GT_row_counts'] is True, 'Development final checkpoint/order/GT audit missing')
    require(summary['registered_minimum_meaningful_effect_pp'] == objective['evaluation']['minimum_meaningful_effect_pp'] == .5,
            'Promotion requires the pre-existing +0.5 pp minimum')
    expected_primary = [(arm,base) for arm in ('F','O') for base in ('M0_fp32','native1')]
    require(summary['registered_primary_contrasts'] == [list(pair) for pair in expected_primary], 'Wrong development primary contrasts')
    rows = summary['comparisons']; pairs = [(row['left'],row['right']) for row in rows]
    require(pairs == expected_primary+[('native1','M0_fp32'),('M0_fp32','M0')], 'Development comparison set/order differs')
    comparisons = {pair:row for pair,row in zip(pairs,rows)}; qualification = {}; candidates = []
    for arm in ('F','O'):
        tests = []
        for base in ('M0_fp32','native1'):
            row = comparisons[(arm,base)]; metric = row['metrics']['future_macro_gmo']
            difference = (summary['models'][arm]['metrics']['future_macro_gmo']['estimate']-
                          summary['models'][base]['metrics']['future_macro_gmo']['estimate'])
            require(row['primary_contrast'] is True and math.isfinite(difference) and
                    math.isclose(100*difference,metric['delta_pp'],rel_tol=1e-10,abs_tol=1e-12) and
                    math.isclose(difference,metric['delta_ratio'],rel_tol=1e-10,abs_tol=1e-12), 'Development effect arithmetic differs')
            ci = metric['ci95_pp']; ratios = metric['ci95_ratio']
            require(len(ci) == len(ratios) == 2 and all(math.isfinite(value) for value in ci+ratios) and ci[0] <= ci[1] and
                    all(math.isclose(value,100*ratio,rel_tol=1e-10,abs_tol=1e-12) for value,ratio in zip(ci,ratios)),
                    'Invalid development paired interval')
            tests.append(dict(against=base,delta_pp=metric['delta_pp'],ci95_pp=ci,
                point_at_least_half_pp=metric['delta_pp'] >= .5,ci_lower_positive=ci[0] > 0))
        passed = all(row['point_at_least_half_pp'] and row['ci_lower_positive'] for row in tests)
        qualification[arm] = dict(qualifies=passed,comparisons=tests)
        if passed:
            candidates.append(arm)
    require(candidates and evaluation['candidate_names'] == candidates,
            'Require all and only ordered F/O candidates passing both registered +0.5 pp and positive-CI gates')
    arms = ['native1']+candidates
    require(set(evaluation['final_sources']) == set(arms), 'Frozen final source/model set differs')
    for arm in arms:
        receipt = summary['source_receipts'][arm]; final = evaluation['final_sources'][arm]
        files = receipt['reuse_audit']['files_sha256'] if arm == 'native1' else receipt['files_sha256']
        require(final['files_sha256'] == files and
                final['final_head_state_sha256'] == receipt['checkpoint']['final_head_state_sha256'] and
                final['files_sha256']['latest.pth'] == receipt['checkpoint']['checkpoint_sha256'],
                'Evaluation model differs from the dev-selected final tensor/checkpoint: '+arm)
        require(Path(final['directory']).is_absolute(), 'Final source directory must be frozen as an absolute path')
    disclosure = dict(summary_sha256=binding['summary_sha256'],complete_sha256=binding['complete_sha256'],
        directory=str(directory.resolve()),samples=200,scenes=100,all_five_models_disclosed=True,
        models=summary['models'],comparisons=summary['comparisons'],qualification=qualification,candidate_names=candidates,
        selection_rule='For each F/O: both versus M0_fp32 and native1 have point >= 0.5 pp and 95% CI lower > 0',
        historical_validation_exposure=True,multiple_comparison_adjustment=False)
    return summary, disclosure


def load_protocols(a):
    parent, _ = read(a.protocol,PARENT_SHA); objective, _ = read(a.objective_protocol,OBJECTIVE_SHA)
    evaluation, evaluation_sha = read(a.evaluation_protocol)
    require(evaluation['schema'] == 'objective-joint-evaluation-protocol-v2' and
            evaluation['status'] == 'FROZEN_BEFORE_OBJECTIVE_JOINT_PILOT' and
            evaluation['parent_protocol_sha256'] == PARENT_SHA and
            evaluation['objective_train_protocol_sha256'] == OBJECTIVE_SHA, 'Wrong frozen evaluation protocol')
    require(evaluation['nomination_rule'] == NOMINATION_RULE, 'Nomination rule differs from preregistered development gate')
    for key in ('training','numerical_policy','m0_sha256','config_sha256','selection_sha256'):
        require(objective[key] == parent[key], 'Parent/objective training contract differs: '+key)
    require(parent['m0_sha256'] == M0_SHA and parent['training']['seed'] == 11, 'Wrong original model/training seed')
    for key in ('config_sha256','m0_sha256','numerical_policy'):
        require(evaluation[key] == parent[key], 'Joint source/precision differs: '+key)
    require(evaluation['runtime_source_sha256'] == objective['runtime_source_sha256'] and
            evaluation['historical_validation_exposure'] is True and evaluation['new_blind_test'] is False and
            evaluation['multiple_comparison_adjustment'] is False and evaluation['training_seeds_added'] == 0,
            'Joint runtime/scientific disclosure differs')
    for mode in ('pilot','full'):
        cap = evaluation['resources'][mode]
        require(cap['shard_count'] == 2 and type(cap['max_seconds']) is int and 0 < cap['max_seconds'] <= 43200 and
                math.isfinite(cap['max_allocated_gib']) and cap['max_allocated_gib'] > 0, 'Invalid frozen resource ceiling')
    require(evaluation['source_sha256']['objective_joint_merge_v2.py'] == sha(__file__), 'Merger revision differs from protocol')
    source_union = {}
    for sources in (parent['source_sha256'],objective['source_sha256'],evaluation['source_sha256']):
        for filename, expected in sources.items():
            require(filename not in source_union or source_union[filename] == expected, 'Conflicting source binding: '+filename)
            require(Path(filename).name == filename and sha(PKG/filename) == expected, 'Frozen package source differs: '+filename)
            source_union[filename] = expected
    require(objective['source_sha256']['objective_supervision_aggregate.py'] == OBJECTIVE_AGGREGATE_SHA,
            'Wrong frozen parameterized metric implementation')
    dev, disclosure = development_contract(evaluation,a.evaluation_protocol,objective)
    names = tuple(['M0','M0_fp32','native1']+evaluation['candidate_names'])
    primary = tuple((arm,base) for arm in evaluation['candidate_names'] for base in ('M0_fp32','native1'))
    selection_binding = evaluation['selections'][a.mode]
    require(selection_binding['sha256'] == SELECTION_SHA[a.mode], 'Selection digest differs from frozen evaluation protocol')
    selection, _ = read(a.selection,SELECTION_SHA[a.mode]); rows = validate_selection(selection,a.mode)
    require(selection['config_sha256'] == parent['config_sha256'], 'Selection/config identity differs')
    producer = frozen_module('objective_joint_evaluation_v2.py',evaluation['source_sha256']['objective_joint_evaluation_v2.py'])
    base = frozen_module('joint_native_evaluation.py',evaluation['source_sha256']['joint_native_evaluation.py'])
    producer.validate_inactive_policy(evaluation,a.evaluation_protocol,verify_runtime=False)
    policy_evidence = validate_policy_evidence(evaluation,a.evaluation_protocol,producer)
    _, expected_recipe = producer.derive_run(base,dict(base.__dict__))
    require(len(expected_recipe['edits']) == 9 and all(v == 1 for v in expected_recipe['edits'].values()) and
            expected_recipe['inverse_ast_restores_original_run'] is True, 'V2 must retain the nine audited edits and inverse-AST gate')
    selected, nomination_evidence = producer.nomination(dev,evaluation['nomination_rule'])
    require(selected == evaluation['candidate_names'], 'Independent nomination implementation disagrees')
    return dict(parent=parent,objective=objective,evaluation=evaluation,evaluation_sha=evaluation_sha,
        selection=selection,rows=rows,models=names,arms=names[2:],primary=primary,
        contrasts=primary+(('native1','M0_fp32'),('M0_fp32','M0')),development=dev,development_disclosure=disclosure,
        expected_recipe=expected_recipe,nomination_evidence=nomination_evidence,policy_evidence=policy_evidence)


def validate_arm_receipts(receipts, context, contract):
    evaluation, parent, objective = [context[key] for key in ('evaluation','parent','objective')]
    require(set(receipts) == set(context['arms']), 'Wrong fixed trained-arm receipt set')
    scopes = []; migrations = []
    for arm in context['arms']:
        receipt = receipts[arm]; final = evaluation['final_sources'][arm]; manifest = receipt['manifest']
        require(receipt['directory'] == final['directory'] and receipt['files_sha256'] == final['files_sha256'] and
                all(is_sha(value) for value in receipt['files_sha256'].values()), 'Source artifact ledger differs: '+arm)
        require(receipt['files_sha256']['latest.pth'] == contract['final_checkpoints_sha256'][arm] and
                final['final_head_state_sha256'] == contract['final_head_state_sha256'][arm], 'Wrong final checkpoint/head binding: '+arm)
        require(receipt['final_head_state_sha256'] == final['final_head_state_sha256'] and
                receipt['checkpoint_audit'] == context['development']['source_receipts'][arm]['checkpoint'],
                'Producer actual payload audit differs from the fixed development checkpoint: '+arm)
        require(manifest['arm'] == arm and manifest['seed'] == 11 and manifest['m0_sha256'] == M0_SHA and
                manifest['initial_head_sha256'] == INITIAL_HEAD_SHA and manifest['migration']['mode'] == 'native1' and
                manifest['numerical_policy'] == parent['numerical_policy'] and manifest['passes'] == 4 and
                manifest['examples_per_pass'] == 512 and manifest['trainable_parameters'] == 13274016,
                'Wrong final-model initialization, precision, architecture or budget: '+arm)
        require(manifest['train_index_sha256'] == objective['cache_index_sha256']['train'] and
                manifest['dev_index_sha256'] == objective['cache_index_sha256']['development'], 'Training/development cache differs: '+arm)
        if arm == 'native1':
            require(manifest['protocol_sha256'] == PARENT_SHA and
                    set(manifest['source_sha256']) == {'memory_experiment.py','native_state_cache.py','observation_memory.py'} and
                    all(parent['source_sha256'].get(key) == value for key,value in manifest['source_sha256'].items()),
                    'Control trained source/protocol differs')
        else:
            require(manifest['protocol_sha256'] == OBJECTIVE_SHA and manifest['parent_protocol_sha256'] == PARENT_SHA and
                    manifest['source_sha256'] == objective['source_sha256'] and manifest['parent_source_sha256'] == parent['source_sha256'] and
                    manifest['runtime_source_sha256'] == objective['runtime_source_sha256'], 'Candidate training source/protocol differs: '+arm)
            require(manifest['objective_adapter']['arm'] == arm and
                    manifest['objective_adapter']['adapter_sha256'] == objective['source_sha256']['objective_supervision_adapters.py'] and
                    manifest['objective_adapter']['native_memory_mode'] == 'native1', 'Candidate trained objective identity differs')
        scopes.append(manifest['trainable_parameter_names']); migrations.append(manifest['migration'])
    require(all(scope == scopes[0] for scope in scopes) and all(migration == migrations[0] for migration in migrations),
            'Selected heads must share exact parameter order/capacity and native1 migration')
    native = receipts['native1']['manifest']; reference = native['frozen_precision_reference']
    require(reference['initial_head_sha256'] == INITIAL_HEAD_SHA and reference['source_m0_sha256'] == M0_SHA and
            reference['samples'] == 200 and reference['optimizer_updates'] == 0 and
            reference['matmul_tf32'] is False and reference['cudnn_tf32'] is True and
            reference['sha256'] == receipts['native1']['files_sha256']['frozen_native_fp32_records.jsonl'],
            'Frozen precision reference differs')


def validate_loaded_models(loaded, receipts, context, contract):
    require(loaded['training'] is False and loaded['optimizer_created'] is False, 'Joint evaluation unexpectedly trained')
    heads = loaded['head_state_sha256']; require(set(heads) == set(context['models']) and
        all(is_sha(value) for value in heads.values()), 'Incomplete loaded head digest set')
    require(heads['M0'] == heads['M0_fp32'] == INITIAL_HEAD_SHA, 'Frozen M0 initialization changed')
    for arm in context['arms']:
        require(heads[arm] == contract['final_head_state_sha256'][arm], 'Actual loaded candidate tensor differs: '+arm)
    source = loaded['source']; parent = context['parent']
    require(source['checkpoint_sha256'] == M0_SHA and source['source_model_sha256'] == SOURCE_MODEL_SHA and
            source['config_sha256'] == parent['config_sha256'] and source['checkpoint_epoch'] == 24 and
            source['radar_contract'] == 'native_loader_B_without_common_source_override' and
            source['model_only'] is True and source['optimizer_loaded'] is False, 'Loaded native model/input provenance differs')
    require(set(loaded['migrations']) == set(context['models'][1:]), 'Missing replay migration')
    for name in context['models'][1:]:
        require(loaded['migrations'][name] == receipts['native1' if name == 'M0_fp32' else name]['manifest']['migration'],
                'Evaluation must preserve the trained original single-slot graph: '+name)


def full_authorization(a, context):
    if a.mode == 'pilot':
        require(a.full_authorization is None, 'Pilot must not consume a full-run authorization')
        return None, None, None
    require(a.full_authorization, 'Full evaluation requires separate post-pilot authorization')
    authorization, authorization_sha = read(a.full_authorization)
    require(authorization['schema'] == 'objective-joint-full-authorization-v2' and
            authorization['status'] == 'AUTHORIZED_AFTER_PILOT_PASS' and
            authorization['evaluation_protocol_sha256'] == context['evaluation_sha'], 'Wrong full authorization')
    cap, ceiling = authorization['resources'],context['evaluation']['resources']['full']
    require(set(cap) == {'max_seconds','max_allocated_gib','shard_count'} and cap['shard_count'] == 2 and
            type(cap['max_seconds']) is int and 0 < cap['max_seconds'] <= ceiling['max_seconds'] and
            math.isfinite(cap['max_allocated_gib']) and 0 < cap['max_allocated_gib'] <= ceiling['max_allocated_gib'],
            'Actual post-pilot full resource budget exceeds the pre-frozen ceiling')
    binding = authorization['pilot_merge']; directory = resolve(binding['directory'],a.full_authorization)
    require(not (directory/'failed.json').exists(), 'Pilot merge has a failure receipt')
    complete, _ = read(directory/'complete.json',binding['complete_sha256'])
    summary, _ = read(directory/'summary.json',binding['summary_sha256'])
    require(complete['schema'] == 'objective-joint-merge-complete-v2' and complete['status'] == 'COMPLETE' and complete['mode'] == 'pilot' and complete['samples'] == 2 and
            complete['summary_sha256'] == binding['summary_sha256'] == complete['files_sha256']['summary.json'], 'Pilot completion hash chain differs')
    require(set(complete['files_sha256']) == {'summary.json','report.md','sample_hash_ledger.json','development_disclosure.json'},
            'Pilot merge artifact set differs')
    for filename, expected in complete['files_sha256'].items():
        require(sha(directory/filename) == expected, 'Pilot merge artifact changed: '+filename)
    require(summary['schema'] == 'objective-joint-pilot-merge-v2' and summary['status'] == 'PILOT_ENGINEERING_PARITY_RESOURCE_PASS' and
            summary['candidate_names'] == context['evaluation']['candidate_names'] and summary['model_names'] == list(context['models']) and
            summary['samples'] == summary['scenes'] == 2 and summary['performance_metrics_computed'] is False and
            summary['performance_success_claim'] is False and summary['merger_sha256'] == sha(__file__), 'Pilot is not the fixed engineering proof')
    require(summary['inactive_boundary_policy'] == context['evaluation']['inactive_boundary_policy'] and
            summary['parity']['all_active_boundaries_exact'] is True and
            summary['parity']['full_boundary_equality_not_required'] is True, 'Pilot exception scope was not explicitly retained')
    pilot_ledger, _ = read(directory/'sample_hash_ledger.json',complete['files_sha256']['sample_hash_ledger.json'])
    require(len(pilot_ledger) == 2, 'Pilot boundary ledger must contain two anchors')
    for row in pilot_ledger:
        validate_boundary_parity(row,context['models'])
    require(summary['parity']['full_boundary_hash_equal_by_sample'] ==
            {row['identity']['sample_token']:row['engineering_parity']['full_boundary_hash_equal'] for row in pilot_ledger},
            'Pilot full-boundary equality flags disagree with the per-sample ledger')
    sources = summary['sources']
    require(sources['evaluation_protocol_sha256'] == context['evaluation_sha'] and sources['parent_protocol_sha256'] == PARENT_SHA and
            sources['objective_train_protocol_sha256'] == OBJECTIVE_SHA and sources['selection_sha256'] == SELECTION_SHA['pilot'] and
            sources['final_checkpoints_sha256'] == {arm:context['evaluation']['final_sources'][arm]['files_sha256']['latest.pth'] for arm in context['arms']} and
            sources['final_head_state_sha256'] == {arm:context['evaluation']['final_sources'][arm]['final_head_state_sha256'] for arm in context['arms']},
            'Pilot verified different sources/checkpoints')
    gate = dict(directory=str(directory.resolve()),complete_sha256=binding['complete_sha256'],summary_sha256=binding['summary_sha256'])
    return authorization, authorization_sha, gate


def resource_summary(ordered, shards, models):
    times = np.asarray([row['result']['total_seconds'] for row in ordered],dtype=float)
    return dict(samples=len(ordered),shards=len(shards),
        sum_shard_process_elapsed_seconds=sum(row['seconds'] for row in shards),
        maximum_shard_elapsed_seconds=max(row['seconds'] for row in shards),
        measured_sample_seconds=dict(minimum=float(times.min()),mean=float(times.mean()),maximum=float(times.max())),
        model_mean_seconds={name:float(np.mean([row['result']['model_seconds'][name] for row in ordered])) for name in models},
        maximum_observed_cuda_allocation_bytes=max(row['result']['peak_cuda_allocated_bytes'] for row in ordered),
        accounting='producer process residence and measured sample timings, not outer job-runner residence or GPU kernel busy time',
        pilot_cost_is_not_full_run_guarantee=True)


def full_outputs(out, summary, ordered, names):
    arrays = np.stack([np.stack([row['hist'][name] for row in ordered]) for name in names]).astype(np.uint64)
    np.savez_compressed(out/'raw_confusions.npz',models=np.asarray(names),hist_by_model_sample_horizon=arrays,
        target_counts_0_1_255_by_sample_frame=np.stack([row['counts'] for row in ordered]).astype(np.uint64),
        sample_tokens=np.asarray([row['identity']['sample_token'] for row in ordered]),
        scene_tokens=np.asarray([row['identity']['scene_token'] for row in ordered]),
        official_indices=np.asarray([row['identity']['official_index'] for row in ordered],dtype=np.int64),
        horizon_seconds=np.asarray(HORIZONS))
    with (out/'metrics.csv').open('x',newline='') as stream:
        fields = ['model','horizon_seconds','GMO_ratio','GMO_percent','binary_mIoU_ratio','binary_mIoU_percent','TN','FP','FN','TP']
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for name in names:
            for row in summary['models'][name]['horizons']:
                writer.writerow(dict(model=name,horizon_seconds=row['horizon_seconds'],GMO_ratio=row['gmo_iou'],
                    GMO_percent=100*row['gmo_iou'],binary_mIoU_ratio=row['binary_miou'],binary_mIoU_percent=100*row['binary_miou'],
                    **{key:row[key] for key in ('TN','FP','FN','TP')}))
    with (out/'contrasts.csv').open('x',newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['contrast','primary','metric','delta_ratio','delta_pp','CI95_lower_pp','CI95_upper_pp'])
        for comparison in summary['comparisons']:
            for key,row in comparison['metrics'].items():
                writer.writerow([comparison['contrast'],comparison['primary_contrast'],key,row['delta_ratio'],row['delta_pp'],*row['ci95_pp']])
    lines = ['# 冻结晋级模型的完整原生共同验证','',
        '覆盖全部 5,119 anchors / 150 scenes，按原始 official index 排序。同场景使用自然数量的全部 anchors。模型集合在读取 full 结果前冻结：M0、M0_fp32、C/native1，以及全部通过预登记开发门槛的 F/O 候选。未晋级候选不做 full，其五模型开发集结果完整保存在 development_disclosure.json；不得将未评价记作零分。','',
        '所有模型使用同一原生图像/雷达观测边界的独立副本，以及原始未来 ego/action；无 future GT 输入、目标适配器或坐标干预。单训练 seed11、固定最后 512 次更新；本阶段没有新增训练或选择 checkpoint。','',
        '主指标为每个未来时域先汇总全部样本混淆矩阵，再算 GMO foreground IoU，最后对四未来时域取均值。GMO 是可移动语义类，不是真实运动标注；binary mIoU 是 GMO/non-GMO 两类均值。Pooled 副指标先合并四未来 confusion，不能冒充主指标。','',
        '| 模型 | 0.5s GMO % | 1s % | 1.5s % | 2s % | Future macro GMO % [95% CI] | Future pooled GMO % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        model = summary['models'][name]; hs = [100*row['gmo_iou'] for row in model['horizons'][1:]]
        macro = model['metrics']['future_macro_gmo']; pooled = model['metrics']['future_pooled_gmo']['estimate']
        lines.append(f"| {name} | {hs[0]:.4f} | {hs[1]:.4f} | {hs[2]:.4f} | {hs[3]:.4f} | {100*macro['estimate']:.4f} [{100*macro['ci95'][0]:.4f}, {100*macro['ci95'][1]:.4f}] | {100*pooled:.4f} |")
    lines += ['', '| 配对比较 | 主比较 | Future macro GMO Δ pp | 95% CI pp |', '|---|---|---:|---:|']
    for comparison in summary['comparisons']:
        metric = comparison['metrics']['future_macro_gmo']
        lines.append(f"| {comparison['contrast']} | {'是' if comparison['primary_contrast'] else '否'} | {metric['delta_pp']:+.4f} | [{metric['ci95_pp'][0]:+.4f}, {metric['ci95_pp'][1]:+.4f}] |")
    lines += ['', '区间：150 个场景、10,000 次 seed11 共同配对重采样；每场景全部 anchors 一起重采样，再按原 pooled-confusion 指标计算。逐项双侧 percentile 95% CI 未作多重比较校正，不涵盖神经网络训练随机性或开发选择不确定性。完整验证仍包含已暴露的开发和历史验证场景，不是新盲测。',
        'M0 保留原生 TF32 预测；M0_fp32 与所有训练 head 关闭未来 matmul TF32，观测编码和 cuDNN TF32 不变。没有阈值拟合或 R4 校准。F/O 训练干预没有合并臂，不能识别交互。',
        '本工具不自动宣布总体研究目标完成。raw_confusions.npz 保留全部整数 confusion，轴为模型、样本、时域、GT 行、预测列；sample_hash_ledger.json 绑定每条原生输入和标签摘要及 source JSON。','']
    (out/'report.md').write_text('\n'.join(lines))


def collect(a, context):
    root = Path(a.evaluation_root).resolve(); evaluation, parent, objective = [context[key] for key in ('evaluation','parent','objective')]
    names, rows = context['models'],context['rows']; authorization, auth_sha, pilot_gate = full_authorization(a,context)
    require(not (root/'failed.json').exists(), 'Evaluation root has a failure receipt')
    contract, contract_sha = read(root/'contract.json')
    require(contract['schema'] == 'objective-joint-evaluation-contract-v2' and contract['mode'] == a.mode and
            contract['models'] == list(names) and contract['horizons'] == HORIZONS, 'Common model/scope contract differs')
    require(contract['script_sha256'] == evaluation['source_sha256']['objective_joint_evaluation_v2.py'] and
            contract['evaluation_protocol_sha256'] == context['evaluation_sha'] and
            contract['objective_train_protocol_sha256'] == OBJECTIVE_SHA and contract['parent_protocol_sha256'] == PARENT_SHA and
            contract['selection_sha256'] == SELECTION_SHA[a.mode] and contract['m0_sha256'] == M0_SHA,
            'Producer/protocol/selection/checkpoint contract differs')
    require(contract['config_sha256'] == parent['config_sha256'] and contract['numerical_policy'] == parent['numerical_policy'] and
            contract['source_sha256'] == evaluation['source_sha256'] and contract['parent_source_sha256'] == parent['source_sha256'] and
            contract['objective_source_sha256'] == objective['source_sha256'] and
            contract['runtime_source_sha256'] == objective['runtime_source_sha256'], 'Common source/config/precision differs')
    require(contract['optimizer_steps'] == 0 and contract['historical_validation_exposure'] is True and
            contract['sample_partition'] == 'selection_ordinal modulo shard_count' and contract['selected_samples'] == len(rows) and
            contract['shard_count'] == 2, 'Scientific scope/partition differs')
    require(contract['full_authorization_sha256'] == auth_sha and contract['pilot_gate'] == pilot_gate,
            'Full authorization/pilot gate differs')
    expected_checkpoints = {arm:evaluation['final_sources'][arm]['files_sha256']['latest.pth'] for arm in context['arms']}
    expected_heads = {arm:evaluation['final_sources'][arm]['final_head_state_sha256'] for arm in context['arms']}
    require(contract['final_checkpoints_sha256'] == expected_checkpoints and contract['final_head_state_sha256'] == expected_heads,
            'Final checkpoint/head contract differs from development selection')
    require(all(is_sha(value) for value in expected_checkpoints.values()) and all(is_sha(value) for value in expected_heads.values()),
            'Malformed final weight hash')
    require(contract['inactive_boundary_policy'] == evaluation['inactive_boundary_policy'], 'Inactive exception policy changed in producer')
    require(contract['data_recipe_receipt'] == context['expected_recipe'], 'Native data recipe AST receipt differs from the frozen source')
    require(contract['candidate_names'] == evaluation['candidate_names'] and
            contract['development_summary_sha256'] == evaluation['development_summary']['summary_sha256'] and
            contract['nomination_evidence'] == context['nomination_evidence'], 'Common development nomination differs')
    cap, ceiling = contract['resources'],evaluation['resources'][a.mode]
    require(set(cap) == {'max_seconds','max_allocated_gib','shard_count'} and cap['shard_count'] == 2 and
            type(cap['max_seconds']) is int and 0 < cap['max_seconds'] <= ceiling['max_seconds'] and
            math.isfinite(cap['max_allocated_gib']) and 0 < cap['max_allocated_gib'] <= ceiling['max_allocated_gib'] and
            (a.mode == 'pilot' or cap == authorization['resources']), 'Actual resource limits differ from the frozen ceiling/authorization')
    require(contract['allocator_cap_bytes'] == int(cap['max_allocated_gib']*2**30), 'Common allocator cap differs from actual authorized resources')
    expected_shards = {'shard_00_of_02','shard_01_of_02'}
    require({path.name for path in root.glob('shard_*') if path.is_dir()} == expected_shards, 'Missing or extra shard directory')
    require((root/'samples').is_dir() and {path.name for path in (root/'samples').iterdir()} == {row['sample_token'] for row in rows},
            'Sample directory set is incomplete or contains undeclared extras')
    joined = {}; shards = []; base_receipts = base_loaded = base_pilot_sources = None
    for shard_index in range(2):
        directory = root/('shard_%02d_of_02'%shard_index)
        require(not (directory/'failed.json').exists(), 'Refuse failed/incomplete shard')
        done, done_sha = read(directory/'complete.json')
        require(done['schema'] == 'objective-joint-shard-complete-v2' and done['status'] == 'COMPLETE_JOINT_NATIVE_EVALUATION_SHARD' and done['mode'] == a.mode and
                done['shard_index'] == shard_index and done['shard_count'] == 2 and done['contract_sha256'] == contract_sha,
                'Shard completion scope/hash differs')
        validate_planning(done['planning_disabled_all_models'],names)
        index,index_sha = read(directory/'index.json',done['index_sha256'])
        manifest,manifest_sha = read(directory/'manifest.json',done['manifest_sha256'])
        loaded,loaded_sha = read(directory/'loaded_models.json',done['loaded_models_sha256'])
        require(index['status'] == 'COMPLETE' and done['optimizer_steps'] == 0 and done['threshold_selection'] is False and
                done['all_models_same_inputs'] is True and done['all_pilot_parity_passed'] is (a.mode == 'pilot'), 'Shard engineering scope differs')
        require(all(manifest.get(key) == value for key,value in contract.items()) and manifest['shard_index'] == shard_index,
                'Shard manifest differs from common contract')
        require(type(manifest['max_seconds']) is int and manifest['max_seconds'] == cap['max_seconds'],
                'Shard budget differs from common actual resources')
        expected = [dict(row,selection_ordinal=i) for i,row in enumerate(rows) if i%2 == shard_index]
        require(manifest['selected_records'] == expected and done['samples'] == len(index['records']) == len(expected) and
                done['official_indices'] == [row['official_index'] for row in expected], 'Shard identity/order/exposure differs')
        validate_arm_receipts(manifest['arm_source_receipts'],context,contract)
        validate_loaded_models(loaded,manifest['arm_source_receipts'],context,contract)
        signature = {key:loaded[key] for key in ('head_state_sha256','source','torch_version','cuda_version','gpu_name','migrations')}
        if base_receipts is None:
            base_receipts,base_loaded = manifest['arm_source_receipts'],signature
        else:
            require(manifest['arm_source_receipts'] == base_receipts and signature == base_loaded,
                    'Actual source weights/migrations/runtime differ across shards')
        if a.mode == 'pilot':
            pilot_sources,pilot_sources_sha = read(directory/'pilot_reference_sources.json',done['pilot_reference_sources_sha256'])
            require(len(pilot_sources) == len(context['arms'])+3 and all(is_sha(value) for value in pilot_sources.values()),
                    'Incomplete pilot reference provenance ledger')
            paths = set()
            for arm in context['arms']:
                receipt = manifest['arm_source_receipts'][arm]; path = str(Path(receipt['directory'])/'development_records.jsonl'); paths.add(path)
                require(pilot_sources.get(path) == receipt['files_sha256']['development_records.jsonl'], 'Pilot uses a different candidate dev reference')
            native = manifest['arm_source_receipts']['native1']; path = str(Path(native['directory'])/'frozen_native_fp32_records.jsonl'); paths.add(path)
            require(pilot_sources.get(path) == native['files_sha256']['frozen_native_fp32_records.jsonl'], 'Pilot precision reference differs')
            cache_index = [key for key in pilot_sources if Path(key).name == 'index.json']
            require(len(cache_index) == 1 and Path(cache_index[0]).is_absolute() and
                    pilot_sources[cache_index[0]] == objective['cache_index_sha256']['development'], 'Pilot native cache index differs')
            cache_complete = str(Path(cache_index[0]).with_name('complete.json')); paths.update([cache_index[0],cache_complete])
            require(pilot_sources.get(cache_complete) == context['development']['cache_complete_sha256'] and set(pilot_sources) == paths,
                    'Pilot cache completion/source path ledger differs')
            if base_pilot_sources is None:
                base_pilot_sources = pilot_sources
            else:
                require(pilot_sources == base_pilot_sources, 'Pilot reference identities differ across shards')
        else:
            require(done['pilot_reference_sources_sha256'] is None and not (directory/'pilot_reference_sources.json').exists(),
                    'Full cannot substitute pilot-cache references')
            pilot_sources_sha = None
        for descriptor,planned in zip(index['records'],expected):
            identity = descriptor['identity']; token,ordinal = planned['sample_token'],planned['selection_ordinal']
            require(all(identity.get(key) == value for key,value in planned.items()) and
                    set(identity) == set(planned)|{'data_info_index'} and type(identity['data_info_index']) is int and identity['data_info_index'] >= 0,
                    'Runtime sample index differs from frozen identity/ordinal')
            require(ordinal not in joined and descriptor['directory'] == 'samples/'+token, 'Repeated ordinal or unsafe sample path')
            sample = (root/descriptor['directory']).resolve()
            require(root in sample.parents and {path.name for path in sample.iterdir()} == {'complete.json','records.json'},
                    'Sample path escapes root or its file set is incomplete/unexpected')
            sample_done,sample_done_sha = read(sample/'complete.json',descriptor['complete_sha256'])
            require(sample_done['status'] == 'COMPLETE_SAMPLE' and sample_done['sample_token'] == token and
                    sample_done['official_index'] == planned['official_index'] and sample_done['selection_ordinal'] == ordinal and
                    sample_done['shard_index'] == shard_index and set(sample_done['files_sha256']) == {'records.json'} and
                    sample_done['files_sha256']['records.json'] == descriptor['records_sha256'], 'Sample completion identity/hash differs')
            result,result_sha = read(sample/'records.json',descriptor['records_sha256'])
            hist,counts = validate_sample(result,identity,contract_sha,a.mode,names)
            require(math.isclose(descriptor['total_seconds'],result['total_seconds'],rel_tol=1e-12,abs_tol=1e-12), 'Sample/index timing differs')
            joined[ordinal] = dict(identity=identity,hist=hist,counts=counts,result=result,directory=descriptor['directory'],
                complete_sha256=sample_done_sha,records_sha256=result_sha)
        require(math.isfinite(done['seconds']) and done['seconds'] >= 0, 'Invalid shard timing')
        shards.append(dict(shard_index=shard_index,samples=len(expected),seconds=done['seconds'],complete_sha256=done_sha,
            index_sha256=index_sha,manifest_sha256=manifest_sha,loaded_models_sha256=loaded_sha,
            pilot_reference_sources_sha256=pilot_sources_sha,planning_disabled_all_models=done['planning_disabled_all_models']))
    require(sorted(joined) == list(range(len(rows))), 'Partial or overlapping global coverage')
    ordered = [joined[i] for i in range(len(rows))]
    require([row['identity']['official_index'] for row in ordered] == [row['official_index'] for row in rows], 'Merged official order differs')
    return contract,contract_sha,ordered,shards,base_loaded,auth_sha,pilot_gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-root','evaluation-protocol','objective-protocol','protocol','selection','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=('pilot','full'),default='pilot')
    parser.add_argument('--full-authorization')
    a = parser.parse_args(argv); started = time.monotonic(); out = Path(a.out).resolve()
    require(not out.exists(), 'Output must be a new directory; no overwrite/retry')
    root = Path(a.evaluation_root).resolve()
    protected = [root/'samples',root/'shard_00_of_02',root/'shard_01_of_02']
    require(not any(path == out or path in out.parents for path in protected), 'Output cannot mutate producer sample/shard directories')
    context = load_protocols(a)
    contract,contract_sha,ordered,shards,loaded,auth_sha,pilot_gate = collect(a,context)
    names = context['models']; evaluation = context['evaluation']; identities = [row['identity'] for row in ordered]
    resources = resource_summary(ordered,shards,names)
    sources = dict(evaluation_protocol_sha256=context['evaluation_sha'],objective_train_protocol_sha256=OBJECTIVE_SHA,
        parent_protocol_sha256=PARENT_SHA,selection_sha256=SELECTION_SHA[a.mode],contract_sha256=contract_sha,
        producer_sha256=evaluation['source_sha256']['objective_joint_evaluation_v2.py'],
        final_checkpoints_sha256=contract['final_checkpoints_sha256'],final_head_state_sha256=contract['final_head_state_sha256'],
        actual_loaded_head_sha256=loaded['head_state_sha256'],shards=shards,
        evaluation_source_sha256=evaluation['source_sha256'],parent_source_sha256=context['parent']['source_sha256'],
        objective_source_sha256=context['objective']['source_sha256'],runtime_source_sha256=contract['runtime_source_sha256'],
        data_recipe_receipt=contract['data_recipe_receipt'],full_authorization_sha256=auth_sha,pilot_gate=pilot_gate,
        development_summary_sha256=evaluation['development_summary']['summary_sha256'],
        development_complete_sha256=evaluation['development_summary']['complete_sha256'],
        actual_selection_path=str(Path(a.selection).resolve()),registered_selection_path=evaluation['selections'][a.mode]['file'],
        evaluation_root=str(root),annotation_sha256=context['selection']['ann_sha256'],
        checkpoint_audit='producer CPU payload audit and strict GPU tensor identity are SHA-bound; merger checks receipts against complete development audit without rereading checkpoint bytes',
        raw_source_assets_reread_by_merger=False,runtime_files_rehashed_by_merger=False,
        all_local_bound_python_sources_rehashed=True,inactive_policy_evidence=context['policy_evidence'])
    common = dict(samples=len(ordered),scenes=len({row['scene_token'] for row in identities}),
        candidate_names=evaluation['candidate_names'],model_names=list(names),ordered_identities=identities,
        sources=sources,resources=resources,nomination_rule=NOMINATION_RULE,
        development_disclosure='development_disclosure.json',optimizer_steps=0,
        inactive_boundary_policy=evaluation['inactive_boundary_policy'])
    if a.mode == 'pilot':
        summary = dict(common,schema='objective-joint-pilot-merge-v2',status='PILOT_ENGINEERING_PARITY_RESOURCE_PASS',
            parity=dict(all_sample_hash_chains_valid=True,full_official_pilot_order=True,
                all_models_five_horizon_GT_counts_match=True,all_active_boundaries_exact=True,
                all_original_GT_logits_and_reference_confusion_checks_passed=True,
                full_boundary_equality_not_required=True,
                full_boundary_hash_equal_by_sample={row['identity']['sample_token']:row['result']['engineering_parity']['full_boundary_hash_equal'] for row in ordered},
                planning_disabled_for_every_model=True,comparison_only_projection=True,actual_inputs_modified=False,
                pilot_reference_ledger_SHA_bound_by_each_shard_complete=True),
            performance_metrics_computed=False,performance_success_claim=False,
            interpretation='Two frozen scenes establish active-boundary/GT/prediction/reference parity and resource evidence only. Full-tree equality is reported separately; only inactive random sample_traj is excluded from historical-cache comparison. No effectiveness score or full-run timing guarantee')
    else:
        metric = frozen_module('aggregate_memory.py',context['parent']['source_sha256']['aggregate_memory.py'])
        aggregate = frozen_module('objective_supervision_aggregate.py',OBJECTIVE_AGGREGATE_SHA)
        histograms = {name:np.stack([row['hist'][name] for row in ordered]) for name in names}
        models,comparisons,bootstrap = aggregate.aggregate(histograms,[row['scene_token'] for row in identities],
            metric.metric_arrays,.5,names=names,contrasts=context['contrasts'],primary=context['primary'])
        require(bootstrap['scene_count'] == 150 and bootstrap['replicates'] == 10000 and bootstrap['seed'] == 11 and
                bootstrap['model_order'] == list(names) and bootstrap['primary_contrasts'] == 2*len(evaluation['candidate_names']),
                'Parameterized full bootstrap scope differs')
        summary = dict(common,schema='objective-joint-full-summary-v2',status='COMPLETE_FULL_NATIVE_COMPARISON',
            models=models,comparisons=comparisons,bootstrap=bootstrap,
            registered_primary_contrasts=[list(pair) for pair in context['primary']],registered_minimum_meaningful_effect_pp=.5,
            primary='arithmetic mean of four future independently pooled GMO IoUs',
            secondary='GMO IoU after pooling all four future confusion matrices',
            units=dict(scores='ratio',differences='ratio and pp=100*ratio'),
            historical_validation_exposure=True,fixed_training_seed=11,training_seeds_added=0,
            full_validation=True,new_blind_test=False,threshold_fitting=False,R4_calibration_reused=False,
            historical_cache_projection_applied=False,complete_fresh_input_hash_chain_retained=True,
            multiple_comparison_adjustment=False,training_seed_robustness_claim=False,
            unqualified_candidates_not_evaluated_not_zero_scored=True,F_plus_O_not_run=True,
            goal_complete_decision='NOT_MADE_BY_MERGER')
    # First output mutation follows complete protocol/source/coverage checks.
    out.mkdir(parents=True,exist_ok=False)
    if a.mode == 'pilot':
        (out/'report.md').write_text('# 原生共同评价工程预检\n\n两条冻结 validation anchors、两个不同场景、两个互斥分片的完整哈希链、实际模型权重、有效输入、五时域 GT 及既有预测/混淆矩阵 parity 全部通过。历史缓存比较只排除规划关闭时唯一无效的 sample_traj；双方完整哈希及其相等标志如实保留，并不声称完整边界相同。真实模型输入未改动。模型为 '+', '.join(names)+'。本文件只报告工程与资源，不计算或汇报性能提升；两条首样本时延不能保证完整验证耗时。所有五模型开发结果与固定晋级依据见 development_disclosure.json。\n')
    else:
        full_outputs(out,summary,ordered,names)
    summary.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),merger_sha256=sha(__file__),
                   merge_seconds=time.monotonic()-started)
    write(out/'summary.json',summary)
    ledger = [dict(identity=row['identity'],directory=row['directory'],records_sha256=row['records_sha256'],
        complete_sha256=row['complete_sha256'],input_tree_sha256=row['result']['input_tree_sha256'],
        targets_array_sha256=row['result']['targets_array_sha256'],observed_bev_tensor_sha256=row['result']['observed_bev_tensor_sha256'],
        native_radar_tensor_sha256=row['result']['native_radar_tensor_sha256'],logits_array_sha256=row['result']['logits_array_sha256'],
        engineering_parity=row['result']['engineering_parity'])
        for row in ordered]
    write(out/'sample_hash_ledger.json',ledger); write(out/'development_disclosure.json',context['development_disclosure'])
    filenames = ['summary.json','report.md','sample_hash_ledger.json','development_disclosure.json']
    if a.mode == 'full':
        filenames += ['raw_confusions.npz','metrics.csv','contrasts.csv']
    write(out/'complete.json',dict(schema='objective-joint-merge-complete-v2',status='COMPLETE',mode=a.mode,samples=len(ordered),scenes=summary['scenes'],
        summary_sha256=sha(out/'summary.json'),files_sha256={name:sha(out/name) for name in filenames},
        evaluation_protocol_sha256=context['evaluation_sha'],merge_seconds=time.monotonic()-started,
        source_result_files_modified=False))
    print(json.dumps(dict(status=summary['status'],out=str(out),samples=len(ordered),models=list(names))),flush=True)


if __name__ == '__main__':
    main()
