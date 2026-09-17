"""Native joint evaluation of M0/M0_fp32/C and every frozen qualifying F/O arm.

The native data/capture/replay/evaluator body is derived from the byte-locked
old joint producer, in a private namespace. Only source-contract plumbing,
single-slot model selection, receipt schemas and allocator/audit gates change.
No loss adapter, new objective, threshold, training, resume or dispatch exists.
Full mode additionally requires an immutable authorization bound to pilot PASS.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import signal
import textwrap
import time
import traceback
import types

OLD_PRODUCER_SHA = '280a032a95070f4a618664f5e47788f00a04d8c048865b5e6000f80b724d1578'
PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
OBJECTIVE_SHA = '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d'
INITIAL_HEAD_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
SELECTION_SHA = dict(pilot='e616af491e77ad977dfddfd931c1110f9580921bc38ed6ad84281dfe23a9754d',
                    full='60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391')
NOMINATION_RULE = dict(metric='future_macro_gmo', comparators=['M0_fp32','native1'],
    minimum_point_delta_pp=.5, minimum_ci_lower_pp=0., ci_lower_strict=True,
    include_all_qualifying_in_F_O_order=True)
HORIZONS = [0., .5, 1., 1.5, 2.]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    data = Path(path).read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Hash mismatch: '+str(path))
    return json.loads(data)


def resolve(path, protocol_path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(protocol_path).resolve().parent/path).resolve()


def frozen_module(name, digest):
    path = Path(__file__).with_name(name)
    require(Path(name).name == name and sha(path) == digest, 'Frozen helper changed: '+name)
    spec = importlib.util.spec_from_file_location('_objective_joint_'+path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def nomination(summary, rule):
    require(rule == NOMINATION_RULE, 'Nomination rule differs from the pre-outcome decision')
    require(summary['schema'] == 'm0-objective-supervision-aggregate-v1' and
            summary['status'] == 'COMPLETED_DEVELOPMENT_COMPARISON' and
            set(summary['models']) == {'M0','M0_fp32','native1','F','O'} and
            summary['samples'] == 200 and summary['scenes'] == 100 and summary['audit']['status'] == 'PASS',
            'Require the complete five-model development result, including nonqualifiers')
    rows = {(r['left'],r['right']):r for r in summary['comparisons']}
    require(len(rows) == len(summary['comparisons']) == 6, 'Development contrast schema differs')
    qualifying, evidence = [], {}
    for arm in ('F','O'):
        checks = []
        for comparator in NOMINATION_RULE['comparators']:
            row = rows[(arm,comparator)]; values = row['metrics']['future_macro_gmo']
            point, lower = values['delta_pp'], values['ci95_pp'][0]
            require(row['primary_contrast'] is True and math.isfinite(point) and math.isfinite(lower), 'Invalid nomination statistic')
            passed = point >= .5 and lower > 0
            checks.append(dict(comparator=comparator, point_delta_pp=point, ci95_lower_pp=lower, passed=passed))
        evidence[arm] = checks
        if all(r['passed'] for r in checks):
            qualifying.append(arm)
    require(qualifying, 'No candidate passed the frozen development nomination rule')
    return qualifying, evidence


def load_contracts(evaluation_path, objective_path, parent_path, verify_runtime=True):
    parent, objective = read(parent_path,PARENT_SHA), read(objective_path,OBJECTIVE_SHA)
    protocol = read(evaluation_path)
    require(protocol['schema'] == 'objective-joint-evaluation-protocol-v1' and
            protocol['status'] == 'FROZEN_BEFORE_OBJECTIVE_JOINT_PILOT', 'Wrong immutable joint protocol')
    require(protocol['parent_protocol_sha256'] == PARENT_SHA and
            protocol['objective_train_protocol_sha256'] == OBJECTIVE_SHA, 'Training protocol binding differs')
    required = {'objective_joint_evaluation.py','objective_joint_merge.py','joint_native_evaluation.py',
                'merge_joint_evaluation.py'}
    require(required <= set(protocol['source_sha256']) and
            protocol['source_sha256']['joint_native_evaluation.py'] == OLD_PRODUCER_SHA, 'Missing original/new producer source lock')
    for name, digest in protocol['source_sha256'].items():
        require(Path(name).name == name and sha(Path(__file__).with_name(name)) == digest, 'Joint source changed: '+name)
    trainer = frozen_module('objective_supervision_train.py',objective['source_sha256']['objective_supervision_train.py'])
    checked_parent, checked_objective, helper = trainer.load_contracts(objective_path,parent_path,verify_runtime=verify_runtime)
    require(checked_parent == parent and checked_objective == objective, 'Training protocol changed while reading')
    for key in ('config_sha256','m0_sha256','numerical_policy'):
        require(protocol[key] == objective[key] == parent[key], 'Evaluation changed native source/precision: '+key)
    require(protocol['runtime_source_sha256'] == objective['runtime_source_sha256'], 'Evaluation changed runtime sources')
    require(protocol['historical_validation_exposure'] is True and protocol['new_blind_test'] is False and
            protocol['multiple_comparison_adjustment'] is False and protocol['training_seeds_added'] == 0,
            'Scientific scope disclosure differs')
    directory = resolve(protocol['development_summary']['directory'],evaluation_path)
    require(not (directory/'failed.json').exists(), 'Development aggregation failed')
    complete = read(directory/'complete.json',protocol['development_summary']['complete_sha256'])
    summary = read(directory/'summary.json',protocol['development_summary']['summary_sha256'])
    require(complete['status'] == 'COMPLETE' and complete['summary_sha256'] == protocol['development_summary']['summary_sha256'] and
            complete['objective_protocol_sha256'] == OBJECTIVE_SHA and summary['objective_protocol_sha256'] == OBJECTIVE_SHA and
            summary['parent_protocol_sha256'] == PARENT_SHA and
            summary['script_sha256'] == objective['source_sha256']['objective_supervision_aggregate.py'],
            'Development result protocol/source/hash chain differs')
    for filename,digest in complete['files_sha256'].items():
        require(Path(filename).name == filename and sha(directory/filename) == digest, 'Development output changed: '+filename)
    qualifying, evidence = nomination(summary,protocol['nomination_rule'])
    require(protocol['candidate_names'] == qualifying, 'Candidate list must include all qualifying F/O in fixed order')
    names = ['M0','M0_fp32','native1']+qualifying
    require(set(protocol['final_sources']) == {'native1',*qualifying}, 'Wrong fixed final source set')
    for arm,binding in protocol['final_sources'].items():
        source = summary['source_receipts'][arm]
        files = source['reuse_audit']['files_sha256'] if arm == 'native1' else source['files_sha256']
        require(binding['files_sha256'] == files and
                binding['final_head_state_sha256'] == source['checkpoint']['final_head_state_sha256'],
                'Final source differs from complete development result: '+arm)
    for mode in ('pilot','full'):
        selected = protocol['selections'][mode]
        require(selected['sha256'] == SELECTION_SHA[mode] and
                sha(resolve(selected['file'],evaluation_path)) == selected['sha256'], 'Frozen selection changed: '+mode)
        cap = protocol['resources'][mode]
        require(cap['shard_count'] == 2 and type(cap['max_seconds']) is int and 0 < cap['max_seconds'] <= 43200 and
                math.isfinite(cap['max_allocated_gib']) and cap['max_allocated_gib'] > 0, 'Invalid frozen evaluation budget')
    return parent, objective, protocol, summary, names, evidence, helper


def full_gate(path, evaluation_path, protocol, names):
    require(path is not None, 'Full evaluation requires a separate frozen pilot-PASS authorization')
    authorization = read(path)
    require(authorization['schema'] == 'objective-joint-full-authorization-v1' and
            authorization['status'] == 'AUTHORIZED_AFTER_PILOT_PASS' and
            authorization['evaluation_protocol_sha256'] == sha(evaluation_path), 'Wrong full authorization')
    resources=authorization['resources']; ceiling=protocol['resources']['full']
    require(set(resources)=={'max_seconds','max_allocated_gib','shard_count'} and resources['shard_count']==2 and
            type(resources['max_seconds']) is int and 0<resources['max_seconds']<=ceiling['max_seconds'] and
            math.isfinite(resources['max_allocated_gib']) and 0<resources['max_allocated_gib']<=ceiling['max_allocated_gib'],
            'Full authorization resources exceed immutable evaluation ceilings')
    binding = authorization['pilot_merge']; directory = resolve(binding['directory'],path)
    require(not (directory/'failed.json').exists(), 'Pilot merger has failure receipt')
    complete = read(directory/'complete.json',binding['complete_sha256'])
    summary = read(directory/'summary.json',binding['summary_sha256'])
    require(complete['status'] == 'COMPLETE' and complete['mode'] == 'pilot' and complete['samples'] == 2 and
            complete['summary_sha256'] == binding['summary_sha256'], 'Pilot completion hash chain differs')
    for filename,digest in complete['files_sha256'].items():
        require(Path(filename).name == filename and sha(directory/filename) == digest, 'Pilot merged output changed')
    require(summary['schema'] == 'objective-joint-pilot-merge-v1' and
            summary['status'] == 'PILOT_ENGINEERING_PARITY_RESOURCE_PASS' and summary['samples'] == summary['scenes'] == 2 and
            summary['candidate_names'] == protocol['candidate_names'] and summary['model_names'] == names and
            summary['performance_metrics_computed'] is False and summary['performance_success_claim'] is False and
            summary['merger_sha256'] == protocol['source_sha256']['objective_joint_merge.py'], 'Pilot did not prove the selected joint implementation')
    source = summary['sources']
    require(source['evaluation_protocol_sha256'] == sha(evaluation_path) and source['objective_train_protocol_sha256'] == OBJECTIVE_SHA and
            source['parent_protocol_sha256'] == PARENT_SHA and source['selection_sha256'] == SELECTION_SHA['pilot'], 'Pilot protocol/selection differs')
    require(source['final_checkpoints_sha256'] == {k:v['files_sha256']['latest.pth'] for k,v in protocol['final_sources'].items()} and
            source['final_head_state_sha256'] == {k:v['final_head_state_sha256'] for k,v in protocol['final_sources'].items()},
            'Pilot and full final heads differ')
    return dict(directory=str(directory),complete_sha256=binding['complete_sha256'],summary_sha256=binding['summary_sha256'])


def derive_run(base, namespace):
    """Small, counted AST edits; every data/forward/evaluator expression retained."""
    node = ast.parse(textwrap.dedent(inspect.getsource(base.Evaluation.run))).body[0]
    original = copy.deepcopy(node); counts = dict(common=0,mode=0,sample_schema=0,pilot_flag=0,cap=0,head_gate=0,pilot_ledger=0)
    replacements={};inserted=set()
    class Adapt(ast.NodeTransformer):
        def visit_Assign(self,item):
            if len(item.targets) == 1 and isinstance(item.targets[0],ast.Name):
                target = item.targets[0].id
                if target == 'common':
                    counts['common'] += 1
                    new=ast.parse('common=self.common_contract(protocol,selection,arm_receipts)').body[0]
                    replacements[id(new)]=copy.deepcopy(item);return new
                if target == 'mode':
                    require(isinstance(item.value,ast.IfExp), 'Original single-slot mapping syntax differs')
                    replacements[id(item)]=copy.deepcopy(item)
                    counts['mode'] += 1; item.value = ast.Constant('native1')
                if target == 'wrapped':
                    counts['head_gate'] += 1
                    new=ast.parse('self.check_loaded_heads(head_digests,migrations,arm_receipts)').body[0]
                    inserted.add(id(new));return [new,item]
            return self.generic_visit(item)
        def visit_Constant(self,item):
            if item.value == 'native-joint-evaluation-sample-v1':
                counts['sample_schema'] += 1;new=ast.Constant('objective-joint-sample-v1')
                replacements[id(new)]=copy.deepcopy(item);return new
            if item.value == 'all_5_models_5_horizons_full_GT_confusion_equal':
                counts['pilot_flag'] += 1;new=ast.Constant('all_models_5_horizons_full_GT_confusion_equal')
                replacements[id(new)]=copy.deepcopy(item);return new
            return item
        def visit_Expr(self,item):
            call = item.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr == 'set_device':
                counts['cap'] += 1
                new=ast.parse('self.apply_allocator_cap(torch,device)').body[0]
                inserted.add(id(new));return [item,new]
            return self.generic_visit(item)
        def visit_Call(self,item):
            item = self.generic_visit(item)
            if isinstance(item.func,ast.Name) and item.func.id == 'dict' and any(
                    k.arg == 'status' and isinstance(k.value,ast.Constant) and k.value.value == 'COMPLETE_JOINT_NATIVE_EVALUATION_SHARD'
                    for k in item.keywords):
                counts['pilot_ledger'] += 1
                replacements[id(item)]=copy.deepcopy(item)
                value = ast.parse("sha(self.shard/'pilot_reference_sources.json') if a.mode=='pilot' else None",mode='eval').body
                item.keywords.append(ast.keyword(arg='pilot_reference_sources_sha256',value=value))
            return item
    node = Adapt().visit(node)
    require(all(v == 1 for v in counts.values()), 'Unexpected original native run structure: '+str(counts))
    from frame_consistent_adapter import ast_sha
    class Restore(ast.NodeTransformer):
        def visit(self,item):
            if id(item) in inserted:return None
            if id(item) in replacements:return copy.deepcopy(replacements[id(item)])
            return super().visit(item)
    # Restore the same node identities, then retain an independent compiled copy.
    derived=copy.deepcopy(node);restored=Restore().visit(node)
    require(ast_sha(restored)==ast_sha(original),'Inverse of seven audited edits does not restore original native run')
    node=derived
    receipt = dict(original_producer_sha256=OLD_PRODUCER_SHA,original_run_ast_sha256=ast_sha(original),
        derived_run_ast_sha256=ast_sha(node),edits=counts,
        inverse_ast_restores_original_run=True,
        original_native_loader_capture_replay_GT_evaluator_expressions_unchanged=True,
        namespace='private function globals; original module globals unchanged',
        inference_loss_adapter_installed=False,all_replay_architectures='native1')
    tree = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(tree,'<objective-joint:source-locked-native-run>','exec'),namespace)
    return namespace['run'],receipt


class ObjectiveEvaluation:
    """Own the new source gates; delegate only the locked native stream body."""
    def __init__(self,a,base):
        self.a=a;self.base=base
        self.parent,self.objective,self.protocol,self.development,self.names,self.nomination_evidence,self.helper = load_contracts(
            a.evaluation_protocol,a.objective_protocol,a.protocol)
        self.arms=self.names[2:];cap=self.protocol['resources'][a.mode]
        selected=self.protocol['selections'][a.mode]
        chosen=resolve(selected['file'],a.evaluation_protocol)
        require(a.selection is None or Path(a.selection).resolve() == chosen,'CLI selection differs from frozen protocol')
        a.selection=str(chosen);a.selection_sha256=selected['sha256'];a.shard_count=cap['shard_count'];a.runs_root=None
        require(a.shard_index in (0,1), 'Require one of the two fixed shards')
        if a.mode == 'pilot':
            require(a.full_authorization is None and a.parity_cache is not None,'Pilot requires native dev cache and no full authorization')
            self.pilot_gate=None
            self.authorization_sha=None
            a.max_seconds=cap['max_seconds'] if a.max_seconds is None else a.max_seconds
            a.max_allocated_gib=cap['max_allocated_gib'] if a.max_allocated_gib is None else a.max_allocated_gib
            require(type(a.max_seconds) is int and 0<a.max_seconds<=cap['max_seconds'] and
                    math.isfinite(a.max_allocated_gib) and 0<a.max_allocated_gib<=cap['max_allocated_gib'],
                    'Pilot runtime resources exceed fixed ceilings')
        else:
            require(a.parity_cache is None,'Full uses real native loader; no cached input substitution')
            self.authorization_sha=sha(a.full_authorization) if a.full_authorization is not None else None
            self.pilot_gate=full_gate(a.full_authorization,a.evaluation_protocol,self.protocol,self.names)
            actual=read(a.full_authorization)['resources']
            require(sha(a.full_authorization)==self.authorization_sha,'Full authorization changed while reading')
            require(a.max_seconds==actual['max_seconds'] and a.max_allocated_gib==actual['max_allocated_gib'],
                    'Full CLI resources must exactly equal the post-pilot authorization')
        self.effective_resources=dict(max_seconds=a.max_seconds,max_allocated_gib=a.max_allocated_gib,shard_count=2)
        base.Evaluation.__init__(self,a)
        self.final_receipts=None
        namespace=dict(base.__dict__,MODELS=tuple(self.names),ARMS=tuple(self.arms),
            frozen_arms=self.frozen_arms,pilot_reference=self.pilot_reference)
        run,self.recipe=derive_run(base,namespace);self.run=types.MethodType(run,self)

    def check(self):
        return self.base.Evaluation.check(self)

    def contracts(self):
        require(sha(self.a.config)==self.parent['config_sha256'] and sha(self.a.checkpoint)==self.parent['m0_sha256'],
                'Native config/M0 checkpoint differs')
        current=load_contracts(self.a.evaluation_protocol,self.a.objective_protocol,self.a.protocol)
        require(current[:6] == (self.parent,self.objective,self.protocol,self.development,self.names,self.nomination_evidence),
                'Frozen evaluation context changed')
        if self.a.mode=='full':
            require(sha(self.a.full_authorization)==self.authorization_sha,'Full authorization changed during evaluation')
        return self.parent,read(self.a.selection,SELECTION_SHA[self.a.mode])

    def apply_allocator_cap(self,torch,device):
        total=torch.cuda.get_device_properties(device).total_memory
        cap=int(self.effective_resources['max_allocated_gib']*2**30)
        require(0<cap<=total,'Frozen allocator cap exceeds device capacity')
        torch.cuda.set_per_process_memory_fraction(cap/total,device=device)

    def frozen_arms(self,_unused,parent):
        # Initial CPU payload audit is reusable; final call rehashes every byte
        # bound below. Actual GPU tensors are independently checked in run.
        for arm,binding in self.protocol['final_sources'].items():
            directory=resolve(binding['directory'],self.a.evaluation_protocol)
            require(not (directory/'failed.json').exists() and not (directory/'interrupted.pth').exists(),'Failed/nonfinal source: '+arm)
            for filename,digest in binding['files_sha256'].items():
                require(Path(filename).name==filename and sha(directory/filename)==digest,'Final source changed: '+arm+'/'+filename)
        if self.final_receipts is not None:
            return self.final_receipts
        audit=frozen_module('frame_aggregate.py',self.objective['source_sha256']['frame_aggregate.py'])
        metric=frozen_module('aggregate_memory.py',parent['source_sha256']['aggregate_memory.py'])
        control_binding=self.protocol['final_sources']['native1']
        require(control_binding['files_sha256']==self.objective['control_files_sha256'],'Wrong fixed C source')
        args=types.SimpleNamespace(control_run=str(resolve(control_binding['directory'],self.a.evaluation_protocol)))
        records=self.development['ordered_identities_and_GT']
        control,payload,_=self.helper.validate_control(args,parent,self.objective,{'development':records});del payload
        receipts={};control_payload=None
        for arm in self.arms:
            binding=self.protocol['final_sources'][arm];directory=resolve(binding['directory'],self.a.evaluation_protocol)
            manifest,trained,done=[read(directory/f) for f in ('manifest.json','training_complete.json','complete.json')]
            expected_protocol=PARENT_SHA if arm=='native1' else OBJECTIVE_SHA
            require(manifest['arm']==arm and manifest['protocol_sha256']==expected_protocol and manifest['seed']==11 and
                    manifest['initial_head_sha256']==INITIAL_HEAD_SHA and manifest['m0_sha256']==parent['m0_sha256'], 'Final initialization/protocol differs')
            require(manifest['migration']==control['migration'] and manifest['trainable_parameter_names']==control['trainable_parameter_names'] and
                    manifest['trainable_parameters']==13274016 and manifest['numerical_policy']==parent['numerical_policy'], 'Wrong final single-slot structure/precision')
            require(manifest['train_index_sha256']==self.objective['cache_index_sha256']['train'] and
                    manifest['dev_index_sha256']==self.objective['cache_index_sha256']['development'], 'Wrong final train/dev source')
            require(trained['status']=='TRAINED_FIXED_FINAL' and done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED', 'Missing fixed final completion')
            require(all(r['updates']==512 and r['examples']==2048 and r['checkpoint_sha256']==binding['files_sha256']['latest.pth']
                        for r in (trained,done)) and done['evaluation']['samples']==200 and
                    done['evaluation']['sha256']==binding['files_sha256']['development_records.jsonl'], 'Final training/evaluation budget differs')
            if arm!='native1':
                require(manifest['source_sha256']==self.objective['source_sha256'] and manifest['parent_source_sha256']==parent['source_sha256'] and
                        done['final_head_state_sha256']==binding['final_head_state_sha256'] and
                        done['manifest_sha256']==binding['files_sha256']['manifest.json'] and
                        manifest['objective_adapter']==self.development['source_receipts'][arm]['objective_adapter'], 'Candidate objective/final receipt differs')
            rows=[json.loads(line) for line in (directory/'development_records.jsonl').read_text().splitlines() if line.strip()]
            require(len(rows)==200,'Incomplete fixed dev results')
            for row,identity in zip(rows,records):
                require(row['sample_token']==identity['sample_token'] and row['scene_token']==identity['scene_token'] and
                        row['horizon_seconds']==HORIZONS,'Final dev identity/order differs')
                import numpy as np
                hist=metric.hist_array(row['hist_by_horizon'])
                require(np.array_equal(hist.sum(2),np.asarray(identity['target_counts_0_1_255_by_frame'])[2:,:2]),'Final dev GT counts differ')
            audit.check_log(directory,self.helper,parent['training'])
            loaded=audit.check_payload(directory,manifest,binding['files_sha256']['latest.pth'],arm,
                self.helper.planned_orders(parent['training']),binding['final_head_state_sha256'])
            if arm=='native1':control_payload=loaded
            else:
                require(all(loaded[k]==control_payload[k] for k in ('tensor_shapes','optimizer_groups','sample_orders_sha256')),
                        'Actual candidate final optimizer/order/shape differs from C')
            receipts[arm]=dict(directory=str(directory),files_sha256=binding['files_sha256'],manifest=manifest,
                final_head_state_sha256=binding['final_head_state_sha256'],checkpoint_audit=loaded)
        self.final_receipts=receipts;return receipts

    def common_contract(self,parent,selection,receipts):
        return dict(schema='objective-joint-evaluation-contract-v1',mode=self.a.mode,models=self.names,
            candidate_names=self.protocol['candidate_names'],evaluation_protocol_sha256=sha(self.a.evaluation_protocol),
            objective_train_protocol_sha256=OBJECTIVE_SHA,parent_protocol_sha256=PARENT_SHA,
            selection_sha256=self.a.selection_sha256,script_sha256=sha(__file__),config_sha256=parent['config_sha256'],
            source_sha256=self.protocol['source_sha256'],parent_source_sha256=parent['source_sha256'],
            objective_source_sha256=self.objective['source_sha256'],runtime_source_sha256=self.protocol['runtime_source_sha256'],
            m0_sha256=parent['m0_sha256'],final_checkpoints_sha256={k:v['files_sha256']['latest.pth'] for k,v in receipts.items()},
            final_head_state_sha256={k:v['final_head_state_sha256'] for k,v in receipts.items()},
            development_summary_sha256=self.protocol['development_summary']['summary_sha256'],nomination_evidence=self.nomination_evidence,
            shard_count=self.a.shard_count,selected_samples=selection['samples'],numerical_policy=parent['numerical_policy'],
            horizons=HORIZONS,optimizer_steps=0,sample_partition='selection_ordinal modulo shard_count',
            historical_validation_exposure=True,full_authorization_sha256=self.authorization_sha,
            pilot_gate=self.pilot_gate,data_recipe_receipt=self.recipe,
            resources=self.effective_resources,allocator_cap_bytes=int(self.effective_resources['max_allocated_gib']*2**30))

    def check_loaded_heads(self,heads,migrations,receipts):
        require(set(heads)==set(self.names) and heads['M0']==heads['M0_fp32']==INITIAL_HEAD_SHA,'Loaded frozen M0 changed')
        require(set(migrations)==set(self.names[1:]) and all(v['mode']=='native1' for v in migrations.values()),'No inference geometry/memory intervention permitted')
        for arm in self.arms:
            require(heads[arm]==receipts[arm]['final_head_state_sha256'] and
                    migrations[arm]==receipts[arm]['manifest']['migration'],'Loaded final head/migration differs: '+arm)

    def pilot_reference(self,cache,receipts,parent):
        root=Path(cache).resolve();done,index=read(root/'complete.json'),read(root/'index.json')
        require(done['status']=='COMPLETE_NATIVE_STATE_CACHE' and done['index_sha256']==sha(root/'index.json')==
                self.objective['cache_index_sha256']['development'],'Pilot native cache hash differs')
        require(index['schema']=='m0-native-state-cache-v1' and index['status']=='COMPLETE' and index['split']=='development' and
                index['config_sha256']==parent['config_sha256'] and index['selection_sha256']==parent['selection_sha256'] and
                index['script_sha256']==parent['source_sha256']['native_state_cache.py'] and index['m0_sha256']==parent['m0_sha256'] and
                done['all_samples_bitwise_replayed'] is True,'Pilot native cache provenance differs')
        lookup={r['sample_token']:r for r in index['records']};require(len(lookup)==done['samples']==200,'Pilot native cache coverage differs')
        references={name:{} for name in self.names}
        for token,row in lookup.items():
            require(row['parity']['bitwise_all_five_horizons_three_layers'] and row['parity']['exact_native_confusion'],'Pilot native cache parity absent')
            references['M0'][token]=row['native_hist']
        sources={str(root/'complete.json'):sha(root/'complete.json'),str(root/'index.json'):sha(root/'index.json')}
        for arm in self.arms:
            path=Path(receipts[arm]['directory'])/'development_records.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require([r['sample_token'] for r in rows]==list(lookup),'Pilot final dev order differs')
            references[arm]={r['sample_token']:r['hist_by_horizon'] for r in rows};sources[str(path)]=sha(path)
        path=Path(receipts['native1']['directory'])/'frozen_native_fp32_records.jsonl'
        require(sha(path)==receipts['native1']['manifest']['frozen_precision_reference']['sha256'],'Pilot frozen FP32 reference changed')
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require([r['sample_token'] for r in rows]==list(lookup),'Pilot frozen FP32 coverage/order differs')
        references['M0_fp32']={r['sample_token']:r['hist_by_horizon'] for r in rows};sources[str(path)]=sha(path)
        return root,lookup,references,sources


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-protocol','objective-protocol','protocol','config','checkpoint','repo','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('pilot','full'),default='pilot');p.add_argument('--shard-index',type=int,required=True)
    p.add_argument('--selection');p.add_argument('--parity-cache');p.add_argument('--full-authorization')
    p.add_argument('--device',default='cuda:0');p.add_argument('--max-seconds',type=int)
    p.add_argument('--max-allocated-gib',type=float);p.add_argument('--deadline-unix',type=float)
    a=p.parse_args(argv)
    require(a.deadline_unix is None or math.isfinite(a.deadline_unix),'Invalid absolute deadline')
    base=frozen_module('joint_native_evaluation.py',OLD_PRODUCER_SHA);runner=ObjectiveEvaluation(a,base)
    def interrupted(number,_frame):
        raise InterruptedError('Objective joint evaluation signal '+str(number))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(max(1,int(min(a.max_seconds,max(0,runner.end-time.monotonic())))))
    try:
        runner.run();return 0
    except BaseException as exc:
        base.write(runner.shard/'failed.json',dict(status='FAILED_OR_INTERRUPTED',error=repr(exc),
            traceback=traceback.format_exc(),completed=len(runner.results),optimizer_steps=0,automatic_retry=False),exclusive=True)
        traceback.print_exc();return 1
    finally:
        signal.alarm(0);runner.lock.close()


if __name__=='__main__':
    raise SystemExit(main())
