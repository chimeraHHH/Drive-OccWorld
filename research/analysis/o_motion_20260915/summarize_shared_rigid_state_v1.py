"""Original pooled-count and full-support physical math for fixed Cpl/Fix.

Complete dev200 only; paired whole-scene bootstrap with all anchors/objects.
No metric/threshold/checkpoint selection. Physical EPE is a rigid-box proxy,
not measured dense scene flow, and is never attributed to O.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EVALUATOR_SHA='9785bb3082ac2c805ab54d291152391f0a4943f9948934ce9932cb4e530e9e6a'
COMMON_ARMS=('O','Cpl','Fix')
COMMON_PAIRS=(('Cpl','O'),('Fix','O'),('Cpl','Fix'))
PHYSICAL_ARMS=('D','CRN_CV','Cpl','Fix')
PHYSICAL_PAIRS=(('Cpl','CRN_CV'),('Fix','CRN_CV'),('Cpl','Fix'),('Cpl','D'),('Fix','D'),
                ('CRN_CV','D'),('Cpl','zero'),('Fix','zero'),('CRN_CV','zero'),('D','zero'))


def evaluation_module():
    path=Path(__file__).with_name('evaluate_shared_rigid_state_v1.py')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EVALUATOR_SHA:raise ValueError('Fixed evaluator changed')
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


evaluation=evaluation_module();endpoint=evaluation.endpoint
require,sha,read=evaluation.require,evaluation.sha,evaluation.read


def mathematics():
    contract=evaluation.pre._contract_loader()
    common=contract.import_bound('summarize_object_state_common_v1',evaluation.EXTRA)
    physical=contract.import_bound('summarize_future_state_physical_v1',evaluation.EXTRA)
    return common,physical


def calculate(rows,objects,selection,sparse_manifest):
    common,physical=mathematics()
    arms,scenes,sums=common.collect(rows,selection,arms=COMMON_ARMS)
    result=common.math_module().aggregate_counts(arms,scenes,sums,COMMON_PAIRS,10000,11)
    grouped,keys=physical.physical_rows(objects,{r['sample_token']:r['scene_token'] for r in rows},PHYSICAL_ARMS)
    require(len(keys)==16074,'All original object-horizon pairs required')
    support=physical.check_manifest(rows,grouped,sparse_manifest)
    for row in rows:
        require(set(row['physical_object_rows_by_arm'])==set(PHYSICAL_ARMS),'Physical arm counts missing')
        for arm in PHYSICAL_ARMS:
            require(row['physical_object_rows_by_arm'][arm]==sum(k[0]==row['sample_token'] for k in grouped[arm]),
                    'Physical original support missing')
    motion=physical.aggregate_physical(grouped,keys,PHYSICAL_ARMS,scenes,PHYSICAL_PAIRS,10000,11)
    return dict(schema='shared-rigid-joint-summary-v1',samples=200,scenes=100,scene_tokens=scenes,
        common=result,physical=motion,support=support,physical_rows_per_arm=len(keys),
        common_comparisons=[a+'-minus-'+b for a,b in COMMON_PAIRS],
        physical_comparisons=[a+'-minus-'+b for a,b in PHYSICAL_PAIRS],
        bootstrap=dict(unit='scene',paired=True,repetitions=10000,seed=11,preserve_all_anchors_and_objects=True,
            interval='percentile95',multiplicity_adjusted=False,training_seed_uncertainty_included=False),
        interpretation=dict(primary='four-future mean of pooled GMO IoU; not semantic mIoU',
            physical='mean point XY/XYZ EPE per original anchor-instance, then equal anchor-instance weight',
            full_support_includes_missed_detections_and_uncovered_points=True,future_out_of_ROI_targets_retained=True,
            no_native_O_flow=True,physical_labels_are_rigid_box_proxy=True,
            t0_and_FP_FN_and_motion_recall_and_transitions_all_retained=True,
            fixed512_single_seed11=True,historical_development_exposure=True,
            motion_improvement_cannot_be_inferred_from_GMO_alone=True,
            no_winner_or_full_validation_promotion_selected_by_this_script=True))


def authenticate(a):
    root=Path(a.evaluation_run);p=read(a.evaluation_protocol)
    require(not (root/'failed.json').exists() and sha(root/'complete.json')==a.evaluation_complete_sha256,
            'Named complete evaluation required')
    done=read(root/'complete.json')
    require(done['schema']==evaluation.SCHEMA and done['status']==evaluation.STATUS and done['evaluated_samples']==200
            and done['scenes']==100 and done['optimizer_updates']==0,'Incomplete fixed development evaluation')
    files=('manifest.json','loaded_models.json','records.jsonl','objects.jsonl','reference_checks.json','summary.json')
    endpoint.ledger(root,done['files_sha256'],files)
    require(p['schema']==evaluation.SCHEMA and p['status']=='FROZEN' and p['evaluation']==evaluation.EVALUATION and
            p['training_protocol_sha256']==sha(a.training_protocol) and
            p['analysis_sources_sha256']=={Path(__file__).name:sha(__file__)},'Frozen analysis contract differs')
    contract=evaluation.pre._contract_loader()
    for name,digest in p['sources_sha256'].items():require(sha(contract.source_path(name))==digest,'Producer source changed')
    manifest=read(root/'manifest.json');summary=read(root/'summary.json');loaded=read(root/'loaded_models.json')
    tm,ts,counts,training=endpoint.authenticate(a.training_run,done['fixed_training_complete_sha256'],a.training_protocol,
        a.selection,a.sparse_manifest)
    require(manifest['sources']['training']==training and manifest['sources']['protocol_sha256']==sha(a.evaluation_protocol) and
            manifest['sources']['sources_sha256']==p['sources_sha256'] and
            manifest['sources']['analysis_sources_sha256']==p['analysis_sources_sha256'] and
            manifest['loaded_models_sha256']==sha(root/'loaded_models.json'),'Training/evaluation provenance differs')
    require(summary['status']==evaluation.STATUS and summary['evaluated_samples']==200 and summary['scenes']==100 and
            summary['all_original_O_hist_exact'] and summary['all_common_GT_support_exact'] and
            summary['actual_final_checkpoint_tensors_checked'] and summary['all_model_states_unchanged'],'Missing actual evaluation checks')
    for arm in ('Cpl','Fix'):
        item=loaded['arms'][arm]
        require(item['actual_checkpoint_tensors_checked'] and item['optimizer_updates_performed']==0 and
                item['checkpoint_sha256']==training['final_checkpoints'][arm]['sha256'] and
                item['actual_tensor_digests']==dict(head=ts['final_future_head_sha256'][arm],new=ts['final_new_module_sha256'][arm]),
                'Loaded fixed endpoint differs')
        for group,values in item['optimizer_parameter_steps'].items():
            require(values=={k:(v if v else None) for k,v in counts[arm][group].items()},'Actual optimizer step proof differs')
    common,_=mathematics()
    require(sha(a.o_reference)==common.O_REFERENCE_SHA and sha(a.sparse_manifest)==common.SPARSE_MANIFEST_SHA and
            sha(a.raw_manifest)==common.RAW_MANIFEST_SHA,'Original reference/labels changed')
    rows=endpoint.jsonl(root/'records.jsonl');objects=endpoint.jsonl(root/'objects.jsonl');original=endpoint.jsonl(a.o_reference)
    require(len(rows)==len(original)==200,'Complete common sample coverage required')
    labels={r['identity']['sample_token']:r for r in read(a.sparse_manifest)['records']}
    for ordinal,(row,ref) in enumerate(zip(rows,original)):
        require(row['ordinal']==ordinal and row['sample_token']==ref['sample_token'] and row['scene_token']==ref['scene_token'] and
                row['hist_by_arm']['O']==ref['hist_by_horizon'] and row['sparse_label_sha256']==labels[row['sample_token']]['sha256'] and
                row['raw_label_sha256']==labels[row['sample_token']]['label_source_sha256'],'Original per-anchor evidence changed')
    receipt=dict(evaluation_complete_sha256=a.evaluation_complete_sha256,evaluation_files_sha256=done['files_sha256'],
        evaluation_protocol_sha256=sha(a.evaluation_protocol),training=training,actual_tensor_load_producer_verified=True,
        this_analysis_reloads_checkpoint_tensors=False)
    return rows,objects,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-run','evaluation-complete-sha256','evaluation-protocol','training-run','training-protocol',
                 'selection','sparse-manifest','raw-manifest','o-reference','out'):
        parser.add_argument('--'+name,required=True)
    a=parser.parse_args();out=Path(a.out);require(not out.exists(),'Do not overwrite analysis')
    rows,objects,receipt=authenticate(a);result=calculate(rows,objects,a.selection,a.sparse_manifest)
    result.update(endpoint_authentication=receipt,analysis_source_sha256=sha(__file__))
    with out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(samples=200,scenes=100,out=str(out))))


if __name__=='__main__':main()
