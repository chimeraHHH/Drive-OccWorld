"""Completed V/G versus fixed D, CRN-CV v2 and zero on full dev200 support.

Errors are means over each original object's virtual rigid source points,
then object-anchor pairs have equal weight. Never select detected objects.
Only mean differences receive scene-paired intervals; median/p90 descriptive.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
COMMON_SHA='4a03aeda837e78b97ff18d27140d08602619ee96502b7f62fa82f591ca08f584'
OLD_PHYSICAL_SHA='920f4f96e0bd5df247b3d096bdf1709d80fed593622b70373093ea1160db2de3'
D_COMPLETE_SHA='f0bf075fa36622b38c0c5136b187a184d16d3e07ed00857dfe21a0545f1c5f56'
D_OBJECTS_SHA='b73499b8e4ce1bd0de6ffc3ff6001d3a58ab9f33483c8bfb8f7ed68e6f88d0ea'
D_PROTOCOL_SHA='e933d234c1adf49941a903f12483985eba9bde2393172a3a114e95343aeec259'
D_CHECKPOINT_SHA='7e2750d56ade7fb36e334b6431e83f51aafa997357744780a507924a696951e3'
CRN_COMPLETE_SHA='edfdf0b38145ad0b1d5fb69370ba8e8d91fda973f33540c1f5149917f58077f6'
CRN_SUMMARY_SHA='ba67afc75b02ef481b0846914207fc01c6d04a4d0c675d3a7c00d1129b81141e'
CRN_SCORER_SHA='0de90351e60330cc1a9208e3233f8b1077614ce69695ff044dd01f2f488516b5'
ARMS=('D','V','G','CRN_CV')
PAIRS=(('V','D'),('G','D'),('V','G'),('V','CRN_CV'),('G','CRN_CV'),('CRN_CV','D'),
       ('V','zero'),('G','zero'),('D','zero'),('CRN_CV','zero'))


def common_module():
    path=HERE/'summarize_object_state_common_v1.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=COMMON_SHA:raise ValueError('New completed-endpoint source changed')
    name='summarize_object_state_common_v1'
    if name in sys.modules:
        result=sys.modules[name]
        if Path(result.__file__).resolve()!=path:raise ValueError('Wrong common analysis imported')
        return result
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result;spec.loader.exec_module(result);return result


def reference_objects(D_root,CRN_root):
    common=common_module();require,sha,read=common.require,common.sha,common.read
    D_root=Path(D_root);CRN_root=Path(CRN_root)
    require(sha(D_root/'complete.json')==D_COMPLETE_SHA,'Actual fixed D completion differs')
    done=read(D_root/'complete.json')
    require(done['updates']==512 and done['examples']==2048 and done['evaluated_samples']==200 and
            done['final_checkpoints']['D']['sha256']==D_CHECKPOINT_SHA,'Actual D fixed endpoint differs')
    require(sha(D_root/'development_objects.jsonl')==done['files_sha256']['development_objects.jsonl']==D_OBJECTS_SHA,
            'Original D per-object record differs')
    require(sha(D_root/'manifest.json')==done['files_sha256']['manifest.json'],'Original D manifest differs')
    manifest=read(D_root/'manifest.json');require(manifest['sources']['protocol_sha256']==D_PROTOCOL_SHA,'D training protocol differs')
    D=[r for r in common.jsonl(D_root/'development_objects.jsonl') if r['arm']=='D']
    require(len(D)==16074,'Original D complete source support differs')
    require(sha(CRN_root/'complete.json')==CRN_COMPLETE_SHA,'Fixed CRN-CV v2 completion differs')
    complete=read(CRN_root/'complete.json')
    require(complete['schema']=='predicted-object-state-cv-complete-v2' and
            complete['status']=='COMPLETE_READ_ONLY_DIAGNOSTIC' and complete['samples']==200 and complete['scenes']==100
            and complete['script_sha256']==CRN_SCORER_SHA,'Wrong CRN diagnostic/version')
    common.verify_ledger(CRN_root,complete['files_sha256'],('summary.json','report.md'))
    require(sha(CRN_root/'summary.json')==CRN_SUMMARY_SHA,'Fixed CRN-CV v2 records changed')
    summary=read(CRN_root/'summary.json')
    require(summary['schema']=='predicted-object-state-cv-diagnostic-v2' and
            summary['status']=='COMPLETE_READ_ONLY_DIAGNOSTIC' and summary['samples']==200 and summary['scenes']==100 and
            summary['optimizer_updates']==0 and summary['model_forward_performed'] is False,'Wrong CRN state diagnostic endpoint')
    src=summary['sources']
    require(src['analysis_sha256']==CRN_SCORER_SHA and src['D_complete_sha256']==D_COMPLETE_SHA and
            src['D_objects_sha256']==D_OBJECTS_SHA and src['D_protocol_sha256']==D_PROTOCOL_SHA,
            'CRN and original D support sources differ')
    require(sha(common.source_path('predicted_object_state_cv_diagnostic_v2.py'))==CRN_SCORER_SHA,'Frozen CRN scorer source changed')
    for name,digest in src['dependencies_sha256'].items():require(sha(common.source_path(name))==digest,'CRN prior source changed: '+name)
    CRN=[r for r in summary['physical_object_records'] if r['arm']=='CRN_CV']
    embedded_D=[r for r in summary['physical_object_records'] if r['arm']=='D']
    require(len(CRN)==len(embedded_D)==16074 and len(summary['physical_object_records'])==32148,'CRN full source support differs')
    return D,CRN,embedded_D,summary,dict(D_complete_sha256=D_COMPLETE_SHA,D_objects_sha256=D_OBJECTS_SHA,
        D_checkpoint_sha256=D_CHECKPOINT_SHA,D_protocol_sha256=D_PROTOCOL_SHA,CRN_complete_sha256=CRN_COMPLETE_SHA,
        CRN_summary_sha256=CRN_SUMMARY_SHA,CRN_scorer_sha256=CRN_SCORER_SHA,
        CRN_dependencies_sha256=src['dependencies_sha256'],source_scope='existing completed D tensor-load and corrected CRN-CV records; no tensor/model rerun')


def exact_reference_D(left,right):
    common=common_module()
    key=lambda r:(r['sample_token'],r['instance_token'],r['horizon_seconds'])
    a={key(r):r for r in left};b={key(r):r for r in right}
    common.require(len(a)==len(left)==len(b)==len(right)==16074 and set(a)==set(b),'D object/horizon identity differs')
    fields=('sample_token','scene_token','instance_token','horizon_seconds','dt_seconds','group','source_points',
            'epe_xy_m','epe_3d_m','zero_epe_xy_m','zero_epe_3d_m')
    for k in a:common.require(all(a[k][f]==b[k][f] for f in fields),'Frozen D object result changed: '+str(k))


def summarize(rows,objects,selection_path,sparse_manifest_path,D_root,CRN_root):
    common=common_module();old=common.load_bound('summarize_future_state_physical_v1',OLD_PHYSICAL_SHA)
    _,scenes,_=common.collect(rows,selection_path)
    common.require(set(r['arm'] for r in objects)=={'D','V','G'},'Completed physical outputs must be D/V/G')
    D,CRN,embedded_D,CRN_summary,reference_receipt=reference_objects(D_root,CRN_root)
    current_D=[r for r in objects if r['arm']=='D'];exact_reference_D(current_D,D);exact_reference_D(embedded_D,D)
    anchor_index={r['sample_token']:r['ordinal'] for r in rows}
    for r in objects:common.require(r['ordinal']==anchor_index[r['sample_token']],'Physical ordinal differs')
    grouped,keys=old.physical_rows([*objects,*CRN],{r['sample_token']:r['scene_token'] for r in rows},ARMS)
    common.require(len(keys)==16074,'All original object-horizon support required')
    support=old.check_manifest(rows,grouped,sparse_manifest_path)
    common.require(support==CRN_summary['support'],'Frozen CRN and current sparse-support summaries differ')
    counts={a:{} for a in ('D','V','G')}
    for r in objects:counts[r['arm']][r['sample_token']]=counts[r['arm']].get(r['sample_token'],0)+1
    for row in rows:
        common.require(set(row['physical_object_rows_by_arm'])==set(counts),'Per-anchor physical arm count differs')
        for a in counts:common.require(row['physical_object_rows_by_arm'][a]==counts[a].get(row['sample_token'],0),'Missing per-anchor physical rows')
    result=old.aggregate_physical(grouped,keys,ARMS,scenes,PAIRS,repetitions=10000,seed=11)
    result.update(schema='object-state-physical-pooled-summary-v1',anchors=200,scenes=100,scene_tokens=scenes,
        arms=list(ARMS),physical_rows_per_arm=len(keys),support=support,
        comparisons_order=[a+'-minus-'+b for a,b in PAIRS],reference_authentication=reference_receipt,
        CRN_CV_coverage=CRN_summary['coverage'],
        source_contract=dict(common_summarizer_sha256=COMMON_SHA,old_physical_math_sha256=OLD_PHYSICAL_SHA,
            original_mean_bootstrap_math_sha256=old.PRIOR_SHA,selection_sha256=common.SELECTION_SHA,
            sparse_manifest_sha256=old.SPARSE_MANIFEST_SHA),
        bootstrap=dict(repetitions=10000,seed=11,unit='scene',keep_all_anchors_objects=True,paired=True,
            interval='percentile95',multiplicity_adjusted=False),
        contract=dict(physical='mean point XY/XYZ EPE per original object, then equal anchor-instance weight; not voxel/scene weighted',
            physical_labels='original current unique-box virtual material points with rigid-box future labels; not observed dense scene flow',
            primary_scope='all original valid object-horizon pairs including CRN-uncovered points and missed detections',
            CRN_CV='fixed v2: corrected current boxes, highest original score ownership, uncovered velocity zero; nominal h times predicted velocity',
            all_models_same_source_label_zero_group_dt_contract=True,D_per_object_equals_original_complete=True,
            CRN_CV_is_a_deployable_state_baseline_not_O_native_flow=True,CRN_extra_detector_history_pretraining_and_cost_not_controlled_by_D=True,
            V_G_same_CRN_geometry_capacity_only_velocity_differs=True,O_has_no_native_flow=True,EPE_lower_is_better=True,
            zero_denominator='null, never zero error',median_p90='descriptive point estimates; mean differences receive paired scene intervals',
            single_training_seed=11,bootstrap_seed_not_additional_training=True,historical_development_exposure=True,
            model_selected=False,future_out_of_ROI_targets_retained=True))
    return result


def main():
    common=common_module();parser=argparse.ArgumentParser(description=__doc__);common.add_common_arguments(parser)
    parser.add_argument('--D-training-run',dest='D_training_run',required=True)
    parser.add_argument('--crn-cv-root',required=True);a=parser.parse_args()
    output=Path(a.out);common.require(not output.exists(),'Do not overwrite earlier analysis')
    rows,receipt=common.authenticate_training(a.training_run,a.protocol,a.selection,a.o_reference,a.sparse_manifest,a.raw_manifest)
    objects=common.jsonl(Path(a.training_run)/'development_objects.jsonl')
    result=summarize(rows,objects,a.selection,a.sparse_manifest,a.D_training_run,a.crn_cv_root)
    result['endpoint_authentication']=receipt;result['analysis_source_sha256']=common.sha(__file__)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(anchors=200,scenes=100,arms=list(ARMS),out=str(output))))


if __name__=='__main__':main()
