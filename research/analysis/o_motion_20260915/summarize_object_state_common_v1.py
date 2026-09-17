"""Completed V/G native dev200 counts with frozen scene-paired mathematics.

The CLI requires the complete fixed512 training/200-evaluation endpoint.
No partial-score path, threshold choice, checkpoint selection or tensor load.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType

import numpy as np

HERE=Path(__file__).resolve().parent
OLD_COMMON_SHA='23f6fc0a6d7a9c63a1e77f90ca94a2aedc72b68b82b201df4f34c6d3f3123e8a'
SELECTION_SHA='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
O_REFERENCE_SHA='3f426891f6811a595542a1e9694783521ae89cf18d6385896012e5f52a8e9b9a'
SPARSE_MANIFEST_SHA='cdb11231cbc3213d213e1f3f718fe9d3c90665dccea4154ecf3372ad1ce91efe'
RAW_MANIFEST_SHA='4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
ARMS=('O','V','G')
PAIRS=(('V','O'),('G','O'),('V','G'))
SCHEMA='object-state-forecast-training-v1'
STATUS='COMPLETE_OBJECT_STATE_FORECAST_TRAINING'
REPETITIONS=10000
SEED=11


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def jsonl(path):return [json.loads(x) for x in Path(path).read_bytes().splitlines() if x.strip()]


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


def source_path(name):
    require(Path(name).name==name and name.endswith('.py'),'Expected source basename')
    for path in (HERE/name,HERE.parent/'m0_improvement_20260915'/name):
        if path.is_file():return path
    raise FileNotFoundError(name)


def load_bound(name,digest):
    path=source_path(name+'.py');require(sha(path)==digest,'Frozen math source changed: '+name)
    if name in sys.modules:
        module=sys.modules[name]
        require(Path(module.__file__).resolve()==path and sha(module.__file__)==digest,'Wrong imported math source')
        return module
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def math_module():return load_bound('summarize_future_state_common_v1',OLD_COMMON_SHA)


def collect(rows,selection_path,arms=ARMS):
    """Exact old collector bytecode with only private model-name globals changed.

    The original module/globals and every input record remain unchanged.
    Alternate names exist only for real-old-record mathematics checks; CLI
    always requires O/V/G from authenticated completed training.
    """
    old=math_module()
    private=dict(old.collect.__globals__);private['ARMS']=tuple(arms)
    function=FunctionType(old.collect.__code__,private,'collect_object_state',old.collect.__defaults__)
    return function(rows,selection_path)


def verify_ledger(root,ledger,expected,skip=()):
    require(set(ledger)==set(expected),'Endpoint file ledger differs')
    for name,digest in ledger.items():
        require(valid_sha(digest) and '..' not in Path(name).parts and not Path(name).is_absolute(),'Invalid artifact ledger')
        if name not in skip:require(sha(Path(root)/name)==digest,'Endpoint artifact changed: '+name)


def authenticate_training(root,protocol_path,selection_path,o_reference_path,sparse_manifest_path,raw_manifest_path):
    """Read small authenticated endpoint artifacts; do not reload .pth tensors."""
    root=Path(root);require(not (root/'failed.json').exists(),'Failed/partial training is not analyzable')
    done=read(root/'complete.json');protocol=read(protocol_path)
    require(done['schema']==SCHEMA and done['status']==STATUS and done['mode']=='train'
            and done['updates']==512 and done['examples']==2048 and done['evaluated_samples']==200,
            'Need the complete fixed512 plus full200 endpoint')
    require(protocol['schema']==SCHEMA and protocol['status']=='FROZEN' and
            protocol['training']['seed']==11 and protocol['training']['updates']==512 and
            protocol['training']['examples']==2048 and protocol['training']['accumulate']==4 and
            protocol['training']['passes']==4 and protocol['training']['development_samples']==200,
            'Frozen formal experiment differs')
    verify_ledger(root,done['files_sha256'],('manifest.json','loaded_models.json','summary.json',
        'physical_gradient_diagnostics.jsonl','development_records.jsonl','development_objects.jsonl'))
    manifest=read(root/'manifest.json');summary=read(root/'summary.json');sources=manifest['sources']
    require(manifest['schema']==SCHEMA and manifest['mode']=='train' and manifest['training']==protocol['training']
            and manifest['numerical_policy']==protocol['numerical_policy'] and manifest['fresh_initialization'] is True
            and manifest['optimizer_restored'] is False,'Training manifest/initialization differs')
    require(sources['protocol_sha256']==sha(protocol_path) and sources['sources_sha256']==protocol['sources_sha256']
            and sources['analysis_sources_sha256']==protocol['analysis_sources_sha256'],'Frozen source chain differs')
    require(set(protocol['analysis_sources_sha256'])=={'summarize_object_state_common_v1.py','summarize_object_state_physical_v1.py'},
            'Both predeclared analyses required')
    for name,digest in {**protocol['sources_sha256'],**protocol['analysis_sources_sha256']}.items():
        require(valid_sha(digest) and sha(source_path(name))==digest,'Bound runtime/analysis source differs: '+name)
    require('object_state_forecast_train_v1.py' in protocol['sources_sha256'] and
            'object_state_conditioner_v2.py' in protocol['sources_sha256'],'Wrong trained architecture source')
    for key in ('preflight_protocol_sha256','engineering_preflight_complete_sha256','connected_protocol_sha256',
                'connected_complete_sha256','D_checkpoint_sha256','runtime_source_contract_sha256'):
        require(valid_sha(protocol[key]) and sources[key]==protocol[key],'Source/preflight provenance differs: '+key)
    require(sources['selection_sha256']==protocol['selection_sha256']==sha(selection_path)==SELECTION_SHA,
            'Fixed selection differs')
    require(sources['o_development_records_sha256']==protocol['o_development_records_sha256']==sha(o_reference_path)==O_REFERENCE_SHA,
            'Original O reference differs')
    for split,key in (('train','train_cache'),('development','development_cache')):
        # The producer receipt contains the complete original index identity.
        require(sources[key]['index_sha256']==protocol['cache_index_sha256'][split],'Cache index provenance differs')
    for side in ('train_predictions','development_predictions'):
        for key,value in protocol[side].items():
            require(sources[side][key]==value,'Prediction asset differs: '+side+'/'+key)
    require(summary['schema']==SCHEMA and summary['status']==STATUS and summary['updates']==512 and
            summary['examples']==2048 and summary['evaluated_samples']==200 and summary['fresh_initialization'] is True
            and summary['optimizer_restored'] is False and summary['frozen_D_and_non_head_O_unchanged'] is True,
            'Summary is not the same completed endpoint')
    require(set(done['final_checkpoints'])==set(done['arm_complete_sha256'])=={'V','G'},'Both final arms required')
    orders=np.random.RandomState(11);permutations=[orders.permutation(512).tolist() for _ in range(4)]
    order_sha=hashlib.sha256(json.dumps(permutations,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    # Match the frozen helper's serialization exactly below, without importing Torch.
    helper=load_bound('train_source_motion_v1',protocol['sources_sha256']['train_source_motion_v1.py'])
    order_sha=helper.json_hash(permutations)
    require(manifest['sample_orders_sha256']==order_sha,'Original four-permutation order differs')
    logs={};arm_artifacts={}
    for arm in ('V','G'):
        folder=root/'runs'/arm;ac=read(folder/'complete.json');am=read(folder/'manifest.json')
        require(sha(folder/'complete.json')==done['arm_complete_sha256'][arm] and ac['schema']==SCHEMA and
                ac['status']==STATUS and ac['arm']==arm and ac['updates']==512 and ac['examples']==2048
                and ac['evaluated_samples']==200,'Arm endpoint incomplete')
        verify_ledger(folder,ac['files_sha256'],('manifest.json','training.jsonl','final.pth','development_records.jsonl'),skip=('final.pth',))
        checkpoint=done['final_checkpoints'][arm]
        require(checkpoint['file']=='runs/'+arm+'/final.pth' and checkpoint['sha256']==ac['files_sha256']['final.pth'],
                'Fixed checkpoint declaration differs')
        for stem,key in (('future_head','future_head_state_sha256'),('readout','readout_state_sha256'),('conditioner','conditioner_state_sha256')):
            require(checkpoint[key]==ac['final_'+stem+'_sha256']==summary['final_'+stem+'_sha256'][arm],
                    'Final parameter state identity differs')
        require(am['schema']==SCHEMA and am['arm']==arm and am['common_manifest_sha256']==sha(root/'manifest.json')
                and am['sources']==sources and am['detach_physical_features'] is False
                and am['use_predicted_velocity']==(arm=='V'),'Arm input/gradient contract differs')
        logs[arm]=jsonl(folder/'training.jsonl');require(len(logs[arm])==512,'Missing matched update logs')
        flat=[]
        for u,item in enumerate(logs[arm],1):
            require(item['update']==u and item['examples']==4*u and len(item['samples'])==4,'Update count/accumulation differs')
            flat.extend(r['ordinal'] for r in item['samples'])
        require(flat==[i for order in permutations for i in order],'Actual logged sample order differs')
        state=summary['actual_optimizer_steps'][arm]
        require(set(state)=={'head','readout'} and state['head']['all_steps512'] is True,'Native head optimizer endpoint differs')
        for component,value in state.items():
            require(value['parameters_with_state']==len(value['steps']) and value['steps'] and
                    all(type(v) is int and 0<v<=512 for v in value['steps']), 'Invalid actual optimizer steps')
        arm_artifacts[arm]=dict(complete_sha256=sha(folder/'complete.json'),files_sha256=ac['files_sha256'])
    require(summary['actual_optimizer_steps']['V']==summary['actual_optimizer_steps']['G'],'Paired actual optimizer steps differ')
    equal_keys=('ordinal','sample_token','scene_token','official_index','split','inputs_sha256','targets_sha256','sparse_label_sha256',
                'input_tree_sha256','state_source','paired_rng_before_sha256','paired_rng_after_sha256')
    for l,r in zip(logs['V'],logs['G']):
        require(l['future_head_lr']==r['future_head_lr'] and l['readout_lr']==r['readout_lr'],'Matched LR differs')
        for a,b in zip(l['samples'],r['samples']):
            require(all(a[k]==b[k] for k in equal_keys),'Paired inputs/labels/state/RNG differ')
    require(sha(sparse_manifest_path)==SPARSE_MANIFEST_SHA and sha(raw_manifest_path)==RAW_MANIFEST_SHA,'Original label manifests differ')
    sparse={r['identity']['sample_token']:r for r in read(sparse_manifest_path)['records'] if r['identity']['split']=='development'}
    raw={r['identity']['sample_token']:r for r in read(raw_manifest_path)['records'] if r['identity']['split']=='development'}
    rows=jsonl(root/'development_records.jsonl');reference=jsonl(o_reference_path)
    require(len(rows)==len(reference)==len(sparse)==len(raw)==200,'Complete development coverage required')
    for ordinal,(row,original) in enumerate(zip(rows,reference)):
        require(row['ordinal']==ordinal and row['sample_token']==original['sample_token'] and
                row['scene_token']==original['scene_token'] and row['hist_by_arm']['O']==original['hist_by_horizon'],
                'Actual original O per-anchor confusion differs')
        sd=sparse[row['sample_token']];rd=raw[row['sample_token']]
        require(row['sparse_label_sha256']==sd['sha256'] and row['raw_label_sha256']==sd['label_source_sha256']==rd['sha256'],
                'Actual raw/sparse label identity differs')
    receipt=dict(complete_sha256=sha(root/'complete.json'),protocol_sha256=sha(protocol_path),
        manifest_sha256=sha(root/'manifest.json'),files_sha256=done['files_sha256'],arms=arm_artifacts,
        final_checkpoints=done['final_checkpoints'],bound_sources_sha256=protocol['sources_sha256'],
        analysis_sources_sha256=protocol['analysis_sources_sha256'],actual_logged_orders_sha256=order_sha,
        final_checkpoint_bytes_rehashed=False,checkpoint_tensor_load_performed=False,
        checkpoint_scope='producer final state/checkpoint declarations cross-linked; this CPU analysis rehashes all small artifacts and recomputes raw counts',
        selection_sha256=SELECTION_SHA,sparse_manifest_sha256=SPARSE_MANIFEST_SHA,raw_manifest_sha256=RAW_MANIFEST_SHA)
    return rows,receipt


def summarize(rows,selection_path):
    arms,scenes,sums=collect(rows,selection_path)
    old=math_module();result=old.aggregate_counts(arms,scenes,sums,PAIRS,REPETITIONS,SEED)
    result.update(schema='object-state-common-pooled-summary-v1',samples=200,scenes=100,scene_tokens=scenes,
        arms=list(arms),comparisons_order=[a+'-minus-'+b for a,b in PAIRS],
        source_contract=dict(original_common_sha256=OLD_COMMON_SHA,v3_math_sha256=old.V3_SHA,
            common_metric_sha256=old.METRIC_SHA,selection_sha256=SELECTION_SHA,private_collector_names_only=True),
        protocol=dict(primary='four-future mean of per-horizon pooled GMO IoU; original task unchanged',
            pooling='sum integer confusion/GT/TP/FN before ratios; never average sample IoU',
            all_t0_horizons_global_FP_FN_eight_groups_transitions_retained=True,
            t0_prediction_policy='model_specific_native; same GT and each arm self-consistency',
            transition_interpretation='both predicted endpoints can change; not isolated physical-motion accuracy',
            bootstrap=dict(unit='scene',preserve_all_anchors_within_scene=True,paired=True,repetitions=10000,seed=11,
                interval='percentile95',multiplicity_adjusted=False),
            no_training_seed_uncertainty=True,historical_development_exposure=True,no_candidate_selection=True,
            no_native_flow_claim_for_O=True,zero_denominator='null; never a perfect empty score'))
    return result


def add_common_arguments(parser):
    for name in ('training-run','protocol','selection','o-reference','sparse-manifest','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--raw-manifest',default=str(HERE/'motion_targets_v1/manifest.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_common_arguments(parser);a=parser.parse_args()
    output=Path(a.out);require(not output.exists(),'Do not overwrite prior analysis')
    rows,receipt=authenticate_training(a.training_run,a.protocol,a.selection,a.o_reference,a.sparse_manifest,a.raw_manifest)
    result=summarize(rows,a.selection);result['endpoint_authentication']=receipt;result['analysis_source_sha256']=sha(__file__)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(samples=200,scenes=100,arms=list(ARMS),out=str(output))))


if __name__=='__main__':main()
