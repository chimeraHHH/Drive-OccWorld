"""CPU-only posthoc clip trajectory audit after the strict F/O final aggregate.

Reads complete receipts/logs and reuses the frozen aggregate's actual CPU
checkpoint/order proof. It does NOT reopen .pth, load torch, evaluate a model,
read GT arrays, choose thresholds, remove samples, or alter any training state.
C/F/O same-update comparisons are different model/moment trajectories.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

PARENT_SHA = '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a'
OBJECTIVE_SHA = '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d'
INITIAL_SHA = '6e20f6ab75edc9f682f5bf63d7396f6081e2d447b475c6543fe42cda3d837a60'
RARE = '706d7015b80d4329b80d4a8c52139ab4'
RARE_UPDATES = [67, 229, 275, 395]
ARMS = ('C', 'F', 'O')


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
    require(expected is None or sha(path) == expected, 'SHA differs: '+str(path))
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = Path(path).with_suffix('.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def make_groups(train):
    require(len(train) == 512 and len({r['sample_token'] for r in train}) == 512 and
            all(r['split'] == 'train' for r in train), 'Expected exact train512 selection')
    orders = []; groups = []; rng = np.random.RandomState(11)
    identity = ('sample_token','scene_token','official_index','split')
    for epoch in range(4):
        order = rng.permutation(512).tolist(); orders.append(order)
        for index in range(128):
            members = [{k:train[j][k] for k in identity} for j in order[index*4:index*4+4]]
            groups.append(dict(update=epoch*128+index+1, pass_index=epoch+1,
                members=members, contains_fixed_rare=any(r['sample_token']==RARE for r in members)))
    require([r['update'] for r in groups if r['contains_fixed_rare']] == RARE_UPDATES, 'Fixed rare groups changed')
    digest = hashlib.sha256(json.dumps(orders,separators=(',',':')).encode()).hexdigest()
    return groups, digest


def expected_lr(index):
    factor = (.1+.9*index/50) if index<50 else (.1+.9*.5*(1+math.cos(math.pi*(index-50)/(512-50))))
    return 1e-5*factor


def summarize_log(rows, groups):
    require(len(rows) == 512, 'Only complete 512-update logs are accepted')
    for index,row in enumerate(rows):
        require(row['update']==index+1 and row['pass_index']==index//128+1 and row['examples']==4*(index+1),
                'Update/pass/exposure sequence differs')
        require(math.isclose(row['lr'],expected_lr(index),rel_tol=1e-14,abs_tol=0), 'Native LR differs')
        require(all(math.isfinite(row[k]) for k in ('grad_norm','loss_mean','loss_min','loss_max')) and row['grad_norm']>=0,
                'Nonfinite/negative gradient statistic')
    norm=np.array([r['grad_norm'] for r in rows],dtype=np.float64)
    factors=np.minimum(1.,35./(norm+1e-6))
    records=[]
    for index,row in enumerate(rows):
        values=norm[(index//128)*128:(index//128+1)*128]
        records.append(dict(update=index+1,pass_index=index//128+1,preclip_gradient_L2=float(norm[index]),
            clip_factor_float64_from_serialized_norm=float(factors[index]), norm_exceeds_35=bool(norm[index]>35),
            factor_less_than_one=bool(factors[index]<1),
            descending_rank_within_pass=1+int((values>norm[index]).sum()),
            lower_fraction_within_pass=float((values<norm[index]).mean()),
            contains_fixed_rare=groups[index]['contains_fixed_rare']))
    q=[0.,.25,.5,.75,.9,.95,.99,1.]
    def distribution(ids):
        return dict(updates=len(ids),norm_exceeds_35=int(np.sum(norm[ids]>35)),
            factor_less_than_one=int(np.sum(factors[ids]<1)),
            preclip_gradient_L2_quantiles={str(v):float(x) for v,x in zip(q,np.quantile(norm[ids],q))},
            clip_factor_quantiles={str(v):float(x) for v,x in zip(q,np.quantile(factors[ids],q))})
    return dict(all_updates=distribution(np.arange(512)),
        passes=[dict(pass_index=i+1,**distribution(np.arange(i*128,(i+1)*128))) for i in range(4)],
        update_records=records, fixed_rare_groups=[records[i-1] for i in RARE_UPDATES])


def load_runs(a, protocol, parent, certified, order_sha):
    manifests={}; logs={}; receipts={}
    for arm in ARMS:
        name='native1' if arm=='C' else arm
        directory=Path(a.control_run) if arm=='C' else Path(a.runs_root)/arm
        require(not (directory/'failed.json').exists() and not (directory/'interrupted.pth').exists(), 'Failed or interrupted arm: '+arm)
        proof=certified['source_receipts'][name]
        bindings=protocol['control_files_sha256'] if arm=='C' else proof['files_sha256']
        files={key:sha(directory/key) for key in ('manifest.json','training_complete.json','complete.json','training.jsonl')}
        require(all(files[k]==bindings[k] for k in files), 'Final source/log receipt mismatch: '+arm)
        manifest,trained,done=[read(directory/k) for k in ('manifest.json','training_complete.json','complete.json')]
        require(done['status']=='TRAINED_AND_DEVELOPMENT_EVALUATED' and trained['status']=='TRAINED_FIXED_FINAL', 'Incomplete final arm')
        require(all(r['updates']==512 and r['examples']==2048 for r in (trained,done)), 'Wrong fixed final budget')
        require(manifest['arm']==name and manifest['seed']==11 and manifest['initial_head_sha256']==INITIAL_SHA and
                manifest['m0_sha256']==parent['m0_sha256'] and manifest['passes']==4 and manifest['examples_per_pass']==512,
                'Wrong initialization/exposure: '+arm)
        require(manifest['protocol_sha256']==(PARENT_SHA if arm=='C' else OBJECTIVE_SHA) and
                manifest['numerical_policy']==parent['numerical_policy'] and manifest['migration']['mode']=='native1', 'Wrong native protocol/precision')
        for role,key in [('train','train_index_sha256'),('development','dev_index_sha256')]:
            require(manifest[key]==protocol['cache_index_sha256'][role], 'Wrong cache binding')
        if arm=='C':
            require(proof['manifest_sha256']==files['manifest.json'] and proof['complete_sha256']==files['complete.json'] and
                    proof['training_complete_sha256']==files['training_complete.json'], 'C aggregate receipts differ')
        else:
            require(done['manifest_sha256']==files['manifest.json'] and done['training_log_sha256']==files['training.jsonl'] and
                    done['objective_protocol_sha256']==OBJECTIVE_SHA and manifest['parent_protocol_sha256']==PARENT_SHA and
                    manifest['source_sha256']==protocol['source_sha256'], 'Candidate producer chain differs')
        require(proof['training']['training_log_sha256']==files['training.jsonl'] and
                proof['training']['updates']==512 and proof['training']['all_update_LRs_match_original'] is True,
                'Certified full log proof differs')
        payload=proof['checkpoint']
        require(payload['actual_payload_checked_on_CPU'] is True and payload['checkpoint_rehashed'] is True and
                payload['sample_orders_sha256']==order_sha and payload['updates']==512 and payload['passes']==4 and
                payload['checkpoint_sha256']==done['checkpoint_sha256']==trained['checkpoint_sha256']==bindings['latest.pth'],
                'Certified final checkpoint/order proof differs')
        if arm!='C':
            require(payload['final_head_state_sha256']==done['final_head_state_sha256'], 'Final tensor state receipt differs')
        logs[arm]=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines() if line.strip()]
        manifests[arm]=manifest
        receipts[arm]=dict(directory=str(directory.resolve()),files_sha256=files,
            reused_checkpoint_payload_receipt=payload, checkpoint_reopened_by_this_script=False)
    for arm in ('F','O'):
        for key in ('initial_head_sha256','trainable_parameters','trainable_parameter_names','numerical_policy','migration'):
            require(manifests[arm][key]==manifests['C'][key], 'C/candidate initialization or scope differs')
        for key in ('optimizer_groups','tensor_shapes','sample_orders_sha256'):
            require(receipts[arm]['reused_checkpoint_payload_receipt'][key]==receipts['C']['reused_checkpoint_payload_receipt'][key],
                    'Certified optimizer/parameter/order contract differs')
    return logs,receipts


def run(a,out):
    parent=read(a.protocol,PARENT_SHA); protocol=read(a.objective_protocol,OBJECTIVE_SHA)
    require(protocol['schema']=='objective-supervision-training-protocol-v1' and protocol['status']=='FROZEN_BEFORE_F_O_TRAINING' and
            protocol['parent_protocol_sha256']==PARENT_SHA and protocol['arms']==['F','O'], 'Wrong frozen objective protocol')
    for key in ('training','numerical_policy','m0_sha256','config_sha256','selection_sha256'):
        require(protocol[key]==parent[key], 'Parent contract differs')
    sources=dict(parent['source_sha256'],**protocol['source_sha256'])
    for name,digest in sources.items():
        require(Path(name).name==name and sha(Path(__file__).with_name(name))==digest, 'Frozen producing source changed')
    selected=read(a.selection,protocol['selection_sha256'])
    groups,order_sha=make_groups([r for r in selected['records'] if r['split']=='train'])
    summary_dir=Path(a.audited_summary)
    require(not (summary_dir/'failed.json').exists(), 'Final aggregate failed')
    done=read(summary_dir/'complete.json')
    require(done['status']=='COMPLETE' and done['schema']=='m0-objective-supervision-aggregate-complete-v1' and
            done['objective_protocol_sha256']==OBJECTIVE_SHA, 'Require complete frozen objective aggregate')
    certified=read(summary_dir/'summary.json',done['summary_sha256'])
    require(done['files_sha256']['summary.json']==done['summary_sha256'] and certified['schema']=='m0-objective-supervision-aggregate-v1' and
            certified['status']=='COMPLETED_DEVELOPMENT_COMPARISON' and certified['objective_protocol_sha256']==OBJECTIVE_SHA and
            certified['parent_protocol_sha256']==PARENT_SHA and certified['script_sha256']==protocol['source_sha256']['objective_supervision_aggregate.py'],
            'Frozen actual aggregate provenance differs')
    require(certified['audit']['status']=='PASS' and certified['audit']['actual_three_final_checkpoints_rehashed_and_payload_checked_on_CPU'] is True and
            certified['audit']['actual_512_updates_2048_examples_LR_and_orders_checked'] is True and
            certified['audit']['same_weight_structure_and_M0_initialization'] is True, 'Missing final payload/order audit')
    logs,receipts=load_runs(a,protocol,parent,certified,order_sha)
    models={arm:summarize_log(logs[arm],groups) for arm in ARMS}
    result=dict(schema='m0-objective-clip-diagnostic-v1',status='PASS_COMPLETE_LOG_DIAGNOSTIC',
        script_sha256=sha(__file__),objective_protocol_sha256=OBJECTIVE_SHA,parent_protocol_sha256=PARENT_SHA,
        selection_sha256=protocol['selection_sha256'],sample_orders_sha256=order_sha,
        certified_summary_sha256=done['summary_sha256'],certified_complete_sha256=sha(summary_dir/'complete.json'),
        receipts=receipts,groups=groups,models=models,fixed_rare_token=RARE,fixed_rare_updates=RARE_UPDATES,
        scope=dict(CPU_only=True,torch_loaded=False,GT_or_prediction_arrays_read=False,new_training_or_forward=False,
            checkpoint_payload_proof='reused SHA-bound frozen aggregate actual CPU audit; checkpoints not reopened here',
            statistics='independently computed from complete original logs and frozen selection/order',
            clip_factor='float64 estimate from serialized preclip norm; not a bitwise CUDA replay',
            causal_limit='same groups/update/LR but distinct evolving weights and AdamW moments; not same-state loss ablation',
            single_rare_contribution_identified=False,AdamW_displacement_reconstructed=False,
            samples_removed=False,performance_selection=False,training_seeds=1))
    write(out/'summary.json',result)
    fields=('arm','update','pass_index','preclip_gradient_L2','clip_factor_float64_from_serialized_norm',
            'norm_exceeds_35','factor_less_than_one','descending_rank_within_pass','lower_fraction_within_pass','contains_fixed_rare','group_tokens')
    for filename,rare_only in [('all_updates.csv',False),('fixed_rare_groups.csv',True)]:
        with (out/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for arm,model in models.items():
                for row in model['update_records']:
                    if rare_only and not row['contains_fixed_rare']:continue
                    writer.writerow(dict(arm=arm,**row,group_tokens='|'.join(r['sample_token'] for r in groups[row['update']-1]['members'])))
    lines=['# 固定累积组裁剪轨迹：只读 CPU 诊断','','完整 C/F/O 最终日志均为 512 更新；本表不改变数据、训练或晋级标准。','',
        '| 臂 | norm > 35 次数/512 | 总体范数中位数 | 总体范数 p99 |','|---|---:|---:|---:|']
    for arm,model in models.items():
        r=model['all_updates'];lines.append('| {} | {} | {:.4f} | {:.4f} |'.format(arm,r['norm_exceeds_35'],r['preclip_gradient_L2_quantiles']['0.5'],r['preclip_gradient_L2_quantiles']['0.99']))
    lines+=['','| 臂 | rare 所在 update | 组裁剪前范数 | 因子估计 | 本 pass 降序名次 |','|---|---:|---:|---:|---:|']
    for arm,model in models.items():
        for r in model['fixed_rare_groups']:
            lines.append('| {} | {} | {:.4f} | {:.6f} | {} |'.format(arm,r['update'],r['preclip_gradient_L2'],r['clip_factor_float64_from_serialized_norm'],r['descending_rank_within_pass']))
    lines+=['','rare 仅为固定事后诊断身份，所在组还包括另外三个真实样本；此表不能把整个组范数归于单个样本。各臂同组/同更新号但权重及 AdamW moments 已不同，不是同状态切换 loss 的实验。clip 因子由保存标量近似重算；不重建 AdamW 位移，不作四点显著性或训练 seed 稳健性声明。',
        'checkpoint 实际内容、精确四轮 sample orders 与 optimizer 合同复用 SHA 绑定的最终聚合审计；本脚本不再次读取 pth、GT、预测或运行模型。详见 summary.json 的 proof scope 与全部组身份。','']
    (out/'report.md').write_text('\n'.join(lines))
    files={name:sha(out/name) for name in ('summary.json','all_updates.csv','fixed_rare_groups.csv','report.md')}
    write(out/'complete.json',dict(status=result['status'],files_sha256=files,summary_sha256=files['summary.json'],
        objective_protocol_sha256=OBJECTIVE_SHA,updates_per_model=512,models=list(ARMS),CPU_only=True))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('objective-protocol','protocol','control-run','runs-root','audited-summary','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--selection')
    a=p.parse_args(argv)
    if a.selection is None:a.selection=str(Path(a.protocol).with_name('selection_v1.json'))
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:run(a,out)
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_CLOSED_NO_PARTIAL_ANALYSIS',error=repr(exc),seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main()
