"""Compare completed material arm to authenticated CRN-CV on same source labels.

Uses XYZ EPE for both. No models or tensor predictions read. Old reference
uses higher-precision norms; bound the zero-target roundoff explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def run(arm):
    n=Path(__file__).parent
    old=n/'server_results/training/shared_rigid_fixed_dev200_v1'
    new=n/'server_results/training/dense_material512_v1'/f'{arm}_evaluate'
    previous=read(old/'complete.json'); current=read(new/'complete.json')
    assert previous['status']=='COMPLETE_SHARED_RIGID_FIXED_DEV200'
    assert current['status']=='COMPLETE_MATERIAL512_DEV200' and current['arm']==arm
    assert previous['files_sha256']['objects.jsonl']=='f3d14584dcf22dcdf12d7d020365ac85541a118e61460a61394c86322f871620'
    for root,done,names in [(old,previous,('manifest.json','objects.jsonl','records.jsonl')),(new,current,('protocol.json','records.jsonl'))]:
        for name in names: assert sha(root/name)==done['files_sha256'][name]
    source=read(old/'manifest.json')['sources']; protocol=read(new/'protocol.json')
    assert source['sparse_labels']==protocol['labels']
    assert source['development_cache']['index_sha256']==protocol['cache_index_sha256']['development']
    objects=[r for r in lines(old/'objects.jsonl') if r['arm']=='CRN_CV']
    assert len(objects)==16074
    reference={r['sample_token']:r for r in lines(old/'records.jsonl')}
    rows=lines(new/'records.jsonl'); assert len(rows)==200
    scenes=sorted({r['scene_token'] for r in rows}); assert len(scenes)==100
    scene_index={s:i for i,s in enumerate(scenes)}
    keys={}
    for row in objects:
        key=(row['sample_token'],row['horizon_seconds'],row['group'])
        keys.setdefault(key,[]).append(row)
    weights=np.random.RandomState(11).multinomial(100,np.full(100,.01),size=10000)
    comparisons=[]; max_zero_gap=0.; checked=0
    for h in range(4):
        for group in ('stationary','ambiguous','moving'):
            counts=np.zeros((100,3),dtype=np.float64)
            for row in rows:
                ref=reference[row['sample_token']]
                for ownkey,oldkey in [('input_sha256','inputs_sha256'),('GT_cache_sha256','targets_sha256'),('raw_label_sha256','raw_label_sha256')]:
                    assert row[ownkey]==ref[oldkey]
                assert row['scene_token']==ref['scene_token']
                cv=keys.get((row['sample_token'],(h+1)*.5,group),[])
                own=row['physical'][h]['groups'][group]
                assert own['objects']==len(cv) and own['points']==sum(r['source_points'] for r in cv)
                assert all(r['scene_token']==row['scene_token'] for r in cv)
                if not cv: continue
                zero=sum(r['zero_epe_3d_m'] for r in cv)/len(cv)
                gap=abs(zero-own['zero_epe_object_sum']/len(cv))
                assert gap<1e-5
                max_zero_gap=max(max_zero_gap,gap);checked+=1
                counts[scene_index[row['scene_token']]] += [own['epe_object_sum'],sum(r['epe_3d_m'] for r in cv),len(cv)]
            totals=counts.sum(0); bootstrap=weights@counts
            delta=(bootstrap[:,0]-bootstrap[:,1])/bootstrap[:,2]
            comparisons.append(dict(endpoint_index=h+1,nominal_horizon_seconds=(h+1)*.5,group=group,
                objects=int(totals[2]),candidate_EPE_XYZ_m=float(totals[0]/totals[2]),CRN_CV_EPE_XYZ_m=float(totals[1]/totals[2]),
                candidate_minus_CRN_CV_m=float((totals[0]-totals[1])/totals[2]),scene95CI_m=np.quantile(delta,[.025,.975]).tolist()))
    result=dict(status='PASS_SAME_SOURCE_PHYSICAL_COMPARISON',arm=arm,source_sha256=sha(Path(__file__)),
        reference_objects_sha256=sha(old/'objects.jsonl'),candidate_records_sha256=sha(new/'records.jsonl'),
        same_sparse_label_manifest=True,same_development_index=True,all_anchor_group_object_and_point_counts_exact=True,
        nonempty_anchor_horizon_group_checks=checked,max_zero_target_mean_norm_roundoff_m=max_zero_gap,roundoff_bound_m=1e-5,
        comparisons=comparisons,bootstrap=dict(seed=11,replicates=10000,unit='scene',scope='exposed dev200; not seed variance'),
        limitation='Aggregate source-label match plus authenticated common producers; candidate records do not contain per-object IDs/predictions. No new O physical head or held-out improvement claim.')
    (n/f'dense_material512_{arm}_physical_reference_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],max_zero_roundoff=max_zero_gap,final_endpoint=comparisons[-3:]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['T','J','D'],required=True)
    run(parser.parse_args().arm)
