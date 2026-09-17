"""Fixed CPU evaluation and independent recomputation for dense-history DELTA.

Prediction archives must already be complete. Never trains or changes models.
Refuses to overwrite any prior evaluation directory or audit receipt.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

N=Path(__file__).resolve().parent
D=N/'server_results/diagnostics'


def run(filename,*args):
    subprocess.run([sys.executable,'-B',str(N/filename),*map(str,args)],check=True)


def main():
    prereg=json.loads((N/'delta_dense_history_scoring_preregistration_v1.json').read_text())
    assert hashlib.sha256((N/'evaluate_history_tracker_v1.py').read_bytes()).hexdigest()==prereg['source_sha256']
    run('audit_delta_dense_history_geometry_v1.py',
        '--dense',D/'delta_dense_history_train16_v1',
        '--twoframe',D/'delta_history_train16_v1/delta',
        '--history',D/'dense_history_inputs_train16_v1',
        '--out',N/'delta_dense_history_geometry_audit_v1.json')
    for method in ('endpoint','lsq'):
        result=N/f'delta_dense_{method}_train16_evaluation_v1'
        run('evaluate_history_tracker_v1.py',
            '--surface',D/f'delta_dense_history_train16_v1/delta_dense_{method}',
            '--out',result)
        run('audit_history_tracker_scores_v1.py','--evaluation',result)
    evaluations={
        'twoframe_delta':N/'delta_history_train16_evaluation_v1',
        'twoframe_matched_raft':N/'raft_matched_history_train16_evaluation_v1',
        'dense_endpoint':N/'delta_dense_endpoint_train16_evaluation_v1',
        'dense_lsq':N/'delta_dense_lsq_train16_evaluation_v1',
    }
    comparison=[];counts={};hashes={}
    for method,root in evaluations.items():
        complete=json.loads((root/'complete.json').read_text())
        assert complete['anchors']==16 and complete['source_points']==41122
        assert complete['valid_objects_by_horizon']==[373,366,358,359]
        assert complete['valid_points_by_horizon']==[40622,39975,38639,39460]
        rows=list(csv.DictReader((root/'summary.csv').open()))
        for row in rows:
            if row['clock']=='nominal' and float(row['horizon_s'])==2 and row['group'] in ('moving','stationary'):
                comparison.append(dict(method=method,**row))
        counts[method]=json.loads((root/'independent_score_audit.json').read_text())['surface_supported_original_points']
        hashes[method]=hashlib.sha256((root/'complete.json').read_bytes()).hexdigest()
    out=dict(status='COMPLETE_MATCHED_DENSE_HISTORY_TRAIN16_COMPARISON',
             anchors=16,scenes=8,source_points=41122,optimizer_updates=0,
             nominal_2s_rows=comparison,surface_support=counts,evaluation_complete_sha256=hashes,
             baseline_is_O=False,O_occupancy_evaluated=False,
             generalization_claim=False,extra_input_observations=True,
             caveat='Masks differ; original material-point population and fallback are fixed. '
                    'Endpoint/LSQ comparison is not a pure fit-rule intervention.')
    with (N/'delta_dense_history_comparison_v1.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    for method in evaluations:
        print(method)
        for arm in ('CRN_CV','CV_D_speed05','surface_all_zero','surface_quality_zero',
                    'CV_surface_all_uncovered','CV_surface_quality_uncovered'):
            v={r['group']:float(r['xy_epe_m']) for r in comparison if r['method']==method and r['arm']==arm}
            print(arm,json.dumps(v))


if __name__=='__main__':
    main()
