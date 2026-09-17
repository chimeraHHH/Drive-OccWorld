"""Independently sum saved confusion counts; no new inference or tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np

N = Path(__file__).resolve().parent
P = N.with_name('m0_improvement_20260915')
ROOT = N/'server_results/probes/future_observation_readout_train16_v5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = [json.loads(line) for line in (ROOT/'records.jsonl').read_text().splitlines()]
    complete = json.loads((ROOT/'complete.json').read_text())
    manifest = json.loads((ROOT/'manifest.json').read_text())
    index = json.loads((P/'server_results/cache/campaign_cache_v1/train/index.json').read_text())
    assert len(rows) == 16 and len({r['scene_token'] for r in rows}) == 8
    assert [r['sample_token'] for r in rows] == [r['sample_token'] for r in index['records'][:16]]
    assert manifest['source_sha256'] == sha(N/'future_observation_readout_probe_v5.py')
    assert complete['frozen_parameters_unchanged'] and complete['optimizer_steps'] == 0
    prior_path = N/'server_results/probes/oracle_transport_train16_v2/records.jsonl'
    prior = {r['sample_token']:r for r in [json.loads(line) for line in prior_path.read_text().splitlines()]}
    for row in rows:
        dt_path = N/'server_results/diagnostics/motion_evidence_train_v1/samples'/f'{row["ordinal"]:04d}_{row["sample_token"]}.npz'
        expected_dt = np.load(dt_path, allow_pickle=False)['dt_future_seconds']
        assert row['native_O_confusion_exact']
        assert row['full_hist']['O'] == prior[row['sample_token']]['hist_by_arm']['O']
        for population in ('full_hist', 'common_support_hist'):
            for arm in ('persistence', 'future_observation'):
                assert row[population][arm][0] == row[population]['O'][0]
                assert np.array_equal(np.asarray(row[population][arm]).sum(-1),
                                      np.asarray(row[population]['O']).sum(-1))
        for h, obs in enumerate(row['teacher_observations']):
            assert not obs['labels_loaded'] and obs['scene_token'] == row['scene_token']
            assert obs['actual_dt_seconds'] == 0 if h == 0 else abs(obs['actual_dt_seconds']-float(expected_dt[h-1])) < 1e-12
    result = dict(anchors=16, scenes=8, original_selection_exact=True, optimizer_steps=0,
                  original_historical_O_all16_all5h_counts_exact=True,
                  current_encoder_max_abs=max(r['current_encoder_max_abs'] for r in rows),
                  direct_readout_max_abs=max(r['direct_readout_max_abs'] for r in rows),
                  unused_identity_resampling_max_abs=max(r['unused_identity_resampling_max_abs'] for r in rows),
                  peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in rows)/1024**3,
                  seconds=complete['seconds'], comparisons={},
                  actual_future_dt_range_seconds=np.stack((np.min([[o['actual_dt_seconds'] for o in r['teacher_observations'][1:]] for r in rows],axis=0), np.max([[o['actual_dt_seconds'] for o in r['teacher_observations'][1:]] for r in rows],axis=0)),axis=1).tolist(),
                  geometry_overlap_fraction_by_horizon=np.mean([
                      [o['supported_fraction'] for o in r['teacher_observations']] for r in rows],axis=0).tolist(),
                  source_sha256={f:sha(ROOT/f) for f in ('manifest.json','records.jsonl','complete.json')})
    for population in ('full_hist', 'common_support_hist'):
        result['comparisons'][population] = {}
        for arm in ('O','persistence','future_observation'):
            counts = np.asarray([r[population][arm] for r in rows], dtype=np.int64).sum(axis=0)
            tp, fp, fn = counts[:,1,1], counts[:,0,1], counts[:,1,0]
            iou = 100*tp/(tp+fp+fn)
            recall = 100*tp/(tp+fn)
            macro = float(iou[1:].mean())
            assert abs(macro-complete['comparisons'][population][arm]['future_macro']['GMO_IoU_percent']) < 1e-12
            result['comparisons'][population][arm] = dict(future_macro_GMO_IoU_percent=macro,
                future_macro_GMO_recall_percent=float(recall[1:].mean()),
                per_horizon_GMO_IoU_percent=iou.tolist(), future_FP=int(fp[1:].sum()),
                future_FN=int(fn[1:].sum()), pooled_confusion=counts.tolist())
    (ROOT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
