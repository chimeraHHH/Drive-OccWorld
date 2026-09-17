"""Metadata-only history availability on the unchanged train512/dev200 anchors.

No images, radar files, annotation boxes, model, predictions or GT are read.
Sample timestamps are not substituted for asynchronous sensor timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


SAMPLE_SHA = '6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d'
INDEX_SHA = dict(train='1dc14643df86f85c5ae87b17b83c60cada99f189a025675d150062acb046e7b1',
                 development='fd42d2e511d754755fa5e2c06527077351869f06e868ecb8acb4404eadaa26c3')
HORIZONS = (1, 2, 3, 4, 8)
CAMERAS = {'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    started = time.monotonic()
    samplepath = Path(a.sample_json)
    assert samplepath.stat().st_size < 10_000_000
    assert sha(samplepath) == SAMPLE_SHA
    samples = json.loads(samplepath.read_text())
    by_token = {s['token']: s for s in samples}
    assert len(samples) == len(by_token) == 34149
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    evidence = dict(schema='visual-history-population-audit-v1',
        sample_json_sha256=SAMPLE_SHA, script_sha256=sha(__file__), splits={},
        image_files_read=0, radar_files_read=0, annotations_read=0,
        models_loaded=0, optimizer_updates=0, predictions_read=0,
        limitation='Metadata availability only; sample timestamps are not acquisition times; H includes current; sensor files not decoded/validated')
    details = []
    for split in ('train', 'development'):
        path = Path(a.cache)/split/'index.json'
        assert sha(path) == INDEX_SHA[split]
        index = json.loads(path.read_text())
        records = index['records']
        assert len(records) == (512 if split == 'train' else 200)
        assert index['status'] == 'COMPLETE'
        assert len({r['sample_token'] for r in records}) == len(records)
        rows = []
        for ordinal, record in enumerate(records):
            current = by_token[record['sample_token']]
            assert current['scene_token'] == record['scene_token']
            chain = [current]
            for _ in range(7):
                if not chain[-1]['prev']:
                    break
                previous = by_token[chain[-1]['prev']]
                assert previous['scene_token'] == current['scene_token']
                assert previous['next'] == chain[-1]['token']
                assert previous['timestamp'] < chain[-1]['timestamp']
                chain.append(previous)
            for s in chain:
                assert CAMERAS <= set(s['data'])
                assert len({s['data'][c] for c in CAMERAS}) == 6
            row = dict(split=split, ordinal=ordinal, sample_token=current['token'],
                scene_token=current['scene_token'], available_up_to_eight=len(chain),
                history={str(h):dict(available=len(chain)>=h,
                    sample_span_seconds=(current['timestamp']-chain[h-1]['timestamp'])/1e6
                        if len(chain)>=h else None) for h in HORIZONS})
            rows.append(row)
            assert time.monotonic()-started < 60
        per_h = {}
        for h in HORIZONS:
            valid = [r for r in rows if r['history'][str(h)]['available']]
            missing = [r for r in rows if not r['history'][str(h)]['available']]
            spans = [r['history'][str(h)]['sample_span_seconds'] for r in valid]
            per_h[str(h)] = dict(anchors=len(valid), scenes=len({r['scene_token'] for r in valid}),
                missing_anchors=len(missing), missing_scenes=len({r['scene_token'] for r in missing}),
                sample_span_minmax_seconds=[min(spans), max(spans)] if spans else None,
                unchanged_original_population=len(valid)==len(records))
        assert per_h['3']['anchors'] == len(records), 'Original history2+current unavailable'
        evidence['splits'][split] = dict(index_sha256=sha(path), anchors=len(records),
            scenes=len({r['scene_token'] for r in rows}), by_camera_times=per_h)
        details.extend(rows)
    (out/'records.json').write_text(json.dumps(details, indent=2)+'\n')
    evidence.update(status='COMPLETE_METADATA_ONLY_AUDIT', elapsed_seconds=time.monotonic()-started,
                    records_sha256=sha(out/'records.json'))
    (out/'summary.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('sample-json', 'cache', 'out'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
