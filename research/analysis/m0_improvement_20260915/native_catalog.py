"""Read-only identities and file provenance for native M0 training and validation."""
import argparse, copy, datetime, hashlib, json, os, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--split-manifest', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    start = time.monotonic()
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    cfg = Config.fromfile(a.config)
    old = json.loads(Path(a.split_manifest).read_text())
    val_scenes = set(old['development_scenes']) | set(old['locked_scenes'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='native-m0-catalog-v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  config_sha256=sha(a.config), split_manifest_sha256=sha(a.split_manifest), splits={})
    for split in ['train', 'validation']:
        dc = copy.deepcopy(cfg.data.test)
        dc.test_mode = True
        dc.pop('samples_per_gpu', None)
        assert dc.radar_cfg.cache_readonly is True and dc.future_metadata_only is True
        if split == 'train':
            dc.ann_file = cfg.data.train.ann_file
        ds = build_dataset(dc)
        rows = [dict(sample_token=str(ds.data_infos[j]['token']), scene_token=str(ds.data_infos[j]['scene_token']),
                     official_index=i, split=('train' if split == 'train' else
                         'development' if str(ds.data_infos[j]['scene_token']) in old['development_scenes'] else 'locked'))
                for i, j in enumerate(ds.usable_index)]
        if split == 'train':
            assert not set(r['scene_token'] for r in rows) & val_scenes
        else:
            assert [r['sample_token'] for r in rows] == [r['sample_token'] for r in old['samples']]
        assert len({r['sample_token'] for r in rows}) == len(rows)
        result['splits'][split] = dict(ann_file=dc.ann_file, ann_sha256=sha(dc.ann_file),
                                      num_scenes=len(set(r['scene_token'] for r in rows)), records=rows)
        print(json.dumps(dict(event='CATALOG', split=split, samples=len(rows), scenes=result['splits'][split]['num_scenes'])), flush=True)
    result['seconds'] = time.monotonic() - start
    (out / 'catalog.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'complete.json').write_text(json.dumps(dict(status='COMPLETE', catalog_sha256=sha(out / 'catalog.json'), seconds=result['seconds']), indent=2) + '\n')


if __name__ == '__main__':
    main()
