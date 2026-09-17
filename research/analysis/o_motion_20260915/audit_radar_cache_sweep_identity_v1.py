"""Read-only counterexample: changing nsweeps does not invalidate this cache.

No raw radar, model, target, GPU, or cache writes. Run with the project's
existing Python environment. Output is JSON on stdout.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    source = Path('/home/wangning/Workspace/RadarFlowOcc-sota-p2/projects/'
                  'mmdet3d_plugin/datasets/radar_bev.py')
    expected = '87ddb0e056163b00abe3962803073090661a2e76e20f382f548fb3134f6c9e77'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('audited_radar_bev', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token = '464fe0be05a74ec9852573cb9e3afd89'
    common = dict(nusc=None, point_cloud_range=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
                  bev_h=200, bev_w=200, min_distance=1., count_clip=32.,
                  velocity_norm=20., rcs_norm=50., time_lag_norm=.5,
                  cache_dir='/home/wangning/data_cache/'
                            'RadarFlowOcc_sota_p2_20260911/radar5',
                  cache_readonly=True)
    loaders = [module.NuScenesRadarBEV(nsweeps=s, **common) for s in (5, 1)]
    paths = [loader.cache_path(token) for loader in loaders]
    assert paths[0] == paths[1]
    path = Path(paths[0])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    arrays = [loader._load_cache(token) for loader in loaders]
    assert all(array is not None for array in arrays)
    assert np.array_equal(arrays[0], arrays[1])
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
    print(json.dumps(dict(
        schema='radar-cache-sweep-identity-audit-v1',
        status='PASS_READ_ONLY_CACHE_COLLISION_COUNTEREXAMPLE',
        source_sha256=expected,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_token=token, nsweeps_settings=[5, 1],
        cache_paths=paths, same_path=True, cache_file_sha256=before,
        shape=list(arrays[0].shape), dtype=str(arrays[0].dtype),
        loaded_arrays_exact=True,
        array_sha256=hashlib.sha256(arrays[0].tobytes()).hexdigest(),
        cache_bytes_unchanged=True, raw_radar_files_read=0,
        models_loaded=0, optimizer_updates=0, cache_writes=0,
        conclusion='A nsweeps=1 loader accepts the existing radar5 cache '
                   'unchanged. Changing nsweeps alone cannot establish '
                   'a current-sweep-only input.',
        limitation='One actual cached sample; no independent raw-sweep '
                   'reconstruction or velocity provenance verification.'
    )))


if __name__ == '__main__':
    main()
