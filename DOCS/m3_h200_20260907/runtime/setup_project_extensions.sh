#!/usr/bin/env bash
set -euo pipefail
umask 077
project=/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200
code_dir=/home/wangning/Workspace/RadarFlowOcc-m3
env_dir=${project}/envs/hym_driveocc_m3
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${env_dir}/bin:${CUDA_HOME}/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=${env_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export CUDA_VISIBLE_DEVICES=''
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4
export TORCH_EXTENSIONS_DIR=${project}/cache/torch_extensions
export PIP_CACHE_DIR=${project}/cache/pip
export TMPDIR=${project}/cache/build/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=${code_dir}
mkdir -p "$TORCH_EXTENSIONS_DIR" "${project}/cache/build/ops_dcnv3"
trap 's=$?; printf "%s\n" "$s" > "${project}/diagnostics/project_extensions.exit"; if (( s == 0 )); then printf "READY_FOR_GPU_VALIDATION\n" > "${project}/diagnostics/runtime_setup.state"; else printf "PROJECT_EXTENSIONS_FAILED_%s\n" "$s" > "${project}/diagnostics/runtime_setup.state"; fi' EXIT
printf 'BUILD_PROJECT_EXTENSIONS\n' > "${project}/diagnostics/runtime_setup.state"
date -Is
if ! python -c 'import importlib.metadata; assert importlib.metadata.version("DCNv3") == "1.0"' >/dev/null 2>&1; then
  rsync -a --exclude=build --exclude='*.so' --exclude='*.egg-info' "${code_dir}/projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3/" "${project}/cache/build/ops_dcnv3/"
  # Build-only change in the private source copy: support an explicit CUDA build
  # while CUDA devices are hidden. The operator source is unchanged.
  python - <<'PY'
from pathlib import Path
p = Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/build/ops_dcnv3/setup.py')
s = p.read_text()
old = 'if torch.cuda.is_available() and CUDA_HOME is not None:'
new = 'if (torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1") and CUDA_HOME is not None:'
assert old in s or new in s
p.write_text(s.replace(old, new))
PY
  python -m pip install --no-deps --no-build-isolation "${project}/cache/build/ops_dcnv3"
fi
cd "$code_dir"
python - <<'PY'
import json, os, subprocess, pathlib, importlib
import torch, mmcv, mmdet3d, DCNv3
import projects.mmdet3d_plugin
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not torch.cuda.is_initialized()
ext = importlib.import_module('projects.mmdet3d_plugin.bevformer.utils.e2e_predictor_utils')
libraries = {
    'mmcv': importlib.import_module('mmcv._ext').__file__,
    'DCNv3': DCNv3.__file__,
    'dvxlr': ext.dvxlr.__file__,
    'dvxlr_v2': ext.dvxlr_v2.__file__,
}
for path in pathlib.Path(mmdet3d.__file__).parent.rglob('*.so'):
    libraries['mmdet3d/' + str(path.relative_to(pathlib.Path(mmdet3d.__file__).parent))] = str(path)
architectures = {}
for name, path in libraries.items():
    result = subprocess.run(['/usr/local/cuda-12.4/bin/cuobjdump', '--list-elf', path], text=True, capture_output=True)
    architectures[name] = dict(path=path, returncode=result.returncode,
                               elf_listing=result.stdout, stderr=result.stderr)
    if name in ('mmcv', 'DCNv3', 'dvxlr', 'dvxlr_v2'):
        assert result.returncode == 0 and 'sm_90' in result.stdout, (name, result.stdout, result.stderr)
record = dict(cpu_plugin_import='passed', cuda_initialized=torch.cuda.is_initialized(),
              torch_compiled_architectures=torch._C._cuda_getArchFlags(),
              libraries=architectures)
out = pathlib.Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/diagnostics/project_extensions.json')
out.write_text(json.dumps(record, indent=2))
print(json.dumps({k:v for k,v in record.items() if k != 'libraries'}, indent=2))
print('verified_libraries', list(libraries))
PY
python -m pip freeze > "${project}/diagnostics/runtime_pip_freeze.txt"
python -m pip check > "${project}/diagnostics/runtime_pip_check.txt" 2>&1 || true
date -Is
