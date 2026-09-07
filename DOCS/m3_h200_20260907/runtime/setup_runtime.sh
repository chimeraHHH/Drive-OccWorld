#!/usr/bin/env bash
# Isolated H200 runtime. Existing environments and backup sources stay untouched.
set -euo pipefail
umask 077
project=/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200
env_dir=${project}/envs/hym_driveocc_m3
source_backup=/storage/data/metaiot_data/wangning/code_backup/RadarFlowOcc/third_party_sources
source_dir=${project}/cache/build/third_party_sources
export CONDA_PKGS_DIRS=${project}/cache/conda_pkgs
export PIP_CACHE_DIR=${project}/cache/pip
export TMPDIR=${project}/cache/build/tmp
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${env_dir}/bin:${CUDA_HOME}/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=${env_dir}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export CUDA_VISIBLE_DEVICES=''
export TORCH_CUDA_ARCH_LIST=9.0
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$source_dir" "$TMPDIR" "${project}/diagnostics" "${project}/cache/wheels"
trap 's=$?; printf "%s\n" "$s" > "${project}/diagnostics/runtime_setup.exit"; if (( s == 0 )); then printf "CPU_RUNTIME_VERIFIED\n" > "${project}/diagnostics/runtime_setup.state"; else printf "FAILED_EXIT_%s\n" "$s" > "${project}/diagnostics/runtime_setup.state"; fi' EXIT
printf 'RUNNING\n' > "${project}/diagnostics/runtime_setup.state"
stage() { date -Is; printf '%s\n' "$1"; printf '%s\n' "$1" > "${project}/diagnostics/runtime_setup.stage"; }
package_is() {
  python - "$1" "$2" <<'PY'
import importlib.metadata, sys
try:
    found = importlib.metadata.version(sys.argv[1])
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if found == sys.argv[2] else 1)
PY
}
date -Is
if [[ ! -x ${env_dir}/bin/python ]]; then
  /home/metaiot/miniconda3/bin/conda create -y -p "$env_dir" --override-channels -c conda-forge python=3.10 pip=24.0 setuptools=69.5.1 wheel
fi
stage PYTORCH
if ! package_is torch 2.1.2+cu121 || ! package_is torchvision 0.16.2+cu121; then
  python -m pip install 'torch==2.1.2' 'torchvision==0.16.2' --index-url https://download.pytorch.org/whl/cu121
fi
stage PYTHON_DEPENDENCIES
python -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'numba==0.58.1' 'pandas==1.5.3' 'matplotlib==3.5.2' 'scikit-image==0.20.0' 'opencv-python==4.8.1.78' 'shapely==1.8.5.post1' 'networkx==2.8.8' 'trimesh==2.35.39' 'yapf==0.40.1' 'iopath==0.1.9' 'timm==0.6.13' 'casadi==3.6.7' 'pytorch-lightning==1.2.5' 'torchmetrics==0.2.0' 'nuscenes-devkit==1.1.11' 'lyft-dataset-sdk==0.0.8' 'plyfile==1.0.3' 'seaborn==0.12.2' 'tensorboard==2.14.0' 'ipython==8.12.3' 'pytest==8.3.5' 'Pillow==9.5.0' 'sympy==1.12' addict packaging pyyaml einops fvcore ninja psutil pyquaternion prettytable terminaltables pycocotools six tqdm
for component in mmcv-full-1.4.0 mmdetection3d detectron2; do
  if [[ ! -f ${source_dir}/${component}/setup.py ]]; then
    mkdir -p "${source_dir}/${component}"
    rsync -a --exclude=build --exclude='*.so' --exclude='*.egg-info' "${source_backup}/${component}/" "${source_dir}/${component}/"
  fi
done
stage MMCV_BUILD
if ! package_is mmcv-full 1.4.0; then
  python -m pip install --no-deps --no-build-isolation "${source_dir}/mmcv-full-1.4.0"
fi
python -m pip install --no-deps 'mmdet==2.14.0' 'mmsegmentation==0.14.1'
stage MMDET3D_BUILD
if ! package_is mmdet3d 0.17.1; then
  python -m pip install --no-deps --no-build-isolation "${source_dir}/mmdetection3d"
fi
stage DETECTRON2_BUILD
if ! package_is detectron2 0.6; then
  python -m pip install --no-build-isolation "${source_dir}/detectron2"
fi
stage CPU_IMPORT_VERIFICATION
python -m pip freeze > "${project}/diagnostics/runtime_pip_freeze.txt"
python - <<'PY'
import json, platform, torch, mmcv, mmdet, mmseg, mmdet3d, detectron2
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
info = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
            mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmseg=mmseg.__version__,
            mmdet3d=mmdet3d.__version__, mmcv_cuda=get_compiling_cuda_version(),
            mmcv_compiler=get_compiler_version())
print(json.dumps(info, indent=2))
with open('/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/diagnostics/runtime_versions.json', 'w') as f:
    json.dump(info, f, indent=2)
PY
date -Is
