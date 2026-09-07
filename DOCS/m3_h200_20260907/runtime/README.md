# H200 runtime snapshot and restart instructions

Captured 2026-09-07 for the independent `hym_driveocc_m3` environment. The installed environment passed a complete CPU import of `projects.mmdet3d_plugin`; MMCV, DCNv3, dvxlr, dvxlr_v2 and all 11 MMDetection3D extensions contain `sm_90` binaries. GPU/model/DDP validation is recorded separately in `../GPU_preflight.md`.

**The archived rebuild recipe has not been independently rerun from an empty environment.** The Conda export, pip freeze and constraints are snapshots of the installed state. Rebuilding also requires the existing patched local backup source trees described below; the version list alone does not reproduce those changes.

## Reuse after a shell/server restart

The NAS environment and extension cache persist. After syncing this archived runtime directory into the deployed code, open a Bash shell on the H200 host and run:

```bash
source /home/wangning/Workspace/RadarFlowOcc-m3/DOCS/m3_h200_20260907/runtime/runtime_env.sh
cd "$M3_RUNTIME_CODE"
CUDA_VISIBLE_DEVICES='' python -c 'import torch, sqlite3, mmcv; print(torch.__version__, mmcv.__version__)'
```

The helper changes shell variables only; it does not install packages, create directories, select a GPU, or set training hyperparameters. Use the reviewed training launcher for GPU selection, seeds, numerical settings and the L40S-equivalent training contract. Do not source the installation scripts: they intentionally hide GPU devices for CPU-only compilation.

`LD_LIBRARY_PATH` must put this environment's `lib` first **before Python starts**. Otherwise a previously loaded system `libstdc++.so.6` can make the Conda ICU/sqlite3 stack fail with `CXXABI_1.3.15` missing. `PYTHONPATH` must point to the deployed project; a stale MMCV source-directory override can hide the installed `_ext.so`.

## Installed versions and paths

| Component | Observed version |
|---|---|
| Python | 3.10.21 |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2+cu121 |
| MMCV / MMDetection / MMSegmentation | 1.4.0 / 2.14.0 / 0.14.1 |
| MMDetection3D / Detectron2 / DCNv3 | 0.17.1 / 0.6 / 1.0 |
| NumPy / numba / networkx | 1.23.5 / 0.58.1 / 2.8.8 |
| CUDA compiler / C++ compiler | CUDA 12.4 / GCC 11.4 |

- Environment: `/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/envs/hym_driveocc_m3`.
- Code: `/home/wangning/Workspace/RadarFlowOcc-m3`.
- Shared extension cache for this run: `/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/torch_extensions`.
- Compiler settings: `CUDA_HOME=/usr/local/cuda-12.4`, `CC=/usr/bin/gcc-11`, `CXX=/usr/bin/g++-11`, `TORCH_CUDA_ARCH_LIST=9.0`.

## Rebuild prerequisites and snapshots

`setup_runtime.sh` creates/reuses the private environment and installs the base frameworks. `setup_project_extensions.sh` builds DCNv3 from a private copy, compiles the two dvxlr modules and checks CPU import and CUDA architectures. Both scripts include the required library path and use `CUDA_VISIBLE_DEVICES=''`, `FORCE_CUDA=1`, and at most four compiler jobs.

The base recipe depends on these existing backup directories under `/storage/data/metaiot_data/wangning/code_backup/RadarFlowOcc/third_party_sources`: `mmcv-full-1.4.0`, `mmdetection3d`, and `detectron2`. It copies source files and excludes old binaries/build directories. Preserve the MMCV C++17 and PyTorch 2.x DDP synchronization compatibility changes. `mmcv_torch21_compat.patch` records the known changes in `setup.py` and `mmcv/parallel/distributed.py`; `mmcv_source_provenance.json` records their source hashes. This patch is not an independent audit of every file in all three backup trees.

For an explicitly requested fresh rebuild, the 27-package `conda-explicit-linux-64.txt` can seed the Conda portion before running the two installation scripts. Export an absolute `PIP_CONSTRAINT` path to `constraints-pypi.txt` when using the scripts to retain the captured PyPI versions. The constraints deliberately omit locally compiled MMCV, MMDetection3D, Detectron2 and DCNv3; obtain those from the patched sources. The raw `runtime_pip_freeze.txt` includes local source URIs and is evidence, not a portable one-command installer.

`runtime_pip_check.txt` retains three known legacy requirement differences: MMDetection3D declares older NumPy, numba and networkx versions than the Python 3.10-compatible versions used here. These declarations were not edited to hide the differences. Actual CPU import and subsequent GPU checks are the relevant execution evidence.

This directory contains only small scripts, metadata, snapshots and a source patch. It contains no SSH credentials, compiled binaries, datasets, checkpoints or full sampler arrays. Archiving and syntax checks did not modify the remote environment or run a GPU task.
