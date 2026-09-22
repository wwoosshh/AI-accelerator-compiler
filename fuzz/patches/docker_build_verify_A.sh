#!/usr/bin/env bash
# CPU-only PyTorch build inside a Linux container with patch 0003 applied, then run the A repros.
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends git build-essential ccache libomp-dev > /dev/null
pip install -q numpy pyyaml setuptools cmake ninja typing_extensions requests expecttest
mkdir -p /src && cd /src
git clone --depth 1 --recursive --shallow-submodules -j8 https://github.com/pytorch/pytorch.git pytorch 2>&1 | tail -2
cd /src/pytorch
git apply --check /work/patches/0003-functionalization-unfold-inverse-as_strided_scatter-PROPOSED-UNCOMPILED.patch && git apply /work/patches/0003-functionalization-unfold-inverse-as_strided_scatter-PROPOSED-UNCOMPILED.patch && echo "PATCH_APPLIED"
pip install -q -r requirements.txt
export USE_CUDA=0 USE_ROCM=0 USE_XPU=0 USE_MPS=0 USE_MKLDNN=0 USE_DISTRIBUTED=0 USE_GLOO=0 USE_NCCL=0 USE_TENSORPIPE=0 \
       USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0 USE_KINETO=0 USE_ITT=0 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 BUILD_TEST=0 USE_NUMPY=1 USE_OPENMP=1 \
       CMAKE_BUILD_TYPE=Release MAX_JOBS=12 CC="ccache gcc" CXX="ccache g++"
date; echo "BUILD_START"
python setup.py develop > /work/patches/docker_build_A.log 2>&1; echo "BUILD_EXIT=$?"; date
tail -5 /work/patches/docker_build_A.log
python -c "import torch; print('built torch', torch.__version__)" && echo "IMPORT_OK"
echo "=== repro_unfold_zero.py (CPU aot_eager) ==="; python /work/repro/repro_unfold_zero.py 2>&1 | grep -vE "^W0"
echo "=== evidence_A_functionalize.py ==="; python /work/repro/evidence_A_functionalize.py 2>&1 | grep -vE "^W0"
echo "=== formula validation ==="; python /work/patches/validate_unfold_inverse_formula.py
echo "VERIFY_DONE"
