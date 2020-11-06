#!/usr/bin/env bash
set -e

if [ "$BUILD_CLX" == "1" ]; then
  echo "Building clx"
  CUDA_REL=${CUDA_VERSION%.*}

  gpuci_conda_retry build -c pytorch -c nvidia -c conda-forge -c defaults conda/recipes/clx --python=$PYTHON
fi