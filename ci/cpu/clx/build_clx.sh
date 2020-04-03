#!/usr/bin/env bash
set -e

if [ "$BUILD_CLX" == "1" ]; then
  echo "Building clx"
  CUDA_REL=${CUDA_VERSION%.*}

  conda build conda/recipes/clx --python=$PYTHON
fi