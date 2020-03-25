#!/usr/bin/env bash
set -e

if [ "$BUILD_LIBCLX" == '1' ]; then
  echo "Building libclx"
  CUDA_REL=${CUDA_VERSION%.*}
  
  conda build conda/recipes/libclx
fi