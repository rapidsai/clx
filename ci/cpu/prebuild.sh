#!/usr/bin/env bash

export BUILD_CLX=1
export BUILD_LIBCLX=1

if [[ "$CUDA" == "10.1" ]]; then
    export UPLOAD_CLX=1
else
    export UPLOAD_CLX=0
fi

if [[ "$PYTHON" == "3.7" ]]; then
    export UPLOAD_LIBCLX=1
else
    export UPLOAD_LIBCLX=0
fi
