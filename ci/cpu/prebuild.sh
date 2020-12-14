#!/usr/bin/env bash

if [[ "$CUDA" == "10.1" ]]; then
    export UPLOAD_CLX=1
else
    export UPLOAD_CLX=0
fi
