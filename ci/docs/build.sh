#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#################################
# CLX Docs build script for CI #
#################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export HOME="$WORKSPACE"
export DOCS_WORKSPACE="$WORKSPACE/docs"
export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_SHORT=${CUDA_REL//./}
export PROJECTS=(clx)

# Switch to project root; also root of repo checkout
cd "$PROJECT_WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

logger "Activate conda env..."
source activate rapids
conda install --freeze-installed -c rapidsai-nightly -c rapidsai -c nvidia -c pytorch -c conda-forge \
    "pytorch>=1.7" torchvision "transformers=3.5.*" requests yaml python-confluent-kafka python-whois markdown beautifulsoup4 jq
    
pip install mockito
pip install "git+https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment.git#egg=slashnext-phishing-ir&subdirectory=Python SDK/src"
pip install cupy-cuda${CUDA_SHORT}

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Show conda info"
conda info
conda config --show-sources
conda list --show-channel-urls

#clx source build
"$PROJECT_WORKSPACE/build.sh" clx

#clx Sphinx Build
gpuci_logger "Build clx docs"
cd "$PROJECT_WORKSPACE/docs"
make html

cd $DOCS_WORKSPACE

if [ ! -d "api/clx/$BRANCH_VERSION" ]; then
  mkdir -p api/clx/$BRANCH_VERSION
fi

rm -rf api/clx/$BRANCH_VERSION/*
mv "$PROJECT_WORKSPACE/docs/build/html/"* $DOCS_WORKSPACE/api/clx/$BRANCH_VERSION


