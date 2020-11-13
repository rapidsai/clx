#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# CLX cpu build
################################################################################
set -e

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

# Setup 'conda' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

###############################################################################
# BUILD - Conda package build
################################################################################

gpuci_logger "Build conda pkg for libclx"
gpuci_conda_retry build conda/recipes/libclx

gpuci_logger "Build conda pkg for clx"
gpuci_conda_retry build -c pytorch -c nvidia -c conda-forge -c defaults conda/recipes/clx --python=$PYTHON

################################################################################
# UPLOAD - Conda package
################################################################################

gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh
