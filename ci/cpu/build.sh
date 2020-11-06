#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# rapidscyber cpu build
################################################################################
set -e

# Logger function for build status output
function gpuci_logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=-4

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

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

###############################################################################
# BUILD - Conda package builds (conda deps: libclx <- clx)
################################################################################

gpuci_logger "Build conda pkg for libclx"
source ci/cpu/libclx/build_libclx.sh

gpuci_logger "Build conda pkg for clx"
source ci/cpu/clx/build_clx.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload libclx conda pkg"
source ci/cpu/libclx/upload-anaconda.sh

gpuci_logger "Upload clx conda pkg"
source ci/cpu/clx/upload-anaconda.sh
