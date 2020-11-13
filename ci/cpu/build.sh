#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# CLX cpu build
################################################################################
set -e

<<<<<<< HEAD
# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
=======
# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
>>>>>>> origin/branch-0.17

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
<<<<<<< HEAD
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls
=======
conda list
>>>>>>> origin/branch-0.17

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

###############################################################################
# BUILD - Conda package build
################################################################################

<<<<<<< HEAD
gpuci_logger "Build conda pkg for libclx"
gpuci_conda_retry build conda/recipes/libclx

gpuci_logger "Build conda pkg for clx"
gpuci_conda_retry build -c pytorch -c nvidia -c conda-forge -c defaults conda/recipes/clx --python=$PYTHON
=======
logger "Build conda pkg for clx..."
source ci/cpu/clx/build_clx.sh
>>>>>>> origin/branch-0.17

################################################################################
# UPLOAD - Conda package
################################################################################

<<<<<<< HEAD
gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh
=======
logger "Upload clx conda pkg..."
source ci/cpu/clx/upload-anaconda.sh
>>>>>>> origin/branch-0.17
