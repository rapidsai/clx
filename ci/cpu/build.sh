#!/usr/bin/env bash
# Copyright (c) 2019, NVIDIA CORPORATION.
################################################################################
# rapidscyber cpu build
################################################################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

conda remove -y nomkl blas libblas

###############################################################################
# BUILD - Conda package builds (conda deps: libclx <- clx)
################################################################################

logger "Build conda pkg for libclx..."
source ci/cpu/libclx/build_libclx.sh

logger "Build conda pkg for clx..."
source ci/cpu/clx/build_clx.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload libclx conda pkg..."
source ci/cpu/libclx/upload-anaconda.sh

logger "Upload clx conda pkg..."
source ci/cpu/clx/upload-anaconda.sh
