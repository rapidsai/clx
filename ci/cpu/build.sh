#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION.
################################################################################
# CLX cpu build
################################################################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH

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
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

###############################################################################
# BUILD - Conda package build
################################################################################

logger "Build conda pkg for clx..."
source ci/cpu/clx/build_clx.sh

################################################################################
# UPLOAD - Conda package
################################################################################

logger "Upload clx conda pkg..."
source ci/cpu/clx/upload-anaconda.sh
