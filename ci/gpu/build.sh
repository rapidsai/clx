#!/usr/bin/env bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##########################################
# CLX GPU build & testscript for CI      #
##########################################

set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_SHORT=${CUDA_REL//./}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install conda dependencies"
gpuci_mamba_retry install -y \
    "cuxfilter=${MINOR_VERSION}" \
    "faker" \
    "python-whois" \
    "seqeval=1.2.2"

pip install -U torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install "git+https://github.com/rapidsai/cudatashader.git"
pip install "git+https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment.git#egg=slashnext-phishing-ir&subdirectory=Python SDK/src"
pip install mockito
pip install wget

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Show conda info"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build clx
################################################################################

#TODO: Move boa installation to gpuci/rapidsai
gpuci_mamba_retry install boa

gpuci_logger "Build and install clx..."
cd "${WORKSPACE}"
CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"
gpuci_conda_retry mambabuild --croot "${CONDA_BLD_DIR}" conda/recipes/clx
gpuci_mamba_retry install -c "${CONDA_BLD_DIR}" clx

################################################################################
# TEST - Test python package
################################################################################
set +e -Eo pipefail
EXITCODE=0
trap "EXITCODE=1" ERR

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    cd "$WORKSPACE/python"
    py.test --ignore=ci --cache-clear --junitxml="$WORKSPACE/junit-clx.xml" -v
    "$WORKSPACE/ci/gpu/test-notebooks.sh" 2>&1 | tee nbtest.log
    python "$WORKSPACE/ci/utils/nbtestlog2junitxml.py" nbtest.log
fi

return "${EXITCODE}"
