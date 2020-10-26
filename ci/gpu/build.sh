#!/usr/bin/env bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Get env"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install conda dependenciess"
gpuci_conda_retry install -y -c pytorch -c gwerbin \
    "rapids-build-env=$MINOR_VERSION.*" \
    "rapids-notebook-env=$MINOR_VERSION.*" \
    "cugraph=${MINOR_VERSION}" \
    "cuml=${MINOR_VERSION}" \
    "dask-cuda=${MINOR_VERSION}" \
    "pytorch>=1.5" \
    "torchvision" \
    "python-confluent-kafka" \
    "transformers" \
    "seqeval=0.0.12" \
    "python-whois" \
    "requests" \
    "matplotlib" \
    "faker"

gpuci_logger "Install cudatashader"
pip install "git+https://github.com/rapidsai/cudatashader.git"
pip install mockito
pip install wget
pip install faker

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Show conda info"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ################################################################################
    # BUILD - Build libclx and clx from source
    ################################################################################

    gpuci_logger "Build libclx and clx"
    $WORKSPACE/build.sh clean libclx clx

    ################################################################################
    # TEST - Test python package
    ################################################################################
    set +e -Eo pipefail
    EXITCODE=0
    trap "EXITCODE=1" ERR

    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
    else
        cd ${WORKSPACE}/python
        py.test --ignore=ci --cache-clear --junitxml=${WORKSPACE}/junit-clx.xml -v
        ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
        python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
    fi
else
    export LD_LIBRARY_PATH="$WORKSPACE/ci/artifacts/clx/cpu/conda_work/build:$LD_LIBRARY_PATH"

    TESTRESULTS_DIR=${WORKSPACE}/test-results
    mkdir -p ${TESTRESULTS_DIR}
    SUITEERROR=0

    gpuci_logger "Check GPU usage..."
    nvidia-smi

    cd $WORKSPACE/python
    
    gpuci_logger "Installing librmm..."
    conda install -c $WORKSPACE/ci/artifacts/clx/cpu/conda-bld/ libclx
    export LIBCLX_BUILD_DIR="$WORKSPACE/ci/artifacts/clx/cpu/conda_work/build"
    
    gpuci_logger "Building clx"
    "$WORKSPACE/build.sh" -v clx
    
    set +e -Eo pipefail
    EXITCODE=0
    trap "EXITCODE=1" ERR

    gpuci_logger "Run pytest for CLX"
    py.test --ignore=ci --cache-clear --junitxml=${WORKSPACE}/junit-clx.xml -v
    EXITCODE=$?

    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
    
    if (( ${EXITCODE} != 0 )); then
        SUITEERROR=${EXITCODE}
        echo "FAILED: 1 or more tests in /clx/python"
    fi

    exit ${SUITEERROR}
fi

return "${EXITCODE}"
