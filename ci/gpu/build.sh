#/usr/bin/env bash
set -e
NUMARGS=$#
ARGS=$*

# gpuci_logger function for build status output
function gpuci_logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
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

gpuci_logger "Check versions"
python --version

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

gpuci_logger "conda install required packages"
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

gpuci_logger "pip install git+https://github.com/rapidsai/cudatashader.git"
pip install "git+https://github.com/rapidsai/cudatashader.git"
gpuci_logger "pip install mockito"
pip install mockito
pip install wget
pip install faker

conda info
conda config --show-sources
conda list --show-channel-urls

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

return "${EXITCODE}"
