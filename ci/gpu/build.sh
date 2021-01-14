#/usr/bin/env bash
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
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

logger "Get env..."
env

logger "Activate conda env..."
source activate rapids

logger "Check versions..."
python --version

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "conda install required packages"
conda install -y -c pytorch -c gwerbin \
    "rapids-build-env=$MINOR_VERSION.*" \
    "rapids-notebook-env=$MINOR_VERSION.*" \
    "cugraph=${MINOR_VERSION}" \
    "cuml=${MINOR_VERSION}" \
    "dask-cuda=${MINOR_VERSION}" \
    "pytorch=1.7.0" \
    "torchvision" \
    "python-confluent-kafka" \
    "transformers=4.*" \
    "seqeval=0.0.12" \
    "python-whois" \
    "requests" \
    "matplotlib" \
    "faker"

logger "pip install git+https://github.com/rapidsai/cudatashader.git"
pip install "git+https://github.com/rapidsai/cudatashader.git"
logger "pip install mockito"
pip install mockito
pip install wget
pip install faker

conda list

################################################################################
# BUILD - Build clx from source
################################################################################

logger "Build clx..."
$WORKSPACE/build.sh clean clx

################################################################################
# TEST - Test python package
################################################################################
set +e -Eo pipefail
EXITCODE=0
trap "EXITCODE=1" ERR

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    cd ${WORKSPACE}/python
    py.test --ignore=ci --cache-clear --junitxml=${WORKSPACE}/junit-clx.xml -v
    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log
fi

return "${EXITCODE}"
