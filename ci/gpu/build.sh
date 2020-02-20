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
source activate gdf

logger "Check versions..."
python --version

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "conda install required packages"
conda install -y -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge -c defaults -c pytorch \
    "cugraph=${MINOR_VERSION}" \
    "cuxfilter=${MINOR_VERSION}" \
    "cupy>=6.6.0,<8.0.0a0,!=7.1.0" \
    "dask>=2.1.0" \
    "distributed>=2.1.0" \
    "dask-cudf=${MINOR_VERSION}" \
    "dask-cuda=${MINOR_VERSION}" \
    "pytorch==1.3.1" \
    "torchvision=0.4.2" \
    "yaml"


# conda install -y pytorch==1.3.1 torchvision -c pytorch

#pip install mockito
# pip install cupy-cuda${CUDA_SHORT}

conda list

################################################################################
# INSTALL - Build package
################################################################################

cd $WORKSPACE
python setup.py install

################################################################################
# TEST - Test python package
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    py.test --ignore=ci --cache-clear --junitxml=${WORKSPACE}/junit-clx.xml -v
fi
