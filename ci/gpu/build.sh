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
gcc --version
g++ --version

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

conda install nvstrings=${MINOR_VERSION} cugraph=${MINOR_VERSION} dask-cudf=${MINOR_VERSION} \
   requests yaml python-confluent-kafka python-whois dask

pip install mockito
pip install cupy-cuda${CUDA_SHORT}

conda list

################################################################################
# INSTALL - Build package
################################################################################

cd $WORKSPACE
python setup.py build_ext --inplace
python setup.py install --single-version-externally-managed --record=record.txt

################################################################################
# TEST - Test python package
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    py.test --cache-clear --junitxml=${WORKSPACE}/junit-clx.xml -v
fi
