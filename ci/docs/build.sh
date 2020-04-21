#!/bin/bash
set -e

export HOME=$WORKSPACE
export DOCS_WORKSPACE=$WORKSPACE/docs
export NIGHTLY_VERSION=$(echo $BRANCH_VERSION | awk -F. '{print $2}')

export CUDA_REL=${CUDA_VERSION%.*}
export CUDA_SHORT=${CUDA_REL//./}
export CLX_HOME=$WORKSPACE/clx_build

# Switch to project root; also root of repo checkout
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

#clx source build
git clone --single-branch --branch branch-${BRANCH_VERSION} https://github.com/rapidsai/clx.git ${CLX_HOME}
${CLX_HOME}/build.sh clean libclx clx

#clx Sphinx Build
logger "Build clx docs..."
cd $CLX_HOME/docs
make html

cd $DOCS_WORKSPACE

if [ ! -d "api/clx/$BRANCH_VERSION" ]; then
  mkdir -p api/clx/$BRANCH_VERSION
fi

rm -rf api/clx/$BRANCH_VERSION/*
mv $CLX_HOME/docs/build/html/* $DOCS_WORKSPACE/api/clx/$BRANCH_VERSION


# Customize HTML documentation
./update_symlinks.sh $NIGHTLY_VERSION
./customization/lib_map.sh
./customization/customize_docs_in_folder.sh api/clx/ $NIGHTLY_VERSION

cd $DOCS_WORKSPACE/api/clx
git add .
git commit -m "[gpuCI] Update docs for clx v$BRANCH_VERSION" || logger "Nothing to commit"