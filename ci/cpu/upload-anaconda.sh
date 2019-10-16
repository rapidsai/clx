#!/bin/bash

set -e

export UPLOADFILE=`conda build conda/recipes/clx --python=$PYTHON --output`
CUDA_REL=${CUDA_VERSION%.*}

SOURCE_BRANCH=master

LABEL_OPTION="--label main --label cuda9.2 --label cuda10.0 --label cuda10.1"
echo "LABEL_OPTION=${LABEL_OPTION}"

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
  echo "Skipping upload"
  return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
fi

echo "Upload"
echo ${UPLOADFILE}
anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${UPLOADFILE}
