# An integration test & dev container based on rapids-dev-nightly with CLX installed from current branch
ARG RAPIDS_VERSION=0.13
ARG CUDA_VERSION=10.1
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7

FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

ADD . /rapids/clx/

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y vim

RUN source activate rapids \
    && conda install -y -c pytorch pytorch==1.4.0 torchvision=0.5.0 datashader>=0.10.* panel=0.6.* geopandas>=0.6.* pyppeteer s3fs \
    && pip install "git+https://github.com/rapidsai/cudatashader.git" \
    && cd /rapids/clx \
    && pip install -e .

WORKDIR /rapids
