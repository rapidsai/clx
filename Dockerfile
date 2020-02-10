# An integration test & dev container based on rapids-dev-nightly with CLX installed from current branch
ARG RAPIDS_VERSION=0.13
ARG CUDA_VERSION=10.0
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7

FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

ADD . /rapids/clx/

SHELL ["/bin/bash", "-c"]

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y vim

RUN source activate rapids \
    && conda install --freeze-installed panel=0.6.* datashader geopandas pyppeteer cuxfilter s3fs \
    && pip install mockito \
    && cd /rapids \
    && git clone https://github.com/rapidsai/cudatashader.git  /rapids/cudatashader \
    && cd /rapids/cudatashader \
    && pip install -e . \
    && cd /rapids/clx \
    && python setup.py install

WORKDIR /rapids

