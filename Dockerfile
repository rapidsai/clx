# An integration test & dev container based on rapids-dev-nightly with CLX installed from current branch
ARG RAPIDS_VERSION=0.15
ARG CUDA_VERSION=10.1
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7

FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

ADD . /rapids/clx/

SHELL ["/bin/bash", "-c"]

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y vim

RUN source activate rapids \
    && conda install datashader>=0.10.* panel=0.6.* geopandas>=0.6.* pyppeteer s3fs ipywidgets \
    && pip install "git+https://github.com/rapidsai/cudatashader.git"

# libclx build/install
RUN source activate rapids && \
    mkdir -p /rapids/clx/cpp/build && \
    cd /rapids/clx/cpp/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
    make -j install

# clx build/install
RUN source activate rapids && \
    cd /rapids/clx/python && \
    python setup.py build_ext --inplace && \
    python setup.py install

WORKDIR /rapids
