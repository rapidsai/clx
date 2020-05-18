# An integration test & dev container based on rapids-dev-nightly with CLX installed from current branch
ARG RAPIDS_VERSION=0.14
ARG CUDA_VERSION=10.2
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7
FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

ARG CMAKE_VERSION=3.14.5
ARG CYTHON_VERSION=0.29

ADD . /rapids/clx/

SHELL ["/bin/bash", "-c"]

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y vim

RUN source activate rapids && \
    conda remove nomkl blas libblas && \
    conda install pytorch torchvision -c pytorch

RUN source activate rapids && \
    conda install -c rapidsai-nightly -c nvidia -c conda-forge -c defaults rapids=${RAPIDS_VERSION} cmake=${CMAKE_VERSION} cmake_setuptools cython=${CMAKE_VERSION} pytest jupyterlab && \
    conda install -c nvidia -c conda-forge -c defaults datashader>=0.10.* panel=0.6.* geopandas>=0.6.* pyppeteer seaborn s3fs nodejs ipywidgets && \
    pip install "git+https://github.com/rapidsai/cudatashader.git"

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
