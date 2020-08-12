# An integration test & dev container which builds and installs CLX from default branch
ARG RAPIDS_VERSION=0.15
ARG CUDA_VERSION=10.1
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7
FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

# Add everything from the local build context
ADD . /rapids/clx/

RUN apt update -y --fix-missing && \
    apt upgrade -y

RUN source activate rapids && \
    conda install -c pytorch pytorch=1.5.* torchvision custreamz=${RAPIDS_VER} scikit-learn>=0.21 ipywidgets python-confluent-kafka transformers seqeval python-whois seaborn requests matplotlib pytest jupyterlab && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget && \
    pip install pytorch-transformers

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

WORKDIR /rapids/clx