# An integration test & dev container which builds and installs CLX from default branch
ARG CUDA_VERSION=10.1
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
FROM gpuci/miniconda-cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

ARG CC=5
ARG CXX=5

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
    apt install -y \
      git \
      gcc-${CC} \
      g++-${CXX} \
      libboost-all-dev \
      tzdata

# Add everything from the local build context
ADD . /rapids/clx/

RUN mkdir -p /rapids/utils 
COPY ./docker/start_jupyter.sh ./docker/stop_jupyter.sh /rapids/utils/

ARG CUDA_SHORT_VERSION
ARG PYTHON_VERSION=3.7
RUN conda env create --name rapids --file /rapids/clx/conda/environments/clx_dev_cuda${CUDA_SHORT_VERSION}.yml python=${PYTHON_VERSION}

# libclx build/install
ENV CC=/usr/bin/gcc-${CC}
ENV CXX=/usr/bin/g++-${CXX}
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

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini && \
    echo "source activate rapids" >> ~/.bashrc
ENTRYPOINT [ "/usr/bin/tini", "--", "/rapids/clx/docker/.run_in_rapids.sh" ]
CMD [ "/bin/bash" ]
