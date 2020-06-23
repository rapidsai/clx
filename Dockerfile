# An integration test & dev container which builds and installs cuDF from master
ARG CUDA_VERSION=10.2
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
ADD . /clx/

ARG CUDA_SHORT_VERSION
RUN conda env create --name clx --file /clx/conda/environments/clx_dev_cuda${CUDA_SHORT_VERSION}.yml

# libclx build/install
ENV CC=/usr/bin/gcc-${CC}
ENV CXX=/usr/bin/g++-${CXX}
RUN source activate clx && \
    mkdir -p /clx/cpp/build && \
    cd /clx/cpp/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
    make -j install

# clx build/install
RUN source activate clx && \
    cd /clx/python && \
    python setup.py build_ext --inplace && \
    python setup.py install

WORKDIR /clx

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini && \
    echo "source activate clx" >> ~/.bashrc
ENTRYPOINT [ "/usr/bin/tini", "--", "/clx/docker/.start_jupyter_run_in_rapids.sh" ]
CMD [ "/bin/bash" ]
