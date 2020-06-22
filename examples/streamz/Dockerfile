ARG RAPIDS_VERSION=0.15
ARG CUDA_VERSION=10.1
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7

FROM gpuci/rapidsai:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}
USER root

ADD examples/streamz/scripts /scripts
ADD examples/streamz/data /data
ADD examples/streamz/python /python
ADD . /rapids/clx

RUN apt-get update -y && \
    apt-get install -y librdkafka-dev \
                       vim \
                       wget \
                       dnsutils \
                       net-tools \
                       git \
                       gdb \
                       build-essential \
                       valgrind \
                       unzip \
                       libboost-all-dev && \
    apt-get clean

# Install CMake from source
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.6/cmake-3.14.6.tar.gz && \
    tar -xzvf cmake-3.14.6.tar.gz && \
    rm -rf cmake-3.14.6.tar.gz && \
    cd cmake-3.14.6 && \
    ./configure && \
    make -j && \
    make install

ENV DEBIAN_FRONTEND noninteractive
ENV SCALA_VERSION 2.11
ENV KAFKA_VERSION 2.3.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"

# Install common conda packages that most issues will require
RUN source activate rapids \
    && conda install -c conda-forge -c pytorch -c anaconda -c rapidsai -c nvidia -c numba \
                    streamz=0.5.3 \
                    python-confluent-kafka=1.1.0 \
                    openjdk=8.0.152 \
                    pytorch=1.3.1 \
                    cudf=${RAPIDS_VERSION} \
                    dask-cudf=${RAPIDS_VERSION} \
                    dask-cuda=${RAPIDS_VERSION} \
                    ipython=7.3* \
                    jupyterlab \
                    rmm=${RAPIDS_VERSION} \
    && pip install transformers==2.7.0

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

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN wget -q http://www.gtlib.gatech.edu/pub/apache/kafka/2.3.0/kafka_2.11-2.3.0.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

ENTRYPOINT ["/bin/bash", "/scripts/entry.sh"]

WORKDIR /rapids/clx

