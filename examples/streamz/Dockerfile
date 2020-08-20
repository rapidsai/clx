# An integratio:n test & dev container which builds and installs CLX from default branch
ARG CUDA_VERSION=10.2
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7
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

RUN apt-get install -y librdkafka-dev \
                       vim \
                       wget \
                       dnsutils \
                       net-tools \
                       gdb \
                       build-essential \
                       valgrind \
                       unzip && \
    apt-get clean

ENV DEBIAN_FRONTEND noninteractive
ENV SCALA_VERSION 2.11
ENV KAFKA_VERSION 2.3.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"

ENV CYBERT_HOME /opt/cybert
ADD examples/streamz/scripts "$CYBERT_HOME"/scripts
ADD examples/streamz/data "$CYBERT_HOME"/data
ADD examples/streamz/python "$CYBERT_HOME"/python
ADD . /rapids/clx

RUN mkdir -p /rapids/utils
COPY ./docker/start_jupyter.sh ./docker/stop_jupyter.sh /rapids/utils/

ARG CUDA_SHORT_VERSION
RUN conda env create --name rapids --file /rapids/clx/conda/environments/clx_dev_cuda${CUDA_SHORT_VERSION}.yml python=${PYTHON_VERSION}

RUN source activate rapids && conda install -c rapidsai -c nvidia -c pytorch -c conda-forge -c defaults \
    python-confluent-kafka \
    streamz \
    dask-cuda \
    openjdk=8.0.152 && \
    pip install transformers

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN wget -q http://www.gtlib.gatech.edu/pub/apache/kafka/2.3.0/kafka_2.11-2.3.0.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

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

WORKDIR /clx

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini && \
    echo "source activate rapids" >> ~/.bashrc
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/cybert/scripts/entry.sh"]
