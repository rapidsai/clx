ARG image=rapidsai/rapidsai-dev-nightly:0.12-cuda10.1-devel-ubuntu18.04-py3.6

FROM ${image}
USER root

ADD examples/streamz/scripts /scripts
ADD examples/streamz/data /data
ADD examples/streamz/python /python
ADD . /clx

RUN apt-get update -y && \
    apt-get install -y librdkafka-dev \
                       vim \
                       git \
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
    && conda install -c conda-forge -c pytorch -c anaconda -c rapidsai \
                    librdkafka=1.1.0=hb2b7465_0 \
                    streamz=0.5.2=py_0 \
                    python-confluent-kafka=1.1.0=py36h516909a_0 \
                    ujson=1.35=py36h14c3975_1001 \
                    pytorch=1.3.1=py3.6_cuda10.1.243_cudnn7.6.3_0 \
                    graphviz=2.40.1=h0f2764d_1 \
                    networkx \
                    pynvml=8.0.3=py_0 \
                    graphviz=2.40.1=h0f2764d_1 \
                    python-graphviz=0.13=py_0 \
                    networkx \
                    snakeviz=2.0.0=py_0 \
                    openjdk=8.0.152 \
    && cd /clx \
    && python setup.py install

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN wget -q http://www.gtlib.gatech.edu/pub/apache/kafka/2.3.0/kafka_2.11-2.3.0.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

CMD ["/bin/bash", "/scripts/entry.sh"]