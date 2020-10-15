# An integration test & dev container which builds and installs CLX from default branch
ARG RAPIDS_VERSION=0.16
ARG CUDA_VERSION=10.2
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.7
FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

# Add everything from the local build context
ADD . /rapids/clx/
RUN chmod -R ugo+w /rapids/clx

RUN apt update -y --fix-missing && \
    apt upgrade -y

RUN apt-get install -y librdkafka-dev \
        krb5-user \
        vim \
        wget \
        dnsutils \
        net-tools \
        gdb \
        build-essential \
        valgrind \
        unzip && \
    apt-get clean

ENV SCALA_VERSION 2.11
ENV KAFKA_VERSION 2.3.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"
ENV CYBERT_HOME /opt/cybert

ADD examples/streamz/scripts "$CYBERT_HOME"/scripts
ADD examples/streamz/data "$CYBERT_HOME"/data
ADD examples/streamz/python "$CYBERT_HOME"/python

RUN wget -q http://www.gtlib.gatech.edu/pub/apache/kafka/2.3.0/kafka_2.11-2.3.0.tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
        tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
        rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

# Download cybert apache model from huggingface for example
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/config.json "$CYBERT_HOME"/data/config.json
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/pytorch_model.bin "$CYBERT_HOME"/data/pytorch_model.bin

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN source activate rapids && \
    conda install -c pytorch "pytorch>=1.5" torchvision "cudf_kafka=${RAPIDS_VERSION}" "custreamz=${RAPIDS_VERSION}" "scikit-learn>=0.21" ipywidgets python-confluent-kafka transformers "seqeval=0.0.12" python-whois seaborn requests matplotlib pytest jupyterlab "openjdk=8.0.152" dask-cuda && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget

RUN source activate rapids \
  && conda install -n rapids jupyterlab-nvdashboard \
  && jupyter labextension install dask-labextension jupyterlab-nvdashboard

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

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/opt/cybert/scripts/entry.sh"]
