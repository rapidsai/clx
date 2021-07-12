# Copyright (c) 2020, NVIDIA CORPORATION.

# An integration test & dev container which builds and installs CLX from default branch
ARG RAPIDS_VERSION=21.06
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

ENV SCALA_VERSION 2.13
ENV KAFKA_VERSION 2.7.0
ENV KAFKA_HOME /opt/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION"
ENV CLX_STREAMZ_HOME /opt/clx_streamz

ADD examples/streamz/scripts "$CLX_STREAMZ_HOME"/scripts
ADD examples/streamz/python "$CLX_STREAMZ_HOME"/python
ADD examples/streamz/resources "$CLX_STREAMZ_HOME"/resources

RUN mkdir -p "$CLX_STREAMZ_HOME"/ml/models/cybert && \
	mkdir "$CLX_STREAMZ_HOME"/ml/models/dga && \
	mkdir "$CLX_STREAMZ_HOME"/data

RUN wget -q https://downloads.apache.org/kafka/$KAFKA_VERSION/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -O /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz && \
    tar xfz /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz -C /opt && \
    rm /tmp/kafka_"$SCALA_VERSION"-"$KAFKA_VERSION".tgz

# Download cybert apache model from huggingface for example
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/config.json -O "$CLX_STREAMZ_HOME"/ml/models/cybert/config.json
RUN wget -q http://models.huggingface.co.s3.amazonaws.com/bert/raykallen/cybert_apache_parser/pytorch_model.bin -O "$CLX_STREAMZ_HOME"/ml/models/cybert/pytorch_model.bin

# Download apache logs
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/apache_raw_sample_1k.txt -O "$CLX_STREAMZ_HOME"/data/apache_raw_sample_1k.txt

# Download dga detection model and sample input data
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/dga_detection_pytorch_model.bin -O "$CLX_STREAMZ_HOME"/ml/models/dga/pytorch_model.bin
RUN wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/cyber/clx/dga_detection_input.jsonlines -O "$CLX_STREAMZ_HOME"/data/dga_detection_input.jsonlines

# Zookeeper
EXPOSE 2181

# Kafka
EXPOSE 9092

RUN source activate rapids && \
    conda install -c pytorch "pytorch=1.7.1" torchvision "cudf_kafka=${RAPIDS_VERSION}" "custreamz=${RAPIDS_VERSION}" "scikit-learn>=0.21" "nodejs>=12" ipywidgets python-confluent-kafka "transformers=4.*" "seqeval=1.2.2" python-whois seaborn requests matplotlib pytest jupyterlab "openjdk=8.0.152" dask-cuda && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget && \
    pip install elasticsearch && \
    pip install elasticsearch-async && \
    pip install "git+https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment.git#egg=slashnext-phishing-ir&subdirectory=Python SDK/src"

# slashnext download and install
RUN source activate rapids && \
    git clone https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment.git /opt/slashnext && \
    pip install /opt/slashnext/Python\ SDK/src/
    
RUN source activate rapids \
  && conda install -n rapids jupyterlab-nvdashboard \
  && jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension jupyterlab-nvdashboard

# clx build/install
RUN source activate rapids && \
    cd /rapids/clx/python && \
    python setup.py install

# clx_streamz_tools install
RUN source activate rapids && \
	cd "$CLX_STREAMZ_HOME"/python && \
	python setup.py install
	
WORKDIR /rapids/clx

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/opt/clx_streamz/scripts/entrypoint.sh"]

CMD [ "/bin/bash" ]
