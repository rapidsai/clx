# An integration test & dev container which builds and installs CLX from default branch
ARG RAPIDS_VERSION=22.06
ARG CUDA_VERSION=11.5
ARG CUDA_SHORT_VERSION=${CUDA_VERSION}
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.8
FROM rapidsai/rapidsai-dev-nightly:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-devel-${LINUX_VERSION}-py${PYTHON_VERSION}

# Add everything from the local build context
ADD . /rapids/clx/
RUN chmod -R ugo+w /rapids/clx/

RUN source activate rapids && \
    gpuci_mamba_retry install -y -n rapids \
        "cudf_kafka=${RAPIDS_VER}" \
        "custreamz=${RAPIDS_VER}" \
        scikit-learn>=0.21 \
        nodejs>=12 \
        ipywidgets \
        python-confluent-kafka \
        seqeval \
        python-whois \
        seaborn \
        requests \
        matplotlib \
        pytest \
        jupyterlab=3.0 \
        faker && \
    pip install -U torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip install "git+https://github.com/rapidsai/cudatashader.git" && \
    pip install mockito && \
    pip install wget && \
    pip install "git+https://github.com/slashnext/SlashNext-URL-Analysis-and-Enrichment.git#egg=slashnext-phishing-ir&subdirectory=Python SDK/src"

# clx build/install
RUN source activate rapids && \
    cd /rapids/clx/python && \
    python setup.py install

WORKDIR /rapids/clx
