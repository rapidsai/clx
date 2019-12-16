ARG repository=rapidsai-dev-nightly
ARG version=0.11-cuda10.0-devel-ubuntu18.04-py3.7

FROM rapidsai/${repository}:${version}

ADD . /rapids/clx/

SHELL ["/bin/bash", "-c"]

RUN source activate rapids \
    && conda install panel=0.6.* geopandas pyppeteer \
    && cd /rapids \
    && git clone https://github.com/rapidsai/cuxfilter.git /rapids/cuxfilter \
    && git clone https://github.com/rapidsai/cudatashader.git  /rapids/cudatashader \
    && cd /rapids/cuxfilter/python \
    && pip install -e . \
    && cd /rapids/cudatashader \
    && pip install -e . \
    && cd /rapids/clx \
    && python setup.py install

WORKDIR /rapids/clx

