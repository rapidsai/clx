ARG repository=rapidsai-dev-nightly
ARG version=0.11-cuda10.0-devel-ubuntu18.04-py3.7

FROM rapidsai/${repository}:${version}

ADD . /clx/

SHELL ["/bin/bash", "-c"]
RUN source activate rapids \
    && cd /clx \
    && python setup.py install

WORKDIR /clx
CMD source activate rapids && sh /rapids/notebooks/utils/start-jupyter.sh