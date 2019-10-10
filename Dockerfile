ARG repository=rapidsai
ARG version=cuda10.0-runtime-ubuntu18.04

FROM rapidsai/${repository}:${version}

ADD . /clx/

SHELL ["/bin/bash", "-c"]
RUN source activate rapids \
    && cd /clx \
    && python setup.py install

WORKDIR /clx
CMD source activate rapids && sh /rapids/notebooks/utils/start-jupyter.sh