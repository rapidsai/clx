ARG version=cuda9.2-runtime-ubuntu16.04

FROM rapidsai/rapidsai:${version}

ADD . /clx/

SHELL ["/bin/bash", "-c"]
RUN source activate rapids \
    && cd /clx \
    && python setup.py install

WORKDIR /clx
CMD source activate rapids && sh /rapids/notebooks/utils/start-jupyter.sh
