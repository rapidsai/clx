ARG version=cuda9.2-runtime-ubuntu16.04

FROM rapidsai/rapidsai:${version}

ADD . /rapidscyber/

SHELL ["/bin/bash", "-c"]
RUN source activate rapids \
    && cd /rapidscyber \
    && python setup.py install

WORKDIR /rapidscyber
CMD source activate rapids && sh /rapids/notebooks/utils/start-jupyter.sh
