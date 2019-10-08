# Contributing to CLX

## Code formatting

CLX uses pre-commit hooks to assist with code formatting. Install [pre-commit](https://pre-commit.com/) using one of the following commands

```
conda install pre_commit
```
```
pip install pre-commit
```

Then run from the root `clx` directory
```
pre-commit install
```

## Setting up your build environment

The base of the CLX environment or docker container is from the [RAPIDS release containers](https://hub.docker.com/r/rapidsai/rapidsai/).
In any case you wish to use a different build environment, such as the [RAPIDS nightly containers](https://hub.docker.com/r/rapidsai/rapidsai-nightly), alter your docker build command as such:

```
docker pull rapidsai/rapidsai-nightly:cuda10.0-runtime-ubuntu18.04

docker build -t clx:dev --build-arg repository=rapidsai-nightly --build-arg version=cuda10.0-runtime-ubuntu18.04 .

docker run --runtime=nvidia \
  --rm -it \
  -p 8888:8888 \
  -p 8787:8787 \
  -p 8686:8686 \
  clx:dev
```

CLX uses pytest to run unit tests and requires several additional test dependencies.

To install pytest
```
pip install pytest
```

Next install test dependencies. The following cupy dependency is for CUDA 10.0 - please install whichever version applies to your environment.
```
pip install mockito && pip install cupy-cuda100
```

## Building from Source

First run tests
```
cd clx/
pytest
```

Then build
```
python setup.py install
```