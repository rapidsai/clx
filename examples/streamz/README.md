# CLX cyBERT and Streamz

This example shows how to integrate CLX cyBERT with [Streamz](https://streamz.readthedocs.io/en/latest/). Streamz has the ability to read from [Kafka](https://kafka.apache.org/) directly into [Dask](https://dask.org/) allowing for computation on a multi-core or cluster environment. This approach is best used for instances in which you hope to increase processing speeds with streaming data.

Here we share an example in which we demonstrate how to read [Windows Event Logs](https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/) from Kafka, perform log parsing using CLX cyBERT and publish result data back to kafka.


## Getting started

First pull the latest version of rapids suitable for your environment

```
docker pull rapidsai/rapidsai-dev-nightly:0.12-cuda10.1-devel-ubuntu18.04-py3.6
```

Then create a new image using the Dockerfile provided. This docker image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/).

```
cd clx/
docker build -f examples/streamz/Dockerfile -t cybert-streamz .
```

Create a new container using the image above. When running this container, it will automatically trigger processing of sample data by cyBERT using streamz. See output below

```
docker run -it --gpus '"device=0"' --name cybert-streamz -d cybert-streamz:latest
```
*NOTE: To run using your own dataset use the following command, replacing `/path/to/data/dir` with the path to your data directory on your host machine.
And replacing `/path/to/data/dir/my_sample.csv` with full path to the specific data file within that directory.
```
docker run -it --gpus '"device=0"' -v /path/to/data/dir:/path/to/data/dir --name cybert-streamz -d cybert-streamz:latest /path/to/data/dir/my_sample.csv
```

View the output in the logs

```
docker logs cybert-streamz