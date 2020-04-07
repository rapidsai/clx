# CLX cyBERT and Streamz

This example shows how to integrate CLX cyBERT with [Streamz](https://streamz.readthedocs.io/en/latest/). Streamz has the ability to read from [Kafka](https://kafka.apache.org/) directly into [Dask](https://dask.org/) allowing for computation on a multi-core or cluster environment. This approach is best used for instances in which you hope to increase processing speeds with streaming data.

Here we share an example in which we demonstrate how to read [Windows Event Logs](https://www.ultimatewindowssecurity.com/securitylog/encyclopedia/) from Kafka, perform log parsing using CLX cyBERT and publish result data back to kafka.

## Getting started
### Prerequisites

- NVIDIA Pascal™ GPU architecture or better
- CUDA 9.2 or 10.0 compatible NVIDIA driver
- Ubuntu 16.04/18.04 or CentOS 7
- Docker CE v18+
- nvidia-docker v2+

First pull the latest version of rapids suitable for your environment

```
docker pull rapidsai/rapidsai-dev-nightly:0.13-cuda10.1-devel-ubuntu18.04-py3.6
```

Then create a new image using the Dockerfile provided. This docker image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/).

```
cd clx/
docker build -f examples/streamz/Dockerfile -t cybert-streamz .
```

Create a new container using the image above. When running this container, it will automatically trigger processing of sample data by cyBERT using streamz. See output below

##### Preferred - Docker CE v19+ and nvidia-container-toolkit
```
docker run -it --gpus '"device=0"' -p 8787:8787 -v /home/nfs/brhodes/cyshare:/rapids/cyshare -v /home/nfs/brhodes/cybermount/datasets/apache/:/datasets/apache --name cybert-streamz -d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.yaml \
--data /path/to/dataset (optional)
```

##### Legacy - Docker CE v18 and nvidia-docker2
```
docker run -it --runtime=nvidia -p 8787:8787 -v /home/nfs/brhodes/cyshare:/rapids/cyshare -v /home/nfs/brhodes/cybermount/datasets/apache/:/datasets/apache --name cybert-streamz -d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.yaml \
--data /path/to/dataset (optional)
```

View the output in the logs

```
docker logs cybert-streamz
```

Output will be pushed to the kafka topic named `output`. To view the output, log into the container and run 
```
$KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --from-beginning --topic output
```
