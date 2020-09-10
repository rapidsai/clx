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
docker pull gpuci/miniconda-cuda:10.2-devel-ubuntu18.04
```

Then create a new image using the Dockerfile provided. This docker image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/).

```
cd clx/
docker build -f examples/streamz/Dockerfile -t cybert-streamz .
```

Create a new container using the image above. When running this container, it will automatically trigger processing of sample data by cyBERT using streamz. See output below

##### Preferred - Docker CE v19+ and nvidia-container-toolkit
```
docker run -it --gpus '"device=0,1,2"' -p 8787:8787 -v /path/to/dataset:/path/to/dataset -v /path/to/model.pth:/path/to/model.pth -v /path/to/label.txt:/path/to/label.json --name cybert-streamz -d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.json \
--cuda_visible_devices 0,1,2 \
--poll_interval 1s \
--max_batch_size 1000 \
--data /path/to/dataset
```

##### Legacy - Docker CE v18 and nvidia-docker2
```
docker run -it --runtime=nvidia -p 8787:8787 -v /path/to/dataset:/path/to/dataset -v /path/to/model.pth:/path/to/model.pth -v /path/to/label.txt:/path/to/label.json --name cybert-streamz -d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.json \
--cuda_visible_devices 0,1,2 \
--poll_interval 1s \
--max_batch_size 1000 \
--data /path/to/dataset
```

Parameters
- `broker`* - Host and port where kafka broker is running
- `group_id`* - Kafka [group id](https://docs.confluent.io/current/installation/configuration/consumer-configs.html#group.id) that uniquely identifies the streamz data consumer.
- `input_topic` - The name for the input topic to send the input dataset. Any name can be indicated here.
- `output_topic` - The name for the output topic to send the output data. Any name can be indicated here.
- `model_file` - The path to your model file
- `label_file` - The path to your label file
- `cuda_visible_devices` - List of gpus use to run streamz with Dask. The gpus should be equal to or a subset of devices indicated within the docker run command (in the example above device list is set to `'"device=0,1,2"'`)
- `poll_interval`* - Interval (in seconds) to poll the Kafka input topic for data
- `max_batch_size`* - Max batch size of data (max number of logs) to ingest into streamz with each `poll_interval`
- `data` - The input dataset to use for this streamz example

``*`` = More information on these parameters can be found in the streamz [documentation](https://streamz.readthedocs.io/en/latest/api.html#streamz.from_kafka_batched).

View the data processing activity on the dask dashboard by visiting `localhost:8787` or `<host>:8787`

View the cyBERT script output in the container logs

```
docker logs cybert-streamz
```

Processed data will be pushed to the kafka topic named `output`. To view all processed output run:
```
docker exec cybert-streamz bash -c 'source activate rapids && $KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic output --from-beginning'
```

##### Benchmark

To capture benchmarks add the benchmark flag (`--benchmark`) to the docker run command

To print benchmark to the docker logs send a SIGINT signal to the running cybert process
```
# To get the PID
$ docker exec cybert-streamz ps aux | grep cybert.py | awk '{print $2}'
# Kill process
$ docker exec cybert-streamz kill -SIGINT <pid>
$ docker logs cybert-streamz
```

