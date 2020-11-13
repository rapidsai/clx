# Getting Started with CLX and Streamz

This is a guide on how [CLX](https://github.com/rapidsai/clx) and [Streamz](https://streamz.readthedocs.io/en/latest/) can be used to build a streaming inference pipeline. We will use CLX's cyBERT for inference.

Streamz has the ability to read from [Kafka](https://kafka.apache.org/) directly into [Dask](https://dask.org/) allowing for computation on a multi-core or cluster environment. This approach is best used for instances in which you hope to increase processing speeds with streaming data.

Here we share an example in which we demonstrate how to read Apache log data from Kafka, perform log parsing using CLX cyBERT and publish result data back to Kafka.

## Build Quickstart Docker Image
For convenience, you can build a Docker image that will include a working environment that's ready for running your pipeline. This image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/). An example pipeline is also included on the container to demonstrate using cyBERT and Streamz to parse a sample of Apache logs.

Prerequisites:
* NVIDIA Pascal™ GPU architecture or better
* CUDA 10.1+ compatible NVIDIA driver
* Ubuntu 16.04/18.04 or CentOS 7
* Docker CE v18+
* nvidia-docker v2+

Run the following to build the image:

`
docker build -f examples/streamz/Dockerfile -t clx-streamz:latest .
`

## Running cyBERT Inference Pipeline

For simplicity, our inference pipeline is started when the Docker container is created using the above image. The following shows the format of the `docker run` to create your container.

**Preferred - Docker CE v19+ and nvidia-container-toolkit**
```
docker run -it
--gpus '"device=0,1,2"' \
-p 8787:8787 \
--name cybert-streamz
-d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.json \
--poll_interval 1s \
--max_batch_size 1000 \
--data /path/to/dataset
```

**Legacy - Docker CE v18 and nvidia-docker2**
```
docker run -it \
--runtime=nvidia \
-p 8787:8787 \
--name cybert-streamz
-d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.json \
--poll_interval 1s \
--max_batch_size 1000 \
--data /path/to/dataset
```

**Parameters:**
- `broker`* - Host and port where kafka broker is running
- `group_id`* - Kafka [group id](https://docs.confluent.io/current/installation/configuration/consumer-configs.html#group.id) that uniquely identifies the streamz data consumer.
- `input_topic` - The name for the input topic to send the input dataset. Any name can be indicated here.
- `output_topic` - The name for the output topic to send the output data. Any name can be indicated here.
- `model_file` - The path to your model file
- `label_file` - The path to your label file
- `poll_interval`* - Interval (in seconds) to poll the Kafka input topic for data
- `max_batch_size`* - Max batch size of data (max number of logs) to ingest into streamz with each `poll_interval`
- `data` - The input dataset to use for this streamz example. This is a filepath to text file containing lines of text to be processed for inference (i.e. log file)

``*`` = More information on these parameters can be found in the streamz [documentation](https://streamz.readthedocs.io/en/latest/api.html#streamz.from_kafka_batched).


The Dockerfile contains an ENTRYPOINT which calls [entry.sh](https://github.com/rapidsai/clx/blob/branch-0.16/examples/streamz/scripts/entry.sh) to:
1. Download, install and configure Kafka
2. Run Kafka and Zookeeper
3. Create input and output Kafka topics
4. Read input data into input Kafka topic
5. Start cyBERT distributed inference using Dask (one worker per GPU)


View the data processing activity on the dask dashboard by visiting http://localhost:8787 or `<host>:8787`

View the cyBERT script output in the container logs

```
docker logs --follow cybert-streamz
```

Processed data will be pushed to the kafka topic named `output`. To view all processed output run:
```
docker exec cybert-streamz bash -c 'source activate rapids && $KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic output --from-beginning'
```

## Capturing Benchmarks

To capture benchmarks add the benchmark flag along with average log size (kb), for throughput (mb/s) and average batch size (mb) estimates, to the docker run command above
```
docker run -it 
--gpus '"device=0,1,2"' \
-p 8787:8787 \
--name cybert-streamz -d cybert-streamz:latest \
--broker localhost:9092 \
--group_id streamz \
--input_topic input \
--output_topic output \
--model_file /path/to/model.pth \
--label_file /path/to/label.json \
--poll_interval 1s \
--max_batch_size 1000 \
--data /path/to/dataset \
--benchmark 20
```

To print benchmark to the docker logs send a SIGINT signal to the running cybert process
```
# To get the PID
$ docker exec cybert-streamz ps aux | grep "cybert\.py" | awk '{print $2}'
# Kill process
$ docker exec cybert-streamz kill -SIGINT <pid>
$ docker logs cybert-streamz
```

## Run cyBERT Streamz Example on Apache Logs

Your Quickstart Docker container includes the data and model required to run cyBERT stream processing on a sample of Apache logs.

The following command can be used to run the pipeline over two GPUs:
```
docker run -it --gpus '"device=0,1"' \
-p 9787:8787 \
--name cybert-streamz \
-d clx-streamz:latest \
-b localhost:9092 \
-g streamz \
-i input \
-o output \
-d /opt/cybert/data/apache_raw_sample_1k.txt \
-m /opt/cybert/data/pytorch_model.bin \
-l /opt/cybert/data/config.json \
-p 1s \
--max_batch_size 500
```