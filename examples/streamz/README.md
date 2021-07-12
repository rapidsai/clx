# Getting Started with CLX and Streamz

This is a guide on how [CLX](https://github.com/rapidsai/clx) and [Streamz](https://streamz.readthedocs.io/en/latest/) can be used to build a streaming inference pipeline.

Streamz has the ability to read from [Kafka](https://kafka.apache.org/) directly into [Dask](https://dask.org/) allowing for computation on a multi-core or cluster environment. This approach is best used for instances in which you hope to increase processing speeds with streaming data.

A selection of workflows such as cyBERT and DGA detection inferencing are implemented in CLX streamz. Here we share an example in which we demonstrate how to read Apache log data from Kafka, perform log parsing using CLX cyBERT and publish result data back to Kafka. Similarly, also for DGA detection.

## Build Quickstart Docker Image
For convenience, you can build a Docker image that will include a working environment that's ready for running your pipeline. This image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/).

Prerequisites:
* NVIDIA Pascal™ GPU architecture or better
* CUDA 11.0 compatible NVIDIA driver
* Ubuntu 16.04/18.04 or CentOS 7
* Docker CE v18+
* nvidia-docker v2+

Run the following to build the image:

`
docker build -f examples/streamz/Dockerfile -t clx-streamz:latest .
`

## Create Docker Container

A Docker container is created using the image above. The 'docker run' format to build your container is shown below.  Note: volume binding to the container is an optional argument.

**Preferred - Docker CE v19+ and nvidia-container-toolkit**

```
docker run -it \
    -p 9787:8787 \
    -p 9888:8888 \
    -v <your_volume_binding_host_directory_path>:<your_volume_binding_container_directory_path> \
    --gpus '"device=0,1,2"' \
    --name clx_streamz \
    -d clx-streamz:latest
```

**Legacy - Docker CE v18 and nvidia-docker2**

```
docker run -it \
    -p 9787:8787 \
    -p 9888:8888 \
     -v <your_volume_binding_host_directory_path>:<your_volume_binding_container_directory_path> \
    --runtime=nvidia \
    --name clx_streamz \
    -d cybert-streamz:latest
```

The Dockerfile contains an ENTRYPOINT which calls [entrypoint.sh](https://github.com/rapidsai/clx/blob/branch-0.17/examples/streamz/scripts/entrypoint.sh) to:
1. Configure and install Kafka
2. Run Kafka broker on `localhost:9092` and Zookeeper on `localhost:2181`
3. Creates (cyBERT and DGA detection) specific input and output kafka topics and publishes sample input data 

Your Quickstart Docker container includes the data and models required to run cyBERT and DGA detection stream processing workflows. Note: we can run multiple workflows on the same container in parallel.

## Run cyBERT Streamz Example on Apache Logs
```
docker exec clx_streamz bash -c 'source activate rapids \
    && python $CLX_STREAMZ_HOME/python/cybert.py \
    --conf $CLX_STREAMZ_HOME/resources/cybert.yaml \
    --model $CLX_STREAMZ_HOME/ml/models/cybert/pytorch_model.bin \
    --label_map $CLX_STREAMZ_HOME/ml/models/cybert/config.json \
    --poll_interval 1s \
    --max_batch_size 500'
```

## Run DGA Streamz Example on Sample Domains
```
docker exec clx_streamz bash -c 'source activate rapids \
    && python $CLX_STREAMZ_HOME/python/dga_detection.py \
    --conf $CLX_STREAMZ_HOME/resources/dga_detection.yaml \
    --model $CLX_STREAMZ_HOME/ml/models/dga/pytorch_model.bin \
    --poll_interval 1s \
    --max_batch_size 500'
```

Processed data will be pushed to the given kafka output topic. To view all processed output run:

```
docker exec clx_streamz bash -c 'source activate rapids \
       && $KAFKA_HOME/bin/kafka-console-consumer.sh \
       --bootstrap-server <broker> \
       --topic <output_topic> \
       --from-beginning'
```

View the data processing activity on the dask dashboard by visiting http://localhost:9787 or `<host>:9787`

## Capturing Benchmarks
To capture benchmarks add the benchmark flag along with average log size (kb), for throughput (mb/s) and average batch size (mb) estimates, to the Docker run command above. In this case, we are benchmarking the cyBERT workflow with the commands below. Similarly, we can also do it for the DGA detection workflow.

```
docker exec clx_streamz bash -c 'source activate rapids \
    && python $CLX_STREAMZ_HOME/python/cybert.py \
    --conf $CLX_STREAMZ_HOME/resources/cybert.yaml \
    --model $CLX_STREAMZ_HOME/ml/models/cybert/pytorch_model.bin \
    --label_map $CLX_STREAMZ_HOME/ml/models/cybert/config.json \
    --poll_interval 1s \
    --max_batch_size 500 \
    --benchmark 20' \
    > cybert_workflow.log 2>&1 &
```

To print benchmark, send a SIGINT signal to the running cybert process.
```
# To get the PID
$ docker exec clx_streamz ps aux | grep "cybert\.py" | awk '{print $2}'
# Kill process
$ docker exec clx_streamz kill -SIGINT <pid>
$ less cybert_workflow.log
```

## Steps to Run Workflow with Custom Arguments

1. Create kafka topics for the clx_streamz workflows that you want to run and publish input data.

    ```
    docker exec clx_streamz /bin/bash -c 'source activate rapids \
        && $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
        -b localhost:9092 \
        -i <input_topic> \
        -o <output_topic> \
        -d <data_filepath>'
    ```
    
2. Start workflow 
    
    ```
    docker exec clx_streamz bash -c 'source activate rapids \
        && python $CLX_STREAMZ_HOME/python/<workflow_script> \
        --conf <configuration filepath> \
        --model <model filepath> \
        --label_map <labels filepath> \
        --poll_interval <poll_interval> \
        --max_batch_size <max_batch_size> \
        --benchmark <avg log size>'
    ```
    **Parameters:**
    - `conf` - The path to specify source and sink configuration properties.
    - `model_file` - The path to your model file
    - `label_file` - The path to your label file
    - `poll_interval`* - Interval (in seconds) to poll the Kafka input topic for data (Ex: 60s)
    - `max_batch_size`* - Max batch size of data (max number of logs) to ingest into streamz with each `poll_interval`
    - `benchmark` - To capture benchmarks add the benchmark flag along with average log size (kb), for throughput (mb/s) and average batch size (mb) estimates.

    ``*`` = More information on these parameters can be found in the streamz [documentation](https://streamz.readthedocs.io/en/latest/api.html#streamz.from_kafka_batched).
    
**Configuration File Properties**
- `cudf_engine` - This value determines whether to use cudf engine while consuming messages using streamz API
- `kafka_conf`
   - `input_topic` - Consumer Kafka topic name
   - `output_topic` - Publisher Kafka topic name
   - `n_partitions` - Number of partitions in the consumer Kafka topic
    - `producer_conf` - User can specify any valid Kafka producer configuration within this block
      - `bootstrap.servers` - Kafka brokers Ex: localhost:9092, localhost2:9092
      - `session.timeout.ms` - Message publishing timout
      - `queue.buffering.max.messages` - Max number of messages that can hold in the queue
      - `...`
   - `consumer_conf` - User can specify any valid Kafka consumer configuration within this block
      - `bootstrap.servers` - Kafka brokers Ex: localhost:9092, localhost2:9092
      - `group.id` - Kafka consumer group id
      - `session.timeout.ms` - Message consuming timout
      - `...`
- `elasticsearch_conf` - Elasticsearch sink configuration
   - `url` - Elasticsearch service url
   - `port` - Elasticsearch service port
   - `cafile` - Path to pem file
   - `username` - Username
   - `password` - Password
   - `index` - Name to index the documents
- `sink` - Sink to write processed data Ex: "kafka" or "elasticsearch" or "filesystem"

**Note: Below properties are used only when sink is set to "filesystem"**
- `col_delimiter` - Column delimiter Ex: ","
- `file_extension` - File extension Ex: ".csv"
- `output_dir` - Output filepath
```
