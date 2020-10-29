# Getting Started with CLX and Streamz

This is a guide on how [CLX](https://github.com/rapidsai/clx) and [Streamz](https://streamz.readthedocs.io/en/latest/) can be used to build a streaming inference pipeline.

Streamz has the ability to read from [Kafka](https://kafka.apache.org/) directly into [Dask](https://dask.org/) allowing for computation on a multi-core or cluster environment. This approach is best used for instances in which you hope to increase processing speeds with streaming data.

A selection of workflows such as cyBERT and DGA detection inferencing are implemented in clx streamz. Here we share an example in which we demonstrate how to read Apache log data from Kafka, perform log parsing using CLX cyBERT and publish result data back to Kafka. Similarly, also for DGA detection.

## Build Quickstart Docker Image
For convenience, you can build a Docker image that will include a working environment that's ready for running your pipeline. This image will contain all needed components including [Kafka](https://kafka.apache.org/) and [Zookeeper](https://zookeeper.apache.org/).

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

## Running CLX_Streamz Pipelines

A docker container is created using the image above. The 'docker run' format to build your container is shown below. Note: volume binding to the container is an optional argument.

**Preferred - Docker CE v19+ and nvidia-container-toolkit**

```
docker run -it \
    -p 8787:8787 \
    -p 8888:8888 \
    -v <your_volume_binding_host_directory_path>:<your_volume_binding_container_directory_path> \
    --gpus '"device=0,1,3"' \
    --name clx_streamz \
    -d clx-streamz:latest
```

**Legacy - Docker CE v18 and nvidia-docker2**

```
docker run -it \
    -p 8787:8787 \
    -p 8888:8888 \
     -v <your_volume_binding_host_directory_path>:<your_volume_binding_container_directory_path> \
    --runtime=nvidia \
    --name clx_streamz \
    -d clx-streamz:latest
```

The Dockerfile contains an ENTRYPOINT which calls [entrypoint.sh](https://github.com/rapidsai/clx/blob/branch-0.17/examples/streamz/scripts/entrypoint.sh) to:
1. Configure and install Kafka
2. Run Kafka broker on `localhost:9092` and Zookeeper on `localhost:2181`
3. Start Dask Scheduler on `localhost:8786`
4. Start Dask CUDA Worker (one worker per GPU)

**Steps to run workflow are followed below**

1. Access docker container
   ```
   docker exec -it clx_streamz bash
   ```
   
2. Create kafka topics for the clx_streamz workflows that we want to run and publish input data. 

    ```
    bash $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh --help
    ```
    ```
    Usage: kafka_topic_setup.sh [POS]... [ARG]...

    Example-1: bash kafka_topic_setup.sh -b localhost:9092 -i cybert_input -o cybert_output -d /opt/clx_streamz/data/cybert_input.csv
    Example-2: bash kafka_topic_setup.sh -i cybert_input -o cybert_output -d /opt/clx_streamz/data/cybert_input.csv
    Example-2: bash kafka_topic_setup.sh -i cybert_input -o cybert_ouput
    
    This script configures the kafka topic, such as creating and loading data or just the topic creation.
    
    Positional:
      -b,  --broker             Kafka broker. Default value is localhost:9092
      -i,  --input_topic	    Input kafka topic
      -o,  --output_topic       Output kafka topic
      -d,  --data_path          Input data filepath
    
      -h, --help                Print this help
    ```
    ```
    bash $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
        -b localhost:9092 \
        -i <input_topic> \
        -o <output_topic> \
        -d <data_filepath>
    ```
    
3. Start workflow with required arugments
    
    ```
    python $CLX_STREAMZ_HOME/python/<workflow_script> \
        --broker <host:port> \
        --input_topic <input_topic> \
        --output_topic <output_topic> \
        --group_id <kafka_consumer_group_id> \
        --model <model filepath> \
        --label_map <labels filepath> \
        --poll_interval <poll_interval> \
        --max_batch_size <max_batch_size> \
        --dask_scheduler <hostname:port> \
        --benchmark <avg log size>
    ```
    **Parameters:**
    - `broker`* - Host and port where kafka broker is running. 
    - `group_id`* - Kafka [group id](https://docs.confluent.io/current/installation/configuration/consumer-configs.html#group.id) that uniquely identifies the streamz data consumer.
    - `input_topic` - The name for the input topic to consumer data.
    - `output_topic` - The name for the output topic to send the output data.
    - `model_file` - The path to your model file
    - `label_file` - The path to your label file
    - `poll_interval`* - Interval (in seconds) to poll the Kafka input topic for data (Ex: 60s)
    - `max_batch_size`* - Max batch size of data (max number of logs) to ingest into streamz with each `poll_interval` 
    - `dask_scheduler` - Dask scheduler address. If not provided a new local dask cuda cluster will be created.
    - `benchmark` - To capture benchmarks add the benchmark flag along with average log size (kb), for throughput (mb/s) and average batch size (mb) estimates. To print benchmark just use `ctrl+c`

    ``*`` = More information on these parameters can be found in the streamz [documentation](https://streamz.readthedocs.io/en/latest/api.html#streamz.from_kafka_batched).


View the data processing activity on the dask dashboard by visiting http://localhost:8787 or `<host>:8787`

Processed data will be pushed to the give kafka output topic. To view all processed output run:

```
docker exec clx_streamz bash -c 'source activate rapids && $KAFKA_HOME/bin/kafka-console-consumer.sh --bootstrap-server <broker> --topic <output_topic> --from-beginning'
```

Your Quickstart Docker container includes the data and models required to run cyBERT and DGA detection stream processing workflows.

## Run cyBERT Streamz Example on Apache Logs
```
bash $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
    -i cybert_input \
    -o cybert_output \
    -d $CLX_STREAMZ_HOME/data/apache_raw_sample_1k.txt
```
```
python $CLX_STREAMZ_HOME/python/cybert.py \
    --broker localhost:9092 \
    --input_topic cybert_input \
    --output_topic cybert_output \
    --group_id streamz \
    --model $CLX_STREAMZ_HOME/ml/models/cybert_pytorch_model.bin \
    --label_map $CLX_STREAMZ_HOME/ml/labels/cybert_config.json \
    --poll_interval 1s \
    --max_batch_size 500 \
    --dask_scheduler localhost:8786 \
    --benchmark 10
```

## Run DGA Streamz Example on Sample Domains

```
bash $CLX_STREAMZ_HOME/scripts/kafka_topic_setup.sh \
    -i dga_detection_input \
    -o dga_detection_output \
    -d $CLX_STREAMZ_HOME/data/dga_detection_input.jsonlines
```
```
python $CLX_STREAMZ_HOME/python/dga_detection.py \
    --broker localhost:9092 \
    --input_topic dga_detection_input \
    --output_topic dga_detection_output \
    --group_id streamz \
    --model $CLX_STREAMZ_HOME/ml/models/dga_detection_pytorch_model.bin \
    --poll_interval 1s \
    --max_batch_size 500 \
    --dask_scheduler localhost:8786 \
    --benchmark 10
```
