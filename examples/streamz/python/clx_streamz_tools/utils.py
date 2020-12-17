# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import yaml
import dask
import argparse
from datetime import datetime
from collections import deque
from distributed import Client
from elasticsearch import helpers
from dask_cuda import LocalCUDACluster

SINK_KAFKA = "kafka"
SINK_FS = "filesystem"
SINK_ES = "elasticsearch"

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


def create_dask_client():
    """
    Creates dask client
    :return: LocalCUDACluster: Dask client
    :rtype: object
    """
    print("Creating local cuda cluster as no dask scheduler is provided.")
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    return client


def kafka_sink(output_topic, parsed_df):
    """
    Writes cudf to Kafka topic.
    :param output_topic: Kafka topic name
    :type output_topic: str
    :param parsed_df: Parsed/Processed data
    :type parsed_df: cudf
    """
    worker = dask.distributed.get_worker()
    producer = worker.data["sink"]
    json_str = parsed_df.to_json(orient="records", lines=True)
    json_recs = json_str.split("\n")
    for json_rec in json_recs:
        try:
           producer.poll(0)
           producer.produce(output_topic, json_rec)
        except BufferError as be:
           producer.poll(0.1)
           print(be)
    producer.flush()


def fs_sink(config, parsed_df):
    """
    Writes cudf to Filesystem.
    :param config: Configuration which contains file format details
    :type output_topic: dict
    :param parsed_df: Parsed/Processed data
    :type parsed_df: cudf
    """
    filename = datetime.now().strftime(TIME_FORMAT) + config["file_extension"]
    filepath = os.path.join(config["output_dir"], filename)
    parsed_df.to_csv(filepath, sep=config["col_delimiter"], index=False)


def es_sink(config, parsed_df):
    """
    Writes cudf to Elasticsearch.
    :param config: Configuration which contains Elasticsearch cluster details
    :type config: dict
    :param parsed_df: Parsed/Processed data
    :type parsed_df: cudf
    """
    worker = dask.distributed.get_worker()
    es_client = worker.data["sink"]
    parsed_df["_index"] = config["index"]
    json_str = parsed_df.to_json(orient="records")
    docs = json.loads(json_str)
    pb = helpers.parallel_bulk(
        es_client, docs, chunk_size=10000, thread_count=10, queue_size=10
    )
    deque(pb, maxlen=0)


def calc_benchmark(processed_data, size_per_log):
    """
    Calculates benchmark for the streamz workflow
    :param processed_data: cudf dataframe
    :type processed_data: cudf
    :param size_per_log: 
    :type size_per_log: double
    :return: (time_diff, throughput_mbps, avg_batch_size): Benchmark output
    :rtype: (double, double, double)
    """
    t1 = int(round(time.time() * 1000))
    t2 = 0
    size = 0.0
    batch_count = 0
    # Find min and max time while keeping track of batch count and size
    for result in processed_data:
        (ts1, ts2, result_size) = (result[1], result[2], result[3])
        if ts1 == 0 or ts2 == 0:
            continue
        batch_count = batch_count + 1
        t1 = min(t1, ts1)
        t2 = max(t2, ts2)
        size += result_size * size_per_log
    time_diff = t2 - t1
    throughput_mbps = size / (1024.0 * time_diff) if time_diff > 0 else 0
    avg_batch_size = size / (1024.0 * batch_count) if batch_count > 0 else 0
    return (time_diff, throughput_mbps, avg_batch_size)


def load_yaml(yaml_file):
    """
    Returns a dictionary of a configuration contained in the given yaml file
    :param yaml_file: YAML configuration filepath
    :type yaml_file: str
    :return: config_dict: Configuration dictionary
    :rtype: dict
    """
    with open(yaml_file) as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    config_dict["sink"] = config_dict["sink"].lower()
    return config_dict


def init_dask_workers(worker, config, obj_dict=None):
    """
    Initalize for all dask workers
    :param worker: Dask worker
    :type worker: object
    :param config: Configuration which contains source and sink details
    :type config: dict
    :param obj_dict: Objects that are required to be present on every dask worker
    :type obj_dict: dict
    :return: worker: Dask worker
    :rtype: object
    """
    if obj_dict is not None:
        for key in obj_dict.keys():
            worker.data[key] = obj_dict[key]

    sink = config["sink"]
    if sink == SINK_KAFKA:
        import confluent_kafka as ck

        producer_conf = config["kafka_conf"]["producer_conf"]
        print("Producer conf: " + str(producer_conf))
        producer = ck.Producer(producer_conf)
        worker.data["sink"] = producer
    elif sink == SINK_ES:
        from elasticsearch import Elasticsearch

        es_conf = config["elasticsearch_conf"]
        if "username" in es_conf and "password" in es_conf:
            es_client = Elasticsearch(
                [
                    es_conf["url"].format(
                        es_conf["username"], es_conf["password"], es_conf["port"]
                    )
                ],
                use_ssl=True,
                verify_certs=True,
                ca_certs=es_conf["ca_file"],
            )
        else:
            es_client = Elasticsearch(
                [{"host": config["elasticsearch_conf"]["url"]}],
                port=config["elasticsearch_conf"]["port"],
            )
        worker.data["sink"] = es_client
    elif sink == SINK_FS:
        print(
            "Streaming process will write the output to location '{}'".format(
                config["output_dir"]
            )
        )
    else:
        print(
            "No valid sink provided in the configuration file. Please provide kafka/elasticsearch/filsesystem"
        )
        sys.exit(-1)

    print("Successfully initialized dask worker " + str(worker))
    return worker


def create_dir(sink, dir_path):
    """
    :param sink: Sink type mentioned in the configuration file
    :type sink: str
    :param dir_path: Directory that needs to be created
    :type dir_path: str
    """
    if sink == SINK_FS and not os.path.exists(dir_path):
        print("Creating directory '{}'".format(dir_path))
        os.makedirs(dir_path)


def parse_arguments():
    """
    Parse script arguments
    """
    parser = argparse.ArgumentParser(
        description="Streamz and Dask. \
                     Data will be read from the input kafka topic, \
                     processed using clx streamz workflows."
    )
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-c", "--conf", help="Source and Sink configuration filepath"
    )
    parser.add_argument("-m", "--model", help="Model filepath")
    parser.add_argument("-l", "--label_map", help="Label map filepath")
    parser.add_argument(
        "--max_batch_size",
        default=1000,
        type=int,
        help="Max batch size to read from kafka",
    )
    required_args.add_argument(
        "--poll_interval", type=str, help="Polling interval (ex: 60s)"
    )
    parser.add_argument(
        "--benchmark",
        help="Captures benchmark, including throughput estimates, with provided avg log size in KB. (ex: 500 or 0.1)",
        type=float,
    )
    args = parser.parse_args()
    return args
