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

import argparse
import gc
import signal
import sys
import time

import confluent_kafka as ck
import cudf
import dask
import pandas as pd
import torch
from dask_cuda import LocalCUDACluster
from distributed import Client
from streamz import Stream


def inference(messages):
    # Messages will be received and run through cyBERT inferencing
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    size = 0
    df = cudf.DataFrame()
    if type(messages) == str:
        df["stream"] = [messages.decode("utf-8")]
    elif type(messages) == list and len(messages) > 0:
        df["stream"] = [msg.decode("utf-8") for msg in messages]
    else:
        print("ERROR: Unknown type encountered in inference")
    parsed_df, confidence_df = worker.data["cybert"].inference(df["stream"])
    size = len(df)
    torch.cuda.empty_cache()
    gc.collect()
    return (parsed_df, confidence_df, batch_start_time, size)


def sink_to_kafka(processed_data):
    # Parsed data and confidence scores will be published to provided kafka producer
    parsed_df = processed_data[0]
    confidence_df = processed_data[1]
    producer = ck.Producer(producer_conf)
    confidence_df = confidence_df.add_suffix("_confidence")
    result_df = pd.concat([parsed_df, confidence_df], axis=1)
    json_str = result_df.to_json(orient="records", lines=True)
    json_recs = json_str.split("\n")
    for rec in json_recs:
        producer.produce(args.output_topic, rec)
    producer.flush()
    return processed_data


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.cybert import Cybert

    worker = dask.distributed.get_worker()
    cy = Cybert()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with cybert model. Model File: "
        + str(args.model)
        + " Label Map: "
        + str(args.label_map)
    )
    cy.load_model(args.model, args.label_map)
    worker.data["cybert"] = cy
    print("Successfully initialized dask worker " + str(worker))


def calc_benchmark(processed_data, size_per_log):
    # Calculates benchmark for the streamz workflow
    t1 = int(round(time.time() * 1000))
    t2 = 0
    size = 0.0
    batch_count = 0
    # Find min and max time while keeping track of batch count and size
    for result in processed_data:
        (ts1, ts2, result_size) = (result[2], result[3], result[4])
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


def signal_term_handler(signal, frame):
    # Receives signal and calculates benchmark if indicated in argument
    print("Exiting streamz script...")
    if args.benchmark:
        (time_diff, throughput_mbps, avg_batch_size) = calc_benchmark(
            output, args.benchmark
        )
        print("*** BENCHMARK ***")
        print(
            "Job duration: {:.3f} secs, Throughput(mb/sec):{:.3f}, Avg. Batch size(mb):{:.3f}".format(
                time_diff, throughput_mbps, avg_batch_size
            )
        )
    sys.exit(0)


def parse_arguments():
    # Establish script arguments
    parser = argparse.ArgumentParser(
        description="Cybert using Streamz and Dask. \
                                                  Data will be read from the input kafka topic, \
                                                  processed using cybert, and output printed."
    )
    parser.add_argument("-b", "--broker", default="localhost:9092", help="Kafka broker")
    parser.add_argument(
        "-i", "--input_topic", default="input", help="Input kafka topic"
    )
    parser.add_argument(
        "-o", "--output_topic", default="output", help="Output kafka topic"
    )
    parser.add_argument("-g", "--group_id", default="streamz", help="Kafka group ID")
    parser.add_argument("-m", "--model", help="Model filepath")
    parser.add_argument("-l", "--label_map", help="Label map filepath")
    parser.add_argument(
        "--dask_scheduler",
        help="Dask scheduler address. If not provided a new dask cluster will be created",
    )
    parser.add_argument(
        "--max_batch_size",
        default=1000,
        type=int,
        help="Max batch size to read from kafka",
    )
    parser.add_argument("--poll_interval", type=str, help="Polling interval (ex: 60s)")
    parser.add_argument(
        "--benchmark",
        help="Captures benchmark, including throughput estimates, with provided avg log size in KB. (ex: 500 or 0.1)",
        type=float,
    )
    args = parser.parse_args()
    return args


def create_dask_client():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    return client


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    # Handle script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)
    # Kafka producer configuration for processed output data
    producer_conf = {
        "bootstrap.servers": args.broker,
        "session.timeout.ms": 10000,
    }
    print("Producer conf:", producer_conf)
    client = create_dask_client()
    client.run(worker_init)

    # Define the streaming pipeline.
    consumer_conf = {
        "bootstrap.servers": args.broker,
        "group.id": args.group_id,
        "session.timeout.ms": 60000,
        "enable.partition.eof": "true",
        "auto.offset.reset": "earliest",
    }
    print("Consumer conf:", consumer_conf)
    source = Stream.from_kafka_batched(
        args.input_topic,
        consumer_conf,
        poll_interval=args.poll_interval,
        npartitions=1,
        asynchronous=True,
        dask=True,
        max_batch_size=args.max_batch_size,
    )

    # If benchmark arg is True, use streamz to compute benchmark
    if args.benchmark:
        print("Benchmark will be calculated")
        output = (
            source.map(inference)
            .map(lambda x: (x[0], x[1], x[2], int(round(time.time())), x[3]))
            .map(sink_to_kafka)
            .gather()
            .sink_to_list()
        )
    else:
        output = source.map(inference).map(sink_to_kafka).gather()

    source.start()
