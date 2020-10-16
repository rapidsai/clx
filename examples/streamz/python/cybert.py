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
import pandas as pd
import confluent_kafka as ck
import cudf
import dask
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
    result_size = df.shape[0]
    torch.cuda.empty_cache()
    gc.collect()
    confidence_df = confidence_df.add_suffix('_conf')
    parsed_df = pd.concat([parsed_df, confidence_df], axis=1)
    return (batch_start_time, result_size, parsed_df)


def sink_to_kafka(processed_data):
    # Parsed data and confidence scores will be published to provided kafka producer
    parsed_df = processed_data[3]
    producer = ck.Producer(producer_conf)
    json_str = parsed_df.to_json(orient='records', lines=True)
    json_recs = json_str.split('\n')
    for json_rec in json_recs:
        producer.produce(args.output_topic, json_rec)
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
        default=1,
    )
    args = parser.parse_args()
    return args


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
    client = create_dask_client(args.dask_scheduler)
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
            .map(lambda x: (x[0], int(round(time.time())), x[1], x[2]))
            .map(sink_to_kafka)
            .gather()
            .sink_to_list()
        )
    else:
        output = source.map(inference).map(sink_to_kafka).gather()

    source.start()
