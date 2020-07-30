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
import cudf
import dask
from dask_cuda import LocalCUDACluster
from distributed import Client
from streamz import Stream
import confluent_kafka as ck
from clx.analytics.cybert import Cybert
import socket
import signal
import random
import time
import sys


def inference(messages):
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    size = 0
    if type(messages) == str:
        df = cudf.DataFrame()
        df["stream"] = [messages.decode("utf-8")]
        parsed_df, confidence_df = worker.data["cybert"].inference(df["stream"])
        size = len(df)
    elif type(messages) == list and len(messages) > 0:
        df = cudf.DataFrame()
        df["stream"] = [msg.decode("utf-8") for msg in messages]
        parsed_df, confidence_df = worker.data["cybert"].inference(df["stream"])
        size = len(df)
    else:
        print("ERROR: Unknown type encountered in inference")
    return (parsed_df, confidence_df, batch_start_time, size)


def sink_to_kafka(processed_data):
    parsed_df = processed_data[0]
    confidence_df = processed_data[1]
    producer = ck.Producer(producer_confs)
    producer.produce(args.output_topic, parsed_df.to_json())
    producer.produce(args.output_topic, confidence_df.to_json())
    producer.flush()
    return processed_data


def worker_init():
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


def calc_benchmark(processed_data):
    t1 = int(round(time.time() * 1000))
    t2 = 0
    size = 0.0
    batch_count = 0
    # Find min and max time while keeping track of batch count and size
    for result in processed_data:
        (ts1, ts2, sz) = (result[2], result[3], result[4])
        if ts1 == 0 or ts2 == 0:
            continue
        batch_count = batch_count + 1
        t1 = min(t1, ts1)
        t2 = max(t2, ts2)
        size += sz / 2
    time_diff = t2 - t1
    return time_diff


def signal_term_handler(signal, frame):
    print("Exiting streamz script...")
    if args.benchmark:
        time_diff = calc_benchmark(output)
        print("Job duration: {:.2f} secs".format(time_diff))
    sys.exit(0)


if __name__ == "__main__":
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
        "--cuda_visible_devices",
        nargs="+",
        type=int,
        help="Cuda visible devices (ex: '0 1 2')",
    )
    parser.add_argument(
        "--max_batch_size",
        default=1000,
        type=int,
        help="Max batch size to read from kafka",
    )
    parser.add_argument("--benchmark", help="Capture benchmark", action="store_true")
    args = parser.parse_args()

    if args.dask_scheduler is not None:
        print("Dask scheduler:", args.dask_scheduler)
        client = Client(args.dask_scheduler)
    else:
        cuda_visible_devices = args.cuda_visible_devices
        n_workers = len(cuda_visible_devices)
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=cuda_visible_devices, n_workers=n_workers
        )
        client = Client(cluster)

    print(client)
    print("Initializing Cybert instances on each Dask worker")
    client.run(worker_init)

    # Define the streaming pipeline.
    j = random.randint(0, 100000)
    consumer_conf = {
        "bootstrap.servers": args.broker,
        "group.id": args.group_id + "-%i" % j,
        "session.timeout.ms": 60000,
        "auto.offset.reset": "earliest",
    }
    print("Consumer conf:", consumer_conf)
    source = Stream.from_kafka_batched(
        args.input_topic,
        consumer_conf,
        poll_interval="1s",
        npartitions=1,
        asynchronous=True,
        dask=True,
        max_batch_size=args.max_batch_size,
    )
    producer_confs = {
        "bootstrap.servers": args.broker,
        # "client.id": socket.gethostname(),
        "session.timeout.ms": 10000,
    }
    # Handle docker container and script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)

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
