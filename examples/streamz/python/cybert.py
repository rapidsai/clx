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
from confluent_kafka import Producer
from clx.analytics.cybert import Cybert
import socket


def inference(messages):
    print("Entering inference ... # Messages: " + str(len(messages)))
    output_df = None
    worker = dask.distributed.get_worker()
    print("Dask Inference worker: " + str(worker))

    if type(messages) == str:
        df = cudf.DataFrame()
        df["stream"] = [messages.decode("utf-8")]
        output_df = worker["cybert"].inference(df)
    elif type(messages) == list and len(messages) > 0:
        df = cudf.DataFrame()
        df["stream"] = [msg.decode("utf-8") for msg in messages]

        output_df = worker["cybert"].inference(df)
    else:
        print("ERROR: Unknown type encountered in inference")
    return [output_df.to_json()]


def sink_to_kafka(event_logs):
    producer_confs = {
        "bootstrap.servers": args.broker,
        "client.id": socket.gethostname(),
        "session.timeout.ms": 10000,
    }
    producer = Producer(producer_confs)
    for event in event_logs:
        producer.produce(args.output_topic, event)
    producer.poll(1)


def worker_init():
    from clx.analytics.cybert import Cybert

    worker = dask.distributed.get_worker()
    print("Worker in worker_init() " + str(worker))

    # Create Cybert instance
    cy = Cybert()
    print("After 'cy' creation")
    print("Model File: " + str(args.model) + " Label Map: " + str(args.label_map))
    cy.load_model(args.model, args.label_map)
    print("after load_model call")
    worker.data["cybert"] = cy

    print("Cybert module created and loaded ...")


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
        "-d",
        "--dask_scheduler",
        help="Dask scheduler address. If not provided a new dask cluster will be created",
    )
    parser.add_argument("-c", "--cuda_visible_devices", nargs="+", type=int, help="")
    args = parser.parse_args()

    if "dask_scheduler" in args:
        client = Client(args["dask_scheduler"])
    else:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=args["cuda_visible_devices"],
            n_workers=len(args["cuda_visible_devices"]),
        )
        client = Client(cluster)

    print(client)
    print("Initializing Cybert instances on each Dask worker")
    client.run(worker_init)

    # Define the streaming pipeline.
    consumer_conf = {
        "bootstrap.servers": args.broker,
        "group.id": args.group_id,
        "session.timeout.ms": 60000,
    }
    print("Consumer conf:", consumer_conf)
    source = Stream.from_kafka_batched(
        args.input_topic,
        consumer_conf,
        poll_interval="1s",
        npartitions=1,
        asynchronous=True,
        dask=True,
    )

    inference = source.map(inference).gather().map(sink_to_kafka)
    source.start()
