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
import socket

def inference(messages):
    if type(messages) == str:
        df = cudf.DataFrame()
        df['stream'] = [messages]
        print(df)
        output_df = cy.inference(df)
    elif type(messages) == list:
        df = cudf.DataFrame()
        df['stream'] = messages
        output_df = cy.inference(df)
    else:
        print("ERROR: Unknown type encountered in inference")
    return output_df.to_json()

def sink_to_kafka(event_logs):
    conf = {"bootstrap.servers": args.broker,
            "client.id": socket.gethostname(),
            "session.timeout.ms": 10000}
    producer = Producer(conf)
    for event in event_logs:
        producer.produce(args.output_topic, event)
    producer.poll(1)

def worker_init():
    from clx.analytics.cybert import Cybert
    worker = dask.distributed.get_worker()
    cybert = Cybert()
    worker.data['cybert'] = cybert

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cybert using Streamz and Dask. Data will be read from the input kafka topic, processed using cybert, and output printed.")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker")
    parser.add_argument("--input_topic", default="input", help="Input kafka topic")
    parser.add_argument("--output_topic", default="output", help="Output kafka topic")
    parser.add_argument("--group_id", default="streamz", help="Kafka group ID")
    parser.add_argument("--model", help="Model filepath")
    parser.add_argument("--label_map", help="Label map filepath")
    args = parser.parse_args()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    client.run(worker_init)
    cy = Cybert()
    cy.load_model(args.model, args.label_map, 21)
    # Define the streaming pipeline.
    consumer_conf = {'bootstrap.servers': args.broker,
                     'group.id': args.group_id, 'session.timeout.ms': 60000}
    source = Stream.from_kafka_batched(args.input_topic, consumer_conf, poll_interval='1s',
                                    npartitions=1, asynchronous=True, dask=False)
    inference = source.map(inference).map(sink_to_kafka)
    # Start the stream.
    source.start()