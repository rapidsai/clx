import argparse
import cudf
import dask
import json
import torch
import torch.nn.functional as F
from dask_cuda import LocalCUDACluster
from distributed import Client
from pytorch_pretrained_bert import BertConfig, BertForTokenClassification, BertTokenizer
from streamz import Stream

# TODO: Takes the RAW Windows Event logs from Kafka and runs NER predictions against each message.
def predict_batch(messages):
    return messages

# TODO: Takes the prediction results and creates a cuDF from them.
def wel_parsing(predictions):
    return predictions

# TODO: Monitor for thresholds and trigger alert by letting those messages pass through here
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

def threshold_alert(event_logs):
    return event_logs

def worker_init():
    import clx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cybert using Streamz and Dask. Data will be read from the input kafka topic, processed using cybert, and output printed.")
    parser.add_argument("broker", default="localhost:9092", help="Kafka broker")
    parser.add_argument("input_topic", default="input", help="Input kafka topic")
    parser.add_argument("group_id", default="streamz", help="Kafka group ID")
    args = parser.parse_args()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    client.run(worker_init)
    # Define the streaming pipeline.
    consumer_conf = {'bootstrap.servers': args.broker,
                 'group.id': args.group_id, 'session.timeout.ms': 60000}
    source = Stream.from_kafka_batched(args.input_topic, consumer_conf, poll_interval='1s',
                                    npartitions=1, asynchronous=True, dask=False)
    inference = source.map(predict_batch)
    wel_parsing = inference.map(wel_parsing)
    alerts = wel_parsing.map(threshold_alert).sink(print)
    # Start the stream.
    source.start()