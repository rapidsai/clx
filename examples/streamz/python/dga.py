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

import gc
import sys
import time
import cudf
import dask
import torch
import signal
import random
import argparse
from streamz import Stream
import confluent_kafka as ck
from clx_streamz_tools import utils

def inference(messages_df):
    # Messages will be received and run through DGA inferencing
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    dd = worker.data["dga_detector"]
    preds = dd.predict(messages_df['domain'])
    messages_df['preds'] = preds
    result_size = messages_df.shape[0]
    print('dataframe size: %s' %(result_size))
    torch.cuda.empty_cache()
    gc.collect()
    return (batch_start_time, result_size, messages_df,)

def sink_to_kafka(processed_data):
    # Prediction data will be published to provided kafka producer
    messages_df = processed_data[3]
    utils.kafka_sink(producer_conf, args.output_topic, messages_df)
    return processed_data


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.dga_detector import DGADetector
    #from commons import utils
    #import imp
    #utils = imp.load_source('utils', 'commons/utils.py')
    worker = dask.distributed.get_worker()
    dd = DGADetector()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with dga model. Model File: "
        + str(args.model)
    )
    dd.load_model(args.model)
    worker.data["dga_detector"] = dd
    print("Successfully initialized dask worker " + str(worker))


def signal_term_handler(signal, frame):
    # Receives signal and calculates benchmark if indicated in argument
    print("Exiting streamz script...")
    if args.benchmark:
        (time_diff, throughput_mbps, avg_batch_size) = utils.calc_benchmark(
            output, args.benchmark
        )
        print("*** BENCHMARK ***")
        print(
            "Job duration: {:.3f} secs, Throughput(mb/sec):{:.3f}, Avg. Batch size(mb):{:.3f}".format(
                time_diff, throughput_mbps, avg_batch_size
            )
        )
    sys.exit(0)
   
if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_arguments()
    
    # Handle script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)
    
    client = utils.create_dask_client(args.dask_scheduler)
    client.run(worker_init)
    
    producer_conf = {
        "bootstrap.servers": args.broker,
        "session.timeout.ms": "10000",
        #"queue.buffering.max.messages": "250000",
        #"linger.ms": "100"
    }
    consumer_conf = {
        "bootstrap.servers": args.broker,
        "group.id": args.group_id,
        "session.timeout.ms": "60000",
        "enable.partition.eof": "true",
        "auto.offset.reset": "earliest",
    }
    print("Consumer conf:", consumer_conf)
    print("Producer conf:", producer_conf)
    
    # Define the streaming pipeline.
    source = Stream.from_kafka_batched(
        args.input_topic,
        consumer_conf,
        poll_interval=args.poll_interval,
        npartitions=1,
        asynchronous=True,
        dask=True,
        engine="cudf",
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
    