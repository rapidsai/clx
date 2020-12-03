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
import signal
import sys
import time
import cudf
import dask
import torch
import pandas as pd
from streamz import Stream
from tornado import ioloop
from clx_streamz_tools import utils


def inference(messages):
    # Messages will be received and run through cyBERT inferencing
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    df = cudf.DataFrame()
    if type(messages) == str:
        df["stream"] = [messages.decode("utf-8")]
    elif type(messages) == list and len(messages) > 0:
        df["stream"] = [msg.decode("utf-8") for msg in messages]
    else:
        print("ERROR: Unknown type encountered in inference")
    
    result_size = df.shape[0]
    print("Processing batch size: " + str(result_size))
    pred, prob = worker.data["phish_detect"].predict(df["stream"])
    results_gdf = cudf.DataFrame({"pred": pred, "prob": prob})
    torch.cuda.empty_cache()
    gc.collect()
    return (results_gdf, batch_start_time, result_size)


def sink_to_kafka(processed_data):
    # Parsed data and confidence scores will be published to provided kafka producer
    parsed_df = processed_data[0]
    utils.kafka_sink(producer_conf, args.output_topic, parsed_df)
    return processed_data


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


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.phishing_detector import PhishingDetector

    worker = dask.distributed.get_worker()
    phish_detect = PhishingDetector()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with phishing detection model. Model File: "
        + str(args.model)
    )
    phish_detect.init_model(args.model)
    worker.data["phish_detect"] = phish_detect
    print("Successfully initialized dask worker " + str(worker))


def start_stream():
    source = Stream.from_kafka_batched(
        args.input_topic,
        consumer_conf,
        poll_interval=args.poll_interval,
        # npartitions value varies based on kafka topic partitions configuration.
        npartitions=1,
        asynchronous=True,
        dask=True,
        max_batch_size=args.max_batch_size,
    )
    global output
    # If benchmark arg is True, use streamz to compute benchmark
    if args.benchmark:
        print("Benchmark will be calculated")
        output = (
            source.map(inference)
            .map(lambda x: (x[0], x[1], int(round(time.time())), x[2]))
            .map(sink_to_kafka)
            .gather()
            .sink_to_list()
        )
    else:
        output = source.map(inference).map(sink_to_kafka).gather()

    source.start()
    
if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_arguments()

    # Handle script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)

    client = utils.create_dask_client()
    client.run(worker_init)

    producer_conf = {"bootstrap.servers": args.broker, "session.timeout.ms": "10000"}
    consumer_conf = {
        "bootstrap.servers": args.broker,
        "group.id": args.group_id,
        "session.timeout.ms": "60000",
        "enable.partition.eof": "true",
        "auto.offset.reset": "earliest",
    }

    print("Producer conf: " + str(producer_conf))
    print("Consumer conf: " + str(consumer_conf))
    
    loop = ioloop.IOLoop.current()
    loop.add_callback(start_stream)
    
    try:
        loop.start()
    except KeyboardInterrupt:
        loop.stop()
