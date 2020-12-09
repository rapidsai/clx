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
import gc
import sys
import time
import dask
import torch
import cudf
import signal
from streamz import Stream
from tornado import ioloop
from clx_streamz_tools import utils
from clx.dns import dns_extractor as dns


def inference(messages_df):
    # Messages will be received and run through DGA inferencing
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    result_size = messages_df.shape[0]
    print("Processing batch size: " + str(result_size))
    dd = worker.data["dga_detector"]
    preds = dd.predict(messages_df["domain"])
    messages_df["preds"] = preds
    torch.cuda.empty_cache()
    gc.collect()
    return (gdf, batch_start_time, result_size)


def sink_to_kafka(processed_data):
    # Prediction data will be published to provided Kafka producer
    utils.kafka_sink(kafka_conf["output_topic"], processed_data[0])
    return processed_data


def sink_to_es(processed_data):
    # Prediction data will be published to ElasticSearch cluster
    utils.es_sink(config["elasticsearch_conf"], processed_data[0])
    return processed_data


def sink_to_fs(processed_data):
    # Prediction data will be written to disk
    utils.fs_sink(config, processed_data[0])
    return processed_data


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.dga_detector import DGADetector

    worker = dask.distributed.get_worker()
    dd = DGADetector()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with dga model. Model File: "
        + str(args.model)
    )
    dd.load_model(args.model)
    worker = utils.init_dask_workers(worker, "dga_detector", dd, config)


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


def start_stream():
    # Define the streaming pipeline.
    source = Stream.from_kafka_batched(
        kafka_conf["input_topic"],
        kafka_conf["consumer_conf"],
        poll_interval=args.poll_interval,
        # npartitions value varies based on kafka topic partitions configuration.
        npartitions=kafka_conf["n_partitions"],
        asynchronous=True,
        dask=True,
        engine="cudf",
        max_batch_size=args.max_batch_size,
    )
    sink = config["sink"]
    global output
    # If benchmark arg is True, use streamz to compute benchmark
    if args.benchmark:
        print("Benchmark will be calculated")
        output = (
            source.map(inference)
            .map(lambda x: (x[0], x[1], int(round(time.time())), x[2]))
            .map(sink_dict[sink])
            .gather()
            .sink_to_list()
        )
    else:
        output = source.map(inference).map(sink_dict[sink]).gather()

    source.start()


if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_arguments()
    config = utils.load_yaml(args.conf)
    kafka_conf = config["kafka_conf"]
    sink_dict = {
        "kafka": sink_to_kafka,
        "elasticsearch": sink_to_es,
        "filesystem": sink_to_fs,
    }
    # create output directory if not exists when sink is set to file system
    utils.create_dir(config['sink'], config['output_dir'])

    # Handle script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)

    client = utils.create_dask_client()
    client.run(worker_init)

    print("Consumer conf: " + str(kafka_conf["consumer_conf"]))

    loop = ioloop.IOLoop.current()
    loop.add_callback(start_stream)

    try:
        loop.start()
    except KeyboardInterrupt:
        worker = dask.distributed.get_worker()
        sink = worker.data["sink"]
        if config["sink"] == utils.SINK_KAFKA:
            sink.close()
        elif config["sink"] == utils.SINK_ES:
            sink.transport.close()
        else:
            pass
        loop.stop()
        