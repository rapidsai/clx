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

import sys
import time
import dask
import signal
from clx_streamz_tools import utils
from streamz import Stream
from tornado import ioloop
from abc import ABC, abstractmethod


class StreamzWorkflow(ABC):
    def __init__(self):
        self.args = utils.parse_arguments()
        self.config = utils.load_yaml(self.args.conf)
        self.kafka_conf = self.config["kafka_conf"]
        self.sink_dict = {
            "kafka": self.sink_to_kafka,
            "elasticsearch": self.sink_to_es,
            "filesystem": self.sink_to_fs,
        }

    def sink_to_kafka(self, processed_data):
        """
        Writes the result to kafka topic.
        :param processed_data: Parsed/Processed data.
        :type processed_data: cudf
        :return: processed_data
        :rtype: cudf
        """
        utils.kafka_sink(self.kafka_conf["output_topic"], processed_data[0])
        return processed_data

    def sink_to_es(self, processed_data):
        """
        Writes the result to Elasticsearch.
        :param processed_data: Parsed/Processed data.
        :type processed_data: cudf
        :return: processed_data
        :rtype: cudf
        """
        utils.es_sink(self.config["elasticsearch_conf"], processed_data[0])
        return processed_data

    def sink_to_fs(self, processed_data):
        """
        Writes the result to Filesystem.
        :param processed_data: Parsed/Processed data.
        :type processed_data: cudf
        :return: processed_data
        :rtype: cudf
        """
        utils.fs_sink(self.config, processed_data[0])
        return processed_data

    def signal_term_handler(self, signal, frame):
        """
        Receives signal and calculates benchmark if indicated in argument.
        """
        print("Exiting streamz script...")
        if self.args.benchmark:
            (time_diff, throughput_mbps, avg_batch_size) = utils.calc_benchmark(
                output, self.args.benchmark
            )
            print("*** BENCHMARK ***")
            print(
                "Job duration: {:.3f} secs, Throughput(mb/sec):{:.3f}, Avg. Batch size(mb):{:.3f}".format(
                    time_diff, throughput_mbps, avg_batch_size
                )
            )
        sys.exit(0)

    def _start_stream(self):
        # Define the streaming pipeline.
        if self.config["cudf_engine"]:
            source = Stream.from_kafka_batched(
                self.kafka_conf["input_topic"],
                self.kafka_conf["consumer_conf"],
                poll_interval=self.args.poll_interval,
                # npartitions value varies based on kafka topic partitions configuration.
                npartitions=self.kafka_conf["n_partitions"],
                asynchronous=True,
                dask=True,
                engine="cudf",
                max_batch_size=self.args.max_batch_size,
            )
        else:
            source = Stream.from_kafka_batched(
                self.kafka_conf["input_topic"],
                self.kafka_conf["consumer_conf"],
                poll_interval=self.args.poll_interval,
                # npartitions value varies based on kafka topic partitions configuration.
                npartitions=self.kafka_conf["n_partitions"],
                asynchronous=True,
                dask=True,
                max_batch_size=self.args.max_batch_size,
            )

        sink = self.config["sink"]
        global output
        # If benchmark arg is True, use streamz to compute benchmark
        if self.args.benchmark:
            print("Benchmark will be calculated")
            output = (
                source.map(self.inference)
                .map(lambda x: (x[0], x[1], int(round(time.time())), x[2]))
                .map(self.sink_dict[sink])
                .gather()
                .sink_to_list()
            )
        else:
            output = source.map(self.inference).map(self.sink_dict[sink]).gather()

        source.start()

    def start(self):
        """
        Configure the workflow settings and starts streaming messages
        """
        # create output directory if not exists when sink is set to file system
        utils.create_dir(self.config["sink"], self.config["output_dir"])

        # Handle script exit
        signal.signal(signal.SIGTERM, self.signal_term_handler)
        signal.signal(signal.SIGINT, self.signal_term_handler)

        client = utils.create_dask_client()
        client.run(self.worker_init)

        print("Consumer conf: " + str(self.kafka_conf["consumer_conf"]))

        loop = ioloop.IOLoop.current()
        loop.add_callback(self._start_stream)

        try:
            loop.start()
        except KeyboardInterrupt:
            worker = dask.distributed.get_worker()
            sink = worker.data["sink"]
            if self.config["sink"] == utils.SINK_KAFKA:
                sink.close()
            elif self.config["sink"] == utils.SINK_ES:
                sink.transport.close()
            else:
                pass
            loop.stop()

    @abstractmethod
    def inference(self, message_df):
        pass

    @abstractmethod
    def worker_init(self):
        pass
