# Copyright (c) 2019, NVIDIA CORPORATION.
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

import logging

from confluent_kafka import Consumer
from confluent_kafka import Producer

from clx.io.factory.abstract_factory import AbstractFactory
from clx.io.reader.kafka_reader import KafkaReader
from clx.io.writer.kafka_writer import KafkaWriter

log = logging.getLogger(__name__)


class KafkaFactory(AbstractFactory):
    def __init__(self, config):
        """
        Constructor method

        :param config: dictionary object of config values for **batch_size**, **time_window**, **publisher_kafka_topic**, **output_delimiter**, **kafka_brokers**, and **group_id**.
        """
        self._config = config

    def get_reader(self):
        """
        Get instance of KafkaReader
        """
        consumer = self._create_consumer()
        if "time_window" in self.config:
            reader = KafkaReader(
                self.config["batch_size"],
                consumer,
                time_window=self.config["time_window"],
            )
        else:
            reader = KafkaReader(self.config["batch_size"], consumer)
        return reader

    def get_writer(self):
        """
        Get instance of KafkaWriter
        """
        producer = self._create_producer()
        writer = KafkaWriter(
            self.config["publisher_kafka_topic"],
            self.config["batch_size"],
            self.config["output_delimiter"],
            producer,
        )
        return writer

    def _create_consumer(self):
        log.info("creating kafka consumer instance")
        consumer_conf = {
            "bootstrap.servers": self.config["kafka_brokers"],
            "group.id": self.config["group_id"],
            "session.timeout.ms": 10000,
            "default.topic.config": {"auto.offset.reset": "largest"},
        }

        c = Consumer(consumer_conf)
        c.subscribe(
            self.config["consumer_kafka_topics"], on_assign=self.print_assignment
        )
        log.info("created kafka consumer instance")
        return c

    def _create_producer(self):
        log.info("creating kafka producer instance")
        producer_conf = {
            "bootstrap.servers": self.config["kafka_brokers"],
            "session.timeout.ms": 10000,
        }
        producer = Producer(producer_conf)
        log.info("created producer instance")
        return producer

    def print_assignment(self, consumer, partitions):
        print("Assignment:", partitions)
