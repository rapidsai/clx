import logging

from confluent_kafka import Consumer
from confluent_kafka import KafkaError
from confluent_kafka import Producer

from factory.abstract_factory import AbstractFactory
from reader.kafka_reader import KafkaReader
from writer.kafka_writer import KafkaWriter


class KafkaFactory(AbstractFactory):
    def __init__(self, config):
        self._config = config

    def getReader(self):
        consumer = self._createConsumer()
        reader = KafkaReader(self.config["batch_size"], consumer)
        return reader

    def getWriter(self):
        producer = self._createProducer()
        writer = KafkaWriter(
            self.config["publisher_kafka_topic"],
            self.config["batch_size"],
            self.config["output_delimiter"],
            producer,
        )
        return writer

    def _createConsumer(self):
        logging.info("creating kafka consumer instance")
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
        logging.info("created kafka consumer instance")
        return c

    def _createProducer(self):
        logging.info("creating kafka producer instance")
        producer_conf = {
            "bootstrap.servers": self.config["kafka_brokers"],
            "session.timeout.ms": 10000,
        }
        producer = Producer(producer_conf)
        logging.info("created producer instance")
        return producer

    def print_assignment(self, consumer, partitions):
        print("Assignment:", partitions)
