import logging

from confluent_kafka import KafkaError
from confluent_kafka import Producer

from rapidscyber.parser.parser_helper import ParserHelper


class KafkaWriter:
    def __init__(self, kafka_topic, batch_size, delimiter, producer):
        self._kafka_topic = kafka_topic
        self._batch_size = batch_size
        self._delimiter = delimiter
        self._producer = producer

    @property
    def producer(self):
        return self._producer

    @property
    def delimiter(self):
        return self._delimiter

    # publish messages to kafka topic
    def write_data(self, df):
        out_df = ParserHelper.generate_delimited_ouput_col(df, self.delimiter)
        for rec in out_df.to_records():
            self.producer.produce(self._kafka_topic, rec["delimited_ouput"])
            if len(self.producer) > self._batch_size:
                logging.debug(
                    "batch reached, calling poll... producer unsent: %s",
                    len(self.producer),
                )
