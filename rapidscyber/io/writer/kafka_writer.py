import logging

from confluent_kafka import KafkaError
from confluent_kafka import Producer


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
        out_df = self._generate_delimited_ouput_col(df)
        for rec in out_df.to_records():
            self.producer.produce(self._kafka_topic, rec["delimited_ouput"])
            if len(self.producer) > self._batch_size:
                logging.debug(
                    "batch reached, calling poll... producer unsent: %s",
                    len(self.producer),
                )

    def _generate_delimited_ouput_col(self, gdf):
        first_col = gdf.columns[0]
        gdf[first_col] = gdf[first_col].data.fillna("")
        gdf["delimited_ouput"] = gdf[first_col].str.rstrip()
        for col in gdf.columns[1:-1]:
            gdf[col] = gdf[col].data.fillna("")
            gdf[col] = gdf[col].str.rstrip()
            gdf["delimited_ouput"] = gdf.delimited_ouput.str.cat(
                gdf[col], sep=self.delimiter
            )
        return gdf
