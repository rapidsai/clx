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

log = logging.getLogger(__name__)


class KafkaWriter:
    """
    Publish to Kafka topic based on config object.

    :param kafka_topic: Kafka topic
    :param batch_size: batch size
    :param delimiter: delimiter
    :param producer: producer
    """

    # Column name of formatted output messages sent to kafka
    output_colname = "delimited_output"

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

    def write_data(self, df):
        """
        publish messages to kafka topic

        :param df: dataframe to publish
        """
        out_df = self._generate_delimited_ouput_col(df)
        for rec in out_df.to_records():
            self.producer.produce(self._kafka_topic, rec[self.output_colname])
            if len(self.producer) > self._batch_size:
                log.debug(
                    "batch reached, calling poll... producer unsent: %s",
                    len(self.producer),
                )
                self.producer.poll(0)

    def _generate_delimited_ouput_col(self, gdf):
        first_col = gdf.columns[0]
        gdf[first_col] = gdf[first_col].astype("str").fillna("")
        gdf[self.output_colname] = gdf[first_col].astype("str").str.rstrip()
        for col in gdf.columns[1:-1]:
            gdf[col] = gdf[col].astype("str").fillna("")
            gdf[col] = gdf[col].astype("str").str.rstrip()
            gdf[self.output_colname] = gdf[self.output_colname].str.cat(
                gdf[col], sep=self.delimiter
            )
        return gdf

    def close(self):
        """
        Close Kafka writer
        """
        log.info("Closing kafka writer...")
        if self._producer is not None:
            self._producer.flush()
        log.info("Closed kafka writer.")
