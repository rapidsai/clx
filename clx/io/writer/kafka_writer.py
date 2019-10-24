import logging

log = logging.getLogger(__name__)

class KafkaWriter:

    #Column name of formatted output messages sent to kafka
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

    # publish messages to kafka topic
    def write_data(self, df):
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
        gdf[first_col] = gdf[first_col].astype('str').str.fillna("")
        gdf[self.output_colname] = gdf[first_col].astype('str').str.rstrip()
        for col in gdf.columns[1:-1]:
            gdf[col] = gdf[col].astype('str').fillna("")
            gdf[col] = gdf[col].astype('str').str.rstrip()
            gdf[self.output_colname] = gdf[self.output_colname].str.cat(
                gdf[col], sep=self.delimiter
            )
        return gdf

    def close(self):
        log.info("Closing kafka writer...")
        if self._producer is not None:
            self._producer.flush()
        log.info("Closed kafka writer.")