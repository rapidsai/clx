import logging
import time

# import sys
from confluent_kafka import KafkaError

# KafkaReader class
class KafkaReader:
    def __init__(self, batch_size, consumer):
        self._batch_size = batch_size
        self._consumer = consumer
        self._has_data = True

    @property
    def consumer(self):
        return self._consumer

    @property
    def has_data(self):
        return self._has_data

    def fetch_data(self):
        events = []
        rec_cnt = 0
        running = True
        current_time = time.time()
        time_window = 30
        try:
            while running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                elif not msg.error():
                    data = msg.value().decode("utf-8")
                    if (
                            rec_cnt < self._batch_size
                            and (time.time() - current_time) < time_window
                    ):
                        events.append(data)
                        rec_cnt += 1
                    else:
                        events.append(data)
                        running = False
                elif msg.error().code() != KafkaError._PARTITION_EOF:
                    logging.error(msg.error())
                    running = False
                else:
                    running = False
            df = cudf.dataframe.DataFrame()
            df["Raw"] = events
            return df
        except KeyboardInterrupt:
            self.close_consumer()
            # sys.stderr.write("%% Aborted by user\n")
            logging.warning("%% Aborted by user\n")
        except:
            logging.error("Error fetching data from kafka topic.")
            raise

    def close_consumer(self):
        logging.info("Closing kafka consumer...")
        if self.consumer is not None:
            self.consumer.close()
            logging.info("Kafka consumer closed.")
        else:
            logging.warning("No kafka consumer is defined.")