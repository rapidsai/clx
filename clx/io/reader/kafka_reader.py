import cudf
import logging
import time
from confluent_kafka import KafkaError
from clx.io.reader.reader import Reader

log = logging.getLogger(__name__)

class KafkaReader(Reader):
    def __init__(self, batch_size, consumer, time_window=30):
        self._batch_size = batch_size
        self._consumer = consumer
        self._has_data = True
        # Max window of time that queued events will wait to be pushed to workflow
        self._time_window = time_window

    @property
    def consumer(self):
        return self._consumer

    @property
    def has_data(self):
        return self._has_data

    @property
    def time_window(self):
        return self._time_window

    def fetch_data(self):
        events = []
        rec_cnt = 0
        running = True
        current_time = time.time()
        try:
            while running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    log.debug("No message received.")
                    continue
                elif not msg.error():
                    data = msg.value().decode("utf-8")
                    log.debug("Message received.")
                    if (
                        rec_cnt < self._batch_size
                        and (time.time() - current_time) < self.time_window
                    ):
                        events.append(data)
                        rec_cnt += 1
                    else:
                        events.append(data)
                        running = False
                elif msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error(msg.error())
                    running = False
                else:
                    running = False
            df = cudf.DataFrame()
            df["Raw"] = events
            return df
        except:
            log.error("Error fetching data from kafka")
            raise

    def close(self):
        log.info("Closing kafka reader...")
        if self.consumer is not None:
            self.consumer.close()
        log.info("Closed kafka reader.")
