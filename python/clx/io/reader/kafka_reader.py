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

import cudf
import logging
import time
from confluent_kafka import KafkaError
from clx.io.reader.reader import Reader

log = logging.getLogger(__name__)


class KafkaReader(Reader):
    """
    Reads from Kafka based on config object.

    :param batch_size: batch size
    :param consumer: Kafka consumer
    :param time_window: Max window of time that queued events will wait to be pushed to workflow
    """
    def __init__(self, batch_size, consumer, time_window=30):
        self._batch_size = batch_size
        self._consumer = consumer
        self._has_data = True
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
        """
        Fetch data from Kafka based on provided config object
        """
        events = []
        rec_cnt = 0
        running = True
        current_time = time.time()
        try:
            while running:
                # First check if batch size or time window has been exceeded
                if (
                    rec_cnt >= self._batch_size or (time.time() - current_time) >= self.time_window
                ):
                    log.debug(
                        "Exceeded record count (" + str(rec_cnt) + ") or time window (" + str(time.time() - current_time) + ")"
                    )
                    running = False
                # Else poll next message in kafka queue
                else:
                    msg = self.consumer.poll(timeout=1.0)
                    if msg is None:
                        log.debug("No message received.")
                        continue
                    elif not msg.error():
                        data = msg.value().decode("utf-8")
                        log.debug("Message received.")
                        events.append(data)
                        rec_cnt += 1
                    elif msg.error().code() != KafkaError._PARTITION_EOF:
                        log.error(msg.error())
                        running = False
                    else:
                        running = False
            df = cudf.DataFrame()
            if len(events) > 0:
                df["Raw"] = events
            log.debug("Kafka reader batch aggregation complete. Dataframe size = " + str(df.shape))
            return df
        except Exception:
            log.error("Error fetching data from kafka")
            raise

    def close(self):
        """
        Close Kafka reader
        """
        log.info("Closing kafka reader...")
        if self.consumer is not None:
            self.consumer.close()
        log.info("Closed kafka reader.")
