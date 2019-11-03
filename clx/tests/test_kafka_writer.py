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
import pytest
from mockito import when, mock, verify
from clx.io.writer.kafka_writer import KafkaWriter

input_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)
kafka_topic = "publisher_topic_t1"
batch_size = 100
delimiter = ","
producer = mock()


@pytest.mark.parametrize("kafka_topic", [kafka_topic])
@pytest.mark.parametrize("batch_size", [batch_size])
@pytest.mark.parametrize("delimiter", [delimiter])
@pytest.mark.parametrize("producer", [producer])
@pytest.mark.parametrize("input_df", [input_df])
def test_write_data(kafka_topic, batch_size, delimiter, producer, input_df):
    writer = KafkaWriter(kafka_topic, batch_size, delimiter, producer)
    when(writer.producer).__len__().thenReturn(1)
    writer.write_data(input_df)
    verify(writer.producer, times=3).produce(...)
