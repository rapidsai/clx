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

import pytest

from confluent_kafka import Consumer
from confluent_kafka import Message, KafkaError
from mockito import when, mock, verify
from clx.io.reader.kafka_reader import KafkaReader

batch_size = 100
message = mock(Message)
kafka_error = mock(KafkaError)
when(kafka_error).code().thenReturn("test")
when(message).value().thenReturn("test message".encode("utf-8"))


@pytest.mark.parametrize("batch_size", [batch_size])
def test_read_data(batch_size):
    consumer = mock(Consumer)
    reader = KafkaReader(batch_size, consumer)
    # Return msg = None 1 time, then return a valid message moving forward
    when(reader.consumer).poll(timeout=1.0).thenReturn(None).thenReturn(message)
    # Always return no message error
    when(message).error().thenReturn(None)
    df = reader.fetch_data()
    assert df.shape == (100, 1)
    assert df.columns == ["Raw"]
    assert df["Raw"][0] == "test message"
    # Call to poll returned 100(Valid messages) + 1(None message) = 101
    verify(reader.consumer, times=101).poll(...)


@pytest.mark.parametrize("batch_size", [batch_size])
def test_read_data_message_error(batch_size):
    consumer = mock(Consumer)
    reader = KafkaReader(batch_size, consumer)
    # Return valid message data
    when(reader.consumer).poll(timeout=1.0).thenReturn(message)
    # Return no message error 1 time, then an error moving forward
    when(message).error().thenReturn(None).thenReturn(kafka_error)
    df = reader.fetch_data()

    # Validate consumer polls
    # 1 (Valid message) + 1 (Error Message) = 2 Consumer polls
    verify(reader.consumer, times=2).poll(...)

    # Validate dataframe output
    assert df.shape == (1, 1)
    assert df.columns == ["Raw"]
    assert df["Raw"].to_arrow().to_pylist() == ["test message"]


@pytest.mark.parametrize("batch_size", [5])
def test_read_data_no_messages(batch_size):
    consumer = mock(Consumer)
    reader = KafkaReader(batch_size, consumer, time_window=5)
    # Return no messages
    when(reader.consumer).poll(timeout=1.0).thenReturn(None)
    df = reader.fetch_data()

    # Validate dataframe output
    assert df.empty
