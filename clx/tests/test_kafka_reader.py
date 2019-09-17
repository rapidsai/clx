import cudf
import pytest

from confluent_kafka import Consumer
from confluent_kafka import Message
from mockito import when, mock, verify
from clx.io.reader.kafka_reader import KafkaReader

batch_size = 100
consumer = mock(Consumer)
message = mock(Message)

@pytest.mark.parametrize("batch_size", [batch_size])
@pytest.mark.parametrize("consumer", [consumer])
def test_read_data(batch_size, consumer):
    reader = KafkaReader(batch_size, consumer)
    when(reader.consumer).poll(timeout=1.0).thenReturn(None).thenReturn(
        message
    ).thenRaise(Exception())
    with pytest.raises(Exception):
        reader.fetch_data()
        verify(reader.consumer, times=1).poll(...)
