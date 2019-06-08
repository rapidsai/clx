import pytest
import cudf
from mock import patch
from confluent_kafka import Producer, KafkaError
from writer.kafka_writer import KafkaWriter

producer_conf = {"bootstrap.servers": "localhost:8191", "session.timeout.ms": 10000}


@pytest.mark.parametrize("producer_conf", [producer_conf])
def test_write_data(producer_conf):
    input_df = cudf.DataFrame(
        [
            ("firstname", ["Emma", "Ava", "Sophia"]),
            ("lastname", ["Olivia", "Isabella", "Charlotte"]),
            ("gender", ["F", "F", "F"]),
        ]
    )
    producer = Producer(producer_conf)
    kafka_writer_obj = KafkaWriter("rapidscyber", 1000, ",", producer)
    kafka_writer_obj.write_data(input_df)
