import pytest
import cudf
from mock import patch
from confluent_kafka import Producer, KafkaError
from rapidscyber.writer.kafka_writer import KafkaWriter
from rapidscyber.parsers.parser_helper import ParserHelper

producer_conf = {"bootstrap.servers": "localhost:8191", "session.timeout.ms": 10000}


@patch(
    "rapidscyber.parsers.parser_helper.ParserHelper.generate_delimited_ouput_col",
    return_value=cudf.DataFrame(
        [
            ("firstname", ["Emma", "Ava", "Sophia"]),
            ("lastname", ["Olivia", "Isabella", "Charlotte"]),
            ("gender", ["F", "F", "F"]),
            (
                "delimited_ouput",
                ["Emma,Olivia,F", "Ava,Isabella,F", "Sophia,Charlotee,F"],
            ),
        ]
    ),
)
@pytest.mark.parametrize("producer_conf", [producer_conf])
def test_write_data(mock_generate_delimited_ouput_col, producer_conf):
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
