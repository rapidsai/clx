import pytest

from factory.factory import Factory
from reader.kafka_reader import KafkaReader
from writer.kafka_writer import KafkaWriter
from reader.fs_reader import FileSystemReader
from writer.fs_writer import FileSystemWriter

kafka_config = {
    "kafka_brokers": "localhost:9092",
    "group_id": "cyber-dp",
    "batch_size": 100,
    "consumer_kafka_topics": ["consumer_topic_t1", "consumer_topic_t2"],
    "publisher_kafka_topic": "publisher_topic_t1",
    "output_delimiter": ",",
}

fs_config = {
    "input_path": "test_input",
    "output_path": "test_output",
    "schema": ["_col1", "_col2", "_col3"],
    "delimiter": ",",
    "required_cols": ["_col1", "_col2", "_col3"],
    "dtype": ["str", "str", "str"],
    "input_format": "text",
    "output_format": "text",
}


@pytest.mark.parametrize("kafka_config", [kafka_config])
def test_get_io_reader_kafka(kafka_config):
    reader = Factory.get_reader("kafka", kafka_config)
    expected_cls = KafkaReader
    assert isinstance(reader, expected_cls)


@pytest.mark.parametrize("kafka_config", [kafka_config])
def test_get_io_writer_kafka(kafka_config):
    writer = Factory.get_writer("kafka", kafka_config)
    expected_cls = KafkaWriter
    assert isinstance(writer, expected_cls)


@pytest.mark.parametrize("fs_config", [fs_config])
def test_get_io_reader_fs(fs_config):
    reader = Factory.get_reader("fs", fs_config)
    expected_cls = FileSystemReader
    assert isinstance(reader, expected_cls)


@pytest.mark.parametrize("fs_config", [fs_config])
def test_get_io_writer_fs(fs_config):
    writer = Factory.get_writer("fs", fs_config)
    expected_cls = FileSystemWriter
    assert isinstance(writer, expected_cls)
