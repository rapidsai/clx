import pytest

from rapidscyber.factory.factory import Factory
from rapidscyber.reader.kafka_reader import KafkaReader
from rapidscyber.reader.nfs_reader import NFSReader
from rapidscyber.writer.kafka_writer import KafkaWriter
from rapidscyber.writer.nfs_writer import NFSWriter

kafka_config = {
    "kafka_brokers": "localhost:8191",
    "group_id": "cyber-dp",
    "batch_size": 100,
    "consumer_kafka_topics": ["localhost:9092", "localhost:9092"],
    "publisher_kafka_topic": "localhost:9092",
    "output_delimiter": ",",
}

nfs_config = {
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
def test_getIoReaderKafka(kafka_config):
    reader = Factory.getIOReader("kafka", kafka_config)
    expected_cls = KafkaReader
    assert isinstance(reader, expected_cls)


@pytest.mark.parametrize("kafka_config", [kafka_config])
def test_getIoWriterKafka(kafka_config):
    writer = Factory.getIOWriter("kafka", kafka_config)
    expected_cls = KafkaWriter
    assert isinstance(writer, expected_cls)


@pytest.mark.parametrize("nfs_config", [nfs_config])
def test_getIoReaderNFS(nfs_config):
    reader = Factory.getIOReader("nfs", nfs_config)
    expected_cls = NFSReader
    assert isinstance(reader, expected_cls)


@pytest.mark.parametrize("nfs_config", [nfs_config])
def test_getIoWriterNFS(nfs_config):
    writer = Factory.getIOWriter("nfs", nfs_config)
    expected_cls = NFSWriter
    assert isinstance(writer, expected_cls)
