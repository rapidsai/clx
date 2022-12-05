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
import cudf

from clx.io.factory.factory import Factory
from clx.io.reader.kafka_reader import KafkaReader
from clx.io.writer.kafka_writer import KafkaWriter
from clx.io.reader.fs_reader import FileSystemReader
from clx.io.writer.fs_writer import FileSystemWriter

kafka_config = {
    "kafka_brokers": "localhost:9092",
    "group_id": "cyber-dp",
    "batch_size": 100,
    "consumer_kafka_topics": ["consumer_topic_t1", "consumer_topic_t2"],
    "publisher_kafka_topic": "publisher_topic_t1",
    "output_delimiter": ",",
}

fs_reader_config = {
    "type": "fs",
    "input_path": "test_input",
    "names": ["_col1", "_col2", "_col3"],
    "delimiter": ",",
    "usecols": ["_col1", "_col2", "_col3"],
    "dtype": ["str", "str", "str"],
    "input_format": "text",
}

fs_writer_config = {
    "type": "fs",
    "output_path": "test_output",
    "output_format": "text",
}

expected_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)


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


@pytest.mark.parametrize("fs_reader_config", [fs_reader_config])
def test_get_io_reader_fs(fs_reader_config):
    reader = Factory.get_reader("fs", fs_reader_config)
    expected_cls = FileSystemReader
    assert isinstance(reader, expected_cls)


@pytest.mark.parametrize("fs_writer_config", [fs_writer_config])
def test_get_io_writer_fs(fs_writer_config):
    writer = Factory.get_writer("fs", fs_writer_config)
    expected_cls = FileSystemWriter
    assert isinstance(writer, expected_cls)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_get_reader_csv(tmpdir, expected_df):
    fname = tmpdir.mkdir("tmp_test_factory").join("person.csv")
    expected_df.to_csv(fname, index=False)
    config = {
        "type": "fs",
        "input_path": fname,
        "names": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "usecols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
        "input_format": "csv",
    }
    reader_from_factory = Factory.get_reader("fs", config)
    fetched_df = reader_from_factory.fetch_data()

    assert fetched_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_get_reader_parquet(tmpdir, expected_df):
    fname = tmpdir.mkdir("tmp_test_factory").join("person.parquet")
    cudf.io.parquet.to_parquet(expected_df, fname)
    config = {
        "type": "fs",
        "input_path": fname,
        "columns": ["firstname", "lastname", "gender"],
        "input_format": "parquet",
    }
    reader_from_factory = Factory.get_reader("fs", config)
    fetched_df = reader_from_factory.fetch_data()

    assert fetched_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_get_reader_orc(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_reader").join("person.orc"))
    cudf.io.orc.to_orc(expected_df, fname)
    config = {
        "type": "fs",
        "input_path": fname,
        "usecols": ["firstname", "lastname", "gender"],
        "input_format": "orc",
    }
    reader_from_factory = Factory.get_reader("fs", config)
    fetched_df = reader_from_factory.fetch_data()

    assert fetched_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_get_reader_json(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_reader").join("person.json"))
    cudf.io.json.to_json(expected_df, fname, orient="records")
    config = {
        "type": "fs",
        "input_path": fname,
        "orient": "records",
        "input_format": "json",
    }
    reader_from_factory = Factory.get_reader("fs", config)
    fetched_df = reader_from_factory.fetch_data()

    assert fetched_df.equals(expected_df)
