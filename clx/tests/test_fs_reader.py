import os
import cudf
import pytest
from pathlib import Path
from clx.io.factory.factory import Factory
from clx.io.reader.fs_reader import FileSystemReader

test_input_base_path = "%s/input" % os.path.dirname(os.path.realpath(__file__))

expected_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)


@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_text(test_input_base_path, expected_df):
    test_input_path = "%s/person.csv" % (test_input_base_path)
    config = {
        "type": "fs",
        "input_path": test_input_path,
        "names": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "usecols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
        "input_format": "text"
    }
    reader = FileSystemReader(config)
    fetched_df = reader.fetch_data()

    assert fetched_df.equals(expected_df)


@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_parquet(test_input_base_path, expected_df):
    test_input_path = "%s/person.parquet" % (test_input_base_path)
    config = {
        "type": "fs",
        "input_path": test_input_path,
        "columns": ["firstname", "lastname", "gender"],
        "input_format": "parquet"
    }

    reader = FileSystemReader(config)
    fetched_df = reader.fetch_data()

    assert fetched_df.equals(expected_df)


@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_orc(test_input_base_path, expected_df):
    test_input_path = "%s/person.orc" % (test_input_base_path)
    config = {
        "type": "fs",
        "input_path": test_input_path,
        "input_format": "orc"
    }

    reader = FileSystemReader(config)
    fetched_df = reader.fetch_data()

    assert fetched_df.equals(expected_df)
