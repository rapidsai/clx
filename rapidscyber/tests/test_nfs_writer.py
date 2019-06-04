import glob
from pathlib import Path

import cudf
import pandas
import pytest

from factory.factory import Factory
from reader.nfs_reader import NFSReader
from writer.nfs_writer import NFSWriter


test_output_base_path = str(Path("test_output").resolve())
df = cudf.DataFrame(
    [
        ("firstname", ["Emma", "Ava", "Sophia"]),
        ("lastname", ["Olivia", "Isabella", "Charlotte"]),
        ("gender", ["F", "F", "F"]),
    ]
)
expected_df = df.to_pandas()


@pytest.mark.parametrize("test_output_base_path", [test_output_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
@pytest.mark.parametrize("df", [df])
def test_write_data_text(test_output_base_path, expected_df, df):
    test_output_path = "%s/person.csv" % (test_output_base_path)
    config = {"output_path": test_output_path, "output_format": "text"}
    writer = NFSWriter(config["output_path"], config["output_format"])
    writer.write_data(df)

    config = {
        "input_path": test_output_path,
        "schema": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "required_cols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
        "input_format": "text",
    }
    reader = NFSReader(config)
    fetched_df = reader.fetch_data()

    assert fetched_df.to_pandas().equals(expected_df)


@pytest.mark.parametrize("test_output_base_path", [test_output_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
@pytest.mark.parametrize("df", [df])
def test_write_data_parquet(test_output_base_path, expected_df, df):
    test_output_path = "%s/person_parquet" % (test_output_base_path)
    config = {"output_path": test_output_path, "output_format": "parquet"}
    writer = NFSWriter(config["output_path"], config["output_format"])
    writer.write_data(df)

    output_files = glob.glob("%s/*" % (test_output_path))
    config = {
        "input_path": output_files[0],
        "schema": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "required_cols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
        "input_format": "parquet",
    }
    reader = NFSReader(config)
    fetched_df = reader.fetch_data()
    assert fetched_df.to_pandas().equals(expected_df)
