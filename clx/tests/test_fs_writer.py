import glob
import csv
import os
import cudf
import pytest
import shutil
import pandas as pd

from pathlib import Path
from clx.io.factory.factory import Factory
from clx.io.writer.fs_writer import FileSystemWriter


test_output_base_path = "%s/output" % os.path.dirname(os.path.realpath(__file__))
df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)
expected_df = df.to_pandas()


@pytest.mark.parametrize("test_output_base_path", [test_output_base_path])
@pytest.mark.parametrize("df", [df])
def test_write_data_text(test_output_base_path, df):
    test_output_path = "%s/person.csv" % (test_output_base_path)
    if os.path.exists(test_output_path):
        os.remove(test_output_path)
    config = {
        "type": "fs",
        "output_path": test_output_path,
        "output_format": "text", "index": False
    }
    writer = FileSystemWriter(config)
    writer.write_data(df)

    with open(test_output_path) as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    assert data[0] == ["firstname", "lastname", "gender"]
    assert data[1] == ["Emma", "Olivia", "F"]
    assert data[2] == ["Ava", "Isabella", "F"]
    assert data[3] == ["Sophia", "Charlotte", "F"]


@pytest.mark.parametrize("test_output_base_path", [test_output_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
@pytest.mark.parametrize("df", [df])
def test_write_data_parquet(test_output_base_path, expected_df, df):
    test_output_path = "%s/person_parquet" % (test_output_base_path)
    if os.path.exists(test_output_path) and os.path.isdir(test_output_path):
        shutil.rmtree(test_output_path)
    config = {
        "type": "fs",
        "output_path": test_output_path,
        "output_format": "parquet"}
    writer = FileSystemWriter(config)
    writer.write_data(df)
    output_files = glob.glob("%s/*" % (test_output_path))
    result = pd.read_parquet(output_files[0], engine="pyarrow")
    assert result.equals(expected_df)
