from pathlib import Path

import cudf
import pandas
import pytest
import os
from clx.io.factory.factory import Factory
from clx.io.reader.dask_fs_reader import DaskFileSystemReader

test_input_base_path = "%s/input" % os.path.dirname(os.path.realpath(__file__))

# Temporarily changing over cuDF to pandasDF because of issue with equality checks.
# Issue: https://github.com/rapidsai/cudf/issues/1750
expected_df = cudf.DataFrame(
    [
        ("firstname", ["Emma", "Ava", "Sophia"]),
        ("lastname", ["Olivia", "Isabella", "Charlotte"]),
        ("gender", ["F", "F", "F"]),
    ]
).to_pandas()


@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_text(test_input_base_path, expected_df):
    test_input_path = "%s/person.csv" % (test_input_base_path)
    config = {
        "input_path": test_input_path,
        "schema": ["firstname", "lastname", "gender"],
        "delimiter": ",",
        "required_cols": ["firstname", "lastname", "gender"],
        "dtype": ["str", "str", "str"],
        "header": 0,
        "input_format": "text",
    }
    reader = DaskFileSystemReader(config)
    fetched_df = reader.fetch_data().compute()

    # Temporarily changing over cuDF to pandasDF because of issue with equality checks.
    # Issue: https://github.com/rapidsai/cudf/issues/1750
    assert fetched_df.to_pandas().equals(expected_df)

@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_parquet(test_input_base_path, expected_df):
    test_input_path = "%s/person.parquet" % (test_input_base_path)
    config = {
        "input_path": test_input_path,
        "required_cols": ["firstname", "lastname", "gender"],
        "input_format": "parquet",
    }

    reader = DaskFileSystemReader(config)
    fetched_df = reader.fetch_data().compute()

    # Temporarily changing over cuDF to pandasDF because of issue with equality checks.
    # Issue: https://github.com/rapidsai/cudf/issues/1750
    assert fetched_df.to_pandas().equals(expected_df)

@pytest.mark.parametrize("test_input_base_path", [test_input_base_path])
@pytest.mark.parametrize("expected_df", [expected_df])
def test_fetch_data_orc(test_input_base_path, expected_df):
    test_input_path = "%s/person.orc" % (test_input_base_path)
    config = {
        "input_path": test_input_path,
        "required_cols": ["firstname", "lastname", "gender"],
        "input_format": "orc",
    }

    reader = DaskFileSystemReader(config)
    fetched_df = reader.fetch_data().compute()

    # Temporarily changing over cuDF to pandasDF because of issue with equality checks.
    # Issue: https://github.com/rapidsai/cudf/issues/1750
    assert fetched_df.to_pandas().equals(expected_df)