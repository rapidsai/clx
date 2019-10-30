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

import cudf
import pytest
import os
from pathlib import Path
from clx.io.factory.factory import Factory
from clx.io.reader.dask_fs_reader import DaskFileSystemReader

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

    assert fetched_df.equals(expected_df)


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

    assert fetched_df.equals(expected_df)


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

    assert fetched_df.equals(expected_df)
