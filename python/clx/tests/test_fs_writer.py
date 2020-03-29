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

from clx.io.writer.fs_writer import FileSystemWriter


expected_df = cudf.DataFrame(
    {
        "firstname": ["Emma", "Ava", "Sophia"],
        "lastname": ["Olivia", "Isabella", "Charlotte"],
        "gender": ["F", "F", "F"],
    }
)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_write_data_csv(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_writer").join("person.csv"))
    config = {
        "type": "fs",
        "output_path": fname,
        "output_format": "csv",
        "index": False
    }
    writer = FileSystemWriter(config)
    writer.write_data(expected_df)

    result_df = cudf.read_csv(fname)
    assert result_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_write_data_parquet(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_writer").join("person.parquet"))
    config = {
        "type": "fs",
        "output_path": fname,
        "output_format": "parquet"
    }
    writer = FileSystemWriter(config)
    writer.write_data(expected_df)

    result_df = cudf.read_parquet(fname)
    assert result_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_write_data_orc(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_writer").join("person.orc"))
    config = {
        "type": "fs",
        "output_path": fname,
        "output_format": "orc",
    }
    writer = FileSystemWriter(config)
    writer.write_data(expected_df)

    result_df = cudf.read_orc(fname)
    assert result_df.equals(expected_df)


@pytest.mark.parametrize("expected_df", [expected_df])
def test_write_data_json(tmpdir, expected_df):
    fname = str(tmpdir.mkdir("tmp_test_fs_writer").join("person.json"))
    config = {
        "type": "fs",
        "output_path": fname,
        "output_format": "json",
        "orient": "records"
    }
    writer = FileSystemWriter(config)
    writer.write_data(expected_df)

    result_df = cudf.read_json(fname, orient="records")
    assert result_df.equals(expected_df)
