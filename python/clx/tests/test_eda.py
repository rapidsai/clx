# Copyright (c) 2020, NVIDIA CORPORATION.
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

from clx.eda.eda import EDA


def test_eda_summary_stats():
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df["b"] = ["a", "b", "c"]
    expected_output = {
        "SummaryStatistics": {
            "a": {"dtype": "int64", "summary": {"unique": "3", "total": "3"}},
            "b": {"dtype": "object", "summary": {"unique": "3", "total": "3"}},
        }
    }
    eda = EDA(df)
    actual_output = eda.analysis
    assert expected_output == actual_output
