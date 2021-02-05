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

import json

import cudf
import cuxfilter
import pandas as pd
import pytest

from clx.eda import EDA


@pytest.fixture
def test_dataframe():
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3, 4]
    df["b"] = ["a", "b", "c", "c"]
    df["c"] = [True, False, True, True]
    df["d"] = cudf.Series(pd.date_range("2000-01-01", periods=3, freq="m"))
    return df


def test_eda_summary_stats(test_dataframe):
    """Test EDA Summary statistics"""
    expected_output = {
        "SummaryStatistics": {
            "a": {"dtype": "int64", "summary": {"unique": "4", "total": "4"}},
            "b": {"dtype": "object", "summary": {"unique": "3", "total": "4"}},
            "c": {"dtype": "bool", "summary": {"true_percent": "0.75"}},
            "d": {
                "dtype": "datetime64[ns]",
                "summary": {"timespan": "60 days, 2880 hours, 0 minutes, 0 seconds"},
            },
        }
    }
    eda = EDA(test_dataframe)
    actual_output = eda.analysis
    assert expected_output == actual_output


def test_eda_save_analysis(tmpdir, test_dataframe):
    """Test saving the analysis to a json file"""
    fdir = str(tmpdir.mkdir("tmp_test_eda"))
    fname = fdir + "/SummaryStatistics.json"
    eda = EDA(test_dataframe)
    eda.save_analysis(fdir)
    expected_output = {
        "a": {"dtype": "int64", "summary": {"unique": "4", "total": "4"}},
        "b": {"dtype": "object", "summary": {"unique": "3", "total": "4"}},
        "c": {"dtype": "bool", "summary": {"true_percent": "0.75"}},
        "d": {
            "dtype": "datetime64[ns]",
            "summary": {"timespan": "60 days, 2880 hours, 0 minutes, 0 seconds"},
        },
    }
    with open(fname) as f:
        actual_output = json.load(f)
    assert expected_output == actual_output


def test_cuxfilter_dashboard(test_dataframe):
    """Test generating the dashboard"""
    eda = EDA(test_dataframe)
    dash = eda.cuxfilter_dashboard()
    assert isinstance(dash, cuxfilter.dashboard.DashBoard)
    assert len(dash.charts) == 2
    assert dash.title == "Exploratory Data Analysis"
