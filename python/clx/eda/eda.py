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
import os

import cuxfilter
from cuxfilter.layouts import feature_and_double_base

from clx.eda.summary_stats import SummaryStatistics


class EDA:
    """An EDA (Exploratory Data Analysis) Object. EDA is used to explore different features of a given dataframe.

    :param dataframe: Dataframe to be used for analysis
    :type dataframe: cudf.DataFrame

    Examples
    --------
    >>> from clx.eda import EDA
    >>> import cudf
    >>> import pandas as pd
    >>> df = cudf.DataFrame()
    >>> df['a'] = [1,2,3,4]
    >>> df['b'] = ['a','b','c','c']
    >>> df['c'] = [True, False, True, True]
    >>> df['d'] = cudf.Series(pd.date_range("2000-01-01", periods=3,freq="m"))
    >>> eda = EDA(df)
    >>> eda
        {
            "SummaryStatistics": {
                "a": {
                    "dtype": "int64",
                    "summary": {
                        "unique": "4",
                        "total": "4"
                    }
                },
                "b": {
                    "dtype": "object",
                    "summary": {
                        "unique": "3",
                        "total": "4"
                    }
                },
                "c": {
                    "dtype": "bool",
                    "summary": {
                        "true_percent": "0.75"
                    }
                },
                "d": {
                    "dtype": "datetime64[ns]",
                    "summary": {
                        "timespan": "60 days, 2880 hours, 0 minutes, 0 seconds"
                    }
                }
            }
        }
    """

    eda_modules = {"SummaryStatistics": SummaryStatistics}

    def __init__(self, dataframe):
        self.__dataframe = dataframe
        self.__analysis, self.__module_ref = self.__generate_analysis(dataframe)

    @property
    def analysis(self):
        """
        Analysis results as a `dict`
        """
        return self.__analysis

    @property
    def dataframe(self):
        """
        Dataframe used for analysis
        """
        return self.__dataframe

    def __repr__(self):
        return json.dumps(self.analysis, indent=2)

    def __generate_analysis(self, dataframe):
        """For each of the modules, generate the analysis"""
        module_ref = {}
        analysis_results = {}
        for key, eda_module in self.eda_modules.items():
            eda_module_obj = eda_module(dataframe)
            module_ref[key] = eda_module_obj
            analysis_results[key] = eda_module_obj.analysis
        return analysis_results, module_ref

    def save_analysis(self, dirpath):
        """Save analysis output to directory path.

        :param dirpath: Directory path to save analysis output.
        :type dirpath: str
        """
        for key, analysis in self.__module_ref.items():
            if os.path.isdir(dirpath):
                output_file = dirpath + "/" + key
                analysis.save_analysis(output_file)

    def cuxfilter_dashboard(self):
        """Create cuxfilter dashboard for Exploratory Data Analysis.

        :return: cuxfilter dashboard with populated with data and charts.
        :rtype: cuxfilter.DashBoard
        """
        for module in self.__module_ref.values():
            charts = module.charts
        cux_df = cuxfilter.DataFrame.from_dataframe(self.__dataframe)
        return cux_df.dashboard(
            charts,
            layout=feature_and_double_base,
            theme=cuxfilter.themes.light,
            title="Exploratory Data Analysis",
        )
