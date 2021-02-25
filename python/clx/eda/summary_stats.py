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

import cuxfilter

from clx.eda.analysis import Analysis


class SummaryStatistics(Analysis):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def __summary_obj(self, series):
        summary = {}
        uniq_count = len(series.unique())
        total = series.notna().sum()
        summary["unique"] = str(uniq_count)
        summary["total"] = str(total)
        return summary

    def __summary_bool(self, series):
        summary = {}
        true_per = (series == True).sum()  # noqa: E712
        summary["true_percent"] = str(true_per / len(series))
        return summary

    def __summary_num(self, series):
        summary = {}
        uniq_count = len(series.unique())
        total = series.notna().sum()
        summary["unique"] = str(uniq_count)
        summary["total"] = str(total)
        return summary

    def __summary_time(self, series):
        summary = {}
        duration = series.max() - series.min()
        days = duration.astype("timedelta64[D]").astype(int)
        seconds = duration.astype("timedelta64[s]").astype(int)
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        msg = "{0} days, {1} hours, {2} minutes, {3} seconds".format(
            days, hours, minutes, seconds
        )
        summary["timespan"] = msg
        return summary

    def _generate_analysis(self, dataframe):
        # This function will receive a dataframe and returns a dictionary of summary statistics
        summary_dict = {}
        for col in dataframe.columns:
            summary_dict[col] = {}
            summary_dict[col]["dtype"] = str(dataframe[col].dtype)
            if dataframe[col].dtype == "object":
                summary_dict[col]["summary"] = self.__summary_obj(dataframe[col])
            elif dataframe[col].dtype == "bool":
                summary_dict[col]["summary"] = self.__summary_bool(dataframe[col])
            elif dataframe[col].dtype in ["int64", "float64", "int8"]:
                summary_dict[col]["summary"] = self.__summary_num(dataframe[col])
            elif dataframe[col].dtype == "datetime64[ns]":
                summary_dict[col]["summary"] = self.__summary_time(dataframe[col])
            else:
                msg = "\t column type (" + str(dataframe[col].dtype) + ") not supported"
                summary_dict[col]["error"] = msg
        return summary_dict

    def _generate_charts(self, dataframe):
        """Get barcharts for the summary analysis"""
        charts = []
        for col in dataframe.columns:
            if dataframe[col].dtype == "object":
                bars = len(dataframe[col].unique())
                if bars < 30:
                    if bars > 1:
                        charts.append(cuxfilter.charts.bar(col))
        return charts
