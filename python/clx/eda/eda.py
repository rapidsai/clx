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

import cuxfilter
from cuxfilter.layouts import feature_and_double_base

from clx.eda.summary_stats import SummaryStatistics


class EDA:
    modules = {"SummaryStatistics": SummaryStatistics}

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.module_output = {}
        self.analysis = {}
        self.analysis = self.__generate_analysis(dataframe)

    def __repr__(self):
        return json.dumps(self.analysis, indent=2)

    def __generate_analysis(self, dataframe):
        # For each of the modules, generate the analysis
        output = {}
        for key, analysis in self.modules.items():
            analysis_output = analysis(dataframe)
            output[key] = analysis_output.analysis
            self.module_output[key] = analysis_output
        return output

    def save_analysis(self):
        for key, analysis in self.module_output.items():
            output_file = key + ".out"
            print("saving output to", output_file)
            analysis.save_analysis(output_file)

    def cuxfilter_dashboard(self, dataframe):
        for module in self.module_output.values():
            c = module.get_charts(dataframe)
        cux_df = cuxfilter.DataFrame.from_dataframe(dataframe)
        return cux_df.dashboard(
            c,
            layout=feature_and_double_base,
            theme=cuxfilter.themes.light,
            title="EDA Prototype",
        )
