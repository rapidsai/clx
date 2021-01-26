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
from abc import ABC, abstractmethod


class Analysis(ABC):
    def __init__(self, dataframe):
        self._analysis = self._generate_analysis(dataframe)
        self._charts = self._generate_charts(dataframe)

    @property
    def analysis(self):
        return self._analysis

    @property
    def charts(self):
        return self._charts

    @abstractmethod
    def _generate_analysis(self, dataframe):
        """Abstract function intended to create a dictionary summarizing analysis results of the dataframe"""
        pass

    @abstractmethod
    def _generate_charts(self, dataframe):
        """Abstract function intended to create a list of cuxfilt"""
        pass

    def to_json(self):
        """Get json version of analysis results"""
        return json.dumps(self.analysis, indent=2)

    def save_analysis(self, output_filepath):
        """Save analysis to a json file
        TODO: Expand to other output types"""
        formatted_output = self.to_json()
        with open(output_filepath + ".json", "w") as file:
            file.write(formatted_output)
