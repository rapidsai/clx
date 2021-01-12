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
    def __init__(self):
        self.analysis = {}

    @abstractmethod
    def generate_analysis(self, dataframe):
        # Returns a dictionary
        pass

    def to_json(self):
        """Get json version of analysis results"""
        return json.dumps(self.analysis, indent=2)

    def save_analysis(self, output_filepath, format="json"):
        formatted_output = ""
        if format == "json":
            formatted_output = self.to_json()
        else:
            raise NotImplementedError()
        with open(output_filepath, "w") as file:
            file.write(formatted_output)
