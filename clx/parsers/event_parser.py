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
import logging
import nvstrings
import yaml

from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class EventParser(ABC):
    """This is an abstract class for all event log parsers.
    """

    def __init__(self, columns, event_name):
        self._columns = columns
        self._event_name = event_name

    @property
    def columns(self):
        """List of columns required to parse from the yaml file"""
        return self._columns

    @property
    def event_name(self):
        """Event name define type of logs that are being processed."""
        return self._event_name

    @abstractmethod
    def parse(self, dataframe, raw_column):
        """Abstract method 'parse' triggers the parsing functionality.
           Subclasses are required to implement and execute any parsing pre-processing steps. """
        log.info("Begin parsing of dataframe")
        pass

    def parse_raw_event(self, dataframe, raw_column, event_regex):
        """Processes parsing of a specific type of raw event records received as a dataframe
        """
        log.debug(
            "Parsing raw events. Event type: "
            + self.event_name
            + " DataFrame shape: "
            + str(dataframe.shape)
        )
        parsed_gdf = cudf.DataFrame({col: [""] for col in self.columns})
        parsed_gdf = parsed_gdf[:0]
        event_specific_columns = event_regex.keys()
        # Applies regex pattern for each expected output column to raw data
        for col in event_specific_columns:
            regex_pattern = event_regex.get(col)
            extracted_nvstrings = dataframe[raw_column].str.extract(regex_pattern)
            if not extracted_nvstrings.empty:
                parsed_gdf[col] = extracted_nvstrings[0]

        remaining_columns = list(self.columns - event_specific_columns)
        # Fill remaining columns with empty.
        for col in remaining_columns:
            parsed_gdf[col] = ""

        return parsed_gdf

    def filter_by_pattern(self, df, column, pattern):
        """Filter based on whether a string contains a regex pattern
        """
        df["present"] = df[column].str.contains(pattern)
        return df[df.present == True]

    def _load_regex_yaml(self, yaml_file):
        """Returns a dictionary of the regex contained in the given yaml file"""
        with open(yaml_file) as yaml_file:
            regex_dict = yaml.safe_load(yaml_file)
        return regex_dict
