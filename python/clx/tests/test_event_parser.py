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
from clx.parsers.event_parser import EventParser


class TestEventParserImpl(EventParser):
    def parse(self, dataframe, raw_column):
        return None


class TestEventParser(object):
    def setup(self):
        # Create Test Event Parser Implementation
        event_name = "eventName"
        columns = ["eventTypeId", "username"]
        self.event_regex = {
            "eventTypeId": r"eventTypeId: ([0-9$]+)",
            "username": r"username: ([a-z\.\-0-9$]+)",
        }
        self.event_parser = TestEventParserImpl(columns, event_name)

    def test_parse_raw_event(self):
        test_dataframe = cudf.DataFrame(
            {
                "Raw": [
                    "eventTypeId: 1 \\nusername: foo",
                    "eventTypeId: 1 \\nusername: bar",
                ]
            }
        )
        parsed_dataframe = self.event_parser.parse_raw_event(
            test_dataframe, "Raw", self.event_regex
        )
        expected_parsed_dataframe = cudf.DataFrame(
            {"eventTypeId": ["1", "1"], "username": ["foo", "bar"]}
        )

        assert parsed_dataframe.equals(expected_parsed_dataframe)
