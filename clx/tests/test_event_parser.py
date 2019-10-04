import cudf
import pandas
from clx.parsers.event_parser import EventParser


class TestEventParserImpl(EventParser):
    def parse(self, dataframe, raw_column):
        return None


class TestEventParser(object):
    def setup(self):
        # Create Test Event Parser Implementation
        event_name = "eventName"
        columns = {"eventTypeId", "username"}
        self.event_regex = {
            "eventTypeId": "eventTypeId: ([0-9$]+)",
            "username": "username: ([a-z\.\-0-9$]+)",
        }
        self.event_parser = TestEventParserImpl(columns, event_name)
        

    def test_parse_raw_event(self):
        test_dataframe = cudf.DataFrame(
            [
                (
                    "Raw",
                    [
                        "eventTypeId: 1 \\nusername: foo",
                        "eventTypeId: 1 \\nusername: bar",
                    ],
                )
            ]
        )
        parsed_dataframe = self.event_parser.parse_raw_event(
            test_dataframe, "Raw", self.event_regex
        )
        expected_parsed_dataframe = cudf.DataFrame(
            [("eventTypeId", ["1", "1"]), ("username", ["foo", "bar"])]
        )
        # Equality checks issue: https://github.com/rapidsai/cudf/issues/1750
        assert parsed_dataframe.to_pandas().equals(
            expected_parsed_dataframe.to_pandas()
        )
