import cudf
import pandas
from parser.event_parser import EventParser


class TestEventParserImpl(EventParser):
    def parse(self, dataframe, raw_column):
        return None

class TestEventParser(object):
    def setup(self):
        # Create Test Event Parser Implementation
        event_type1_regex = {
            "eventTypeId": "eventId: ([0-9$]+)",
            "username": "username: ([a-z\.\-0-9$]+)",
        }
        event_regex = {"eventTypeId1": event_type1_regex}
        columns = ["eventTypeId", "username"]
        self.event_parser = TestEventParserImpl(event_regex, columns)

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
            test_dataframe, "Raw", "eventTypeId1"
        )
        expected_parsed_dataframe = cudf.DataFrame(
            [("eventTypeId", ["1", "1"]), ("username", ["hello", "bar"])]
        )
        # Equality checks issue: https://github.com/rapidsai/cudf/issues/1750
        assert parsed_dataframe.to_pandas().equals(
            expected_parsed_dataframe.to_pandas()
        )
