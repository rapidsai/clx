import logging
import os
import cudf
from clx.parsers.event_parser import EventParser

log = logging.getLogger(__name__)


class WindowsEventParser(EventParser):

    REGEX_FILE = "resources/windows_event_regex.yaml"
    EVENT_NAME = "windows event"

    def __init__(self, interested_eventcodes=None):
        event_regex = {}
        regex_filepath = (
            os.path.dirname(os.path.abspath(__file__)) + "/" + self.REGEX_FILE
        )
        self.interested_eventcodes = interested_eventcodes
        self.event_regex = self._load_regex_yaml(regex_filepath)
        EventParser.__init__(self, self.get_columns(), self.EVENT_NAME)

    def parse(self, dataframe, raw_column):
        """Parses the windows raw evenst"""
        # Clean raw data to be consistent.
        dataframe = self.clean_raw_data(dataframe, raw_column)
        output_chunks = []
        for eventcode in self.event_regex.keys():
            pattern = "eventcode=%s" % (eventcode)
            input_chunk = self.filter_by_pattern(dataframe, raw_column, pattern)
            temp = self.parse_raw_event(
                input_chunk, raw_column, self.event_regex[eventcode]
            )
            output_chunks.append(temp)
        parsed_dataframe = cudf.concat(output_chunks)
        # Replace null values with empty.
        parsed_dataframe = parsed_dataframe.fillna("")
        return parsed_dataframe

    def clean_raw_data(self, dataframe, raw_column):
        """Lower casing and replacing characters"""
        dataframe[raw_column] = (
            dataframe[raw_column]
            .str.lower()
            .str.replace("\\\\t", "")
            .str.replace("\\\\r", "")
            .str.replace("\\\\n", "|")
        )
        return dataframe

    def _load_regex_yaml(self, yaml_file):
        event_regex = EventParser._load_regex_yaml(self, yaml_file)
        if self.interested_eventcodes is not None:
            for eventcode in self.interested_eventcodes:
                required_event_regex = {}
                if eventcode not in event_regex:
                    raise KeyError(
                        "Regex for eventcode %s is not available in the config file. Please choose from %s"
                        % (eventcode, list(event_regex.keys()))
                    )
                required_event_regex[eventcode] = event_regex[eventcode]
            return required_event_regex
        return event_regex

    def get_columns(self):
        columns = set()
        for key in self.event_regex.keys():
            for column in self.event_regex[key].keys():
                columns.add(column)
        return columns
