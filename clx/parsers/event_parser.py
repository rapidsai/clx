import cudf
import logging
import nvstrings
import yaml

from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class EventParser(ABC):
    """This is an abstract class for all event log parsers.
    """

    def __init__(self, event_regex, event_types_filter=None):
        self._event_types_filter = event_types_filter
        self._event_regex = event_regex

    @property
    def event_types_filter(self):
        """arr[str]: This property is an optional filter which holds the events to be parsed.
           The events listed here should match regex configuration key values.
        """
        return self._event_types_filter

    @property
    def event_regex(self):
        """dict: Keys represent event type and values are a dictionary representing regex for the event type."""
        return self._event_regex

    @abstractmethod
    def parse(self, dataframe, raw_column):
        """Abstract method 'parse' triggers the parsing functionality.
           Subclasses are required to implement and execute any parsing pre-processing steps. """
        log.info("Begin parsing of dataframe")
        pass

    def parse_raw_event(self, dataframe, raw_column, event):
        """Processes parsing of a specific type of raw event records received as a dataframe
        """
        log.debug(
            "Parsing raw events. Event type: "
            + event
            + " DataFrame shape: "
            + str(dataframe.shape)
        )
        column_dict = self._event_regex[event]
        parsed_gdf = cudf.DataFrame([(col, [""]) for col in column_dict])
        parsed_gdf = parsed_gdf[:0]

        # Applies regex pattern for each expected output column to raw data
        for col in column_dict:
            regex_pattern = column_dict.get(col)
            extracted_nvstrings = dataframe[raw_column].str.extract(regex_pattern)
            if not extracted_nvstrings.empty:
                parsed_gdf[col] = extracted_nvstrings[0]

        # Applies the intended datatype (string) to each column of the processed output
        for col in column_dict:
            if not parsed_gdf[col].empty:
                if parsed_gdf[col].dtype == "float64":
                    parsed_gdf[col] = gdf[col].astype("int").astype("str")
                elif parsed_gdf[col].dtype == "object":
                    pass
                else:
                    parsed_gdf[col] = gdf[col].astype("str")
            if parsed_gdf[col].empty:
                parsed_gdf[col] = nvstrings.to_device([])

        log.debug("Completed parsing raw events")
        return parsed_gdf

    def _filter_by_pattern(self, df, column, pattern):
        """Filter based on whether a string contains a regex pattern
        """
        df["present"] = df[column].str.contains(pattern)
        return df[df.present == True]

    def _load_regex_yaml(self, yaml_file):
        """Returns a dictionary of the regex contained in the given yaml file"""
        with open(yaml_file) as yaml_file:
            regex_dict = yaml.safe_load(yaml_file)
            regex_dict = {k: v[0] for k, v in regex_dict.items()}
        return regex_dict
