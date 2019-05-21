import cudf
import logging
import nvstrings

from abc import ABC, abstractmethod

log = logging.getLogger("EventParser")


class EventParser(ABC):
    """This is an abstract class for all event log parsers.
    """

    def __init__(self, event_regex, columns, event_types_filter=None):
        self._event_types_filter = event_types_filter
        self._event_regex = event_regex
        self._columns = columns

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

    @property
    def columns(self):
        """arr[str]: describes the dataframe columns to create for parsed key/value pairs from the event"""
        return self._columns

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
        parsed_gdf = cudf.DataFrame([(col, [""]) for col in self._columns])
        parsed_gdf = parsed_gdf[:0]

        # Applies regex pattern for each expected output column to raw data
        for col in self._columns:
            regex_pattern = self._event_regex[event].get(col)
            extracted_nvstrings = dataframe[raw_column].str.extract(regex_pattern)
            if not extracted_nvstrings.empty:
                parsed_gdf[col] = extracted_nvstrings[0]

        # Applies the intended datatype (string) to each column of the processed output
        for col in self._columns:
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
