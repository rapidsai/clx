import logging
import os
from rapidscyber.parsers.event_parser import EventParser

log = logging.getLogger(__name__)

class SplunkNotableParser(EventParser):

    REGEX_FILE = "resources/splunk_notable_regex.yaml"
    EVENT_NAME = "notable"
    COLUMNS = ["time","search_name","orig_time","urgency","user","owner","security_domain","severity","src_ip","src_ip2","src_mac","src_port","dest_ip","dest_ip2","dest_mac","dest_port","dest_priority","device_name","event_name","event_type","id","ip_address"]
        

    def __init__(self):
        event_regex = {}
        regex_filepath = os.path.dirname(os.path.abspath(__file__)) + '/' + self.REGEX_FILE
        event_regex[self.EVENT_NAME] = self._load_regex_yaml(regex_filepath)
        EventParser.__init__(self, event_regex, self.COLUMNS)

    def parse(self, dataframe, raw_column):
        """Parses the Splunk notable raw event"""
        parsed_dataframe = self.parse_raw_event(dataframe, raw_column, self.EVENT_NAME)
        # Post-processing: Merge dest_ip and src_ip extracted strings
        parsed_dataframe['dest_ip'] = parsed_dataframe['dest_ip'].str.cat(parsed_dataframe['dest_ip2'], sep=',')
        parsed_dataframe['src_ip'] = parsed_dataframe['src_ip'].str.cat(parsed_dataframe['src_ip2'], sep=',')
        log.debug("Parsed notable dataframe: " + parsed_dataframe.head())
        return parsed_dataframe