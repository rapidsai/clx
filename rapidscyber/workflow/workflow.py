from abc import ABC
from os import path
from io.factory import Factory
import logging
import yaml


log = logging.getLogger("Workflow")


class Workflow(ABC):

    DEFAULT_CONFIG_FILE = "workflow.yaml"

    def __init__(self, source=None, destination=None):
        # Set configurations if default config file is detected.
        if path.exists(DEFAULT_CONFIG_FILE):
            log.info("Config file detected: {0}".format(DEFAULT_CONFIG_FILE))
            self._set_workflow_config(DEFAULT_CONFIG_FILE)
        else:
            log.info("No config file detected: {0}".format(DEFAULT_CONFIG_FILE))
        # Set configurations for source and destination parameters.
        if source:
            self._source = source
        if destination:
            self._destination = destination
        # Set reader and writer
        self._io_reader = Factory.get_reader(self._source["type"], self._source)
        self._io_writer = Factory.get_writer(
            self._source["destination"], self._destination
        )

    def _set_workflow_config(self, yaml_file):
        log.info(
            "Setting configurations from config file {0}".format(DEFAULT_CONFIG_FILE)
        )
        with open(yaml_file, "r") as ymlfile:
            config = yaml.load(ymlfile)
        if config["source"]:
            self._source = config["source"]
        if config["destination"]:
            self._destination = config["destination"]

    @property
    def source(self):
        """dict: Configuration parameters for the data source"""
        return self._source

    def set_source(self, source):
        self._source = source
        self._io_reader = Factory.get_reader(self._source["type"], self._source)

    @property
    def destination(self):
        """dict: Configuration parameters for the data destination"""
        return self._destination

    def set_destination(self, destination):
        self._destination = destination
        self._io_writer = Factory.get_writer(
            self._source["destination"], self._destination
        )

    def _get_parser(self, parser_config):
        """TODO: Private helper function that fetches a specific parser based upon configuration"""
        pass

    def run_workflow(self):
        try:
            while (
                self._io_reader.has_data()
            ):  # for a file this will be true only once. for streaming this will always return true
                dataframe = (
                    self._io_reader.fetch_data()
                )  # if kafka queue is empty just return None,
                if dataframe:
                    enriched_dataframe = pipeline(dataframe)
                    self._io_writer.write_data(enriched_dataframe)
        except KeyboardInterrupt:
            self.stop_workflow()

    def stop_workflow(self):
        log.info("Workflow stopped")

    @abstract_method
    def workflow(self, dataframe):
        """The pipeline function performs the data enrichment on the data.
        Subclasses must define this function. This function will return a gpu dataframe with enriched data."""
        pass
