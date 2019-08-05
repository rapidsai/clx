from abc import ABC, abstractmethod
import os
from rapidscyber.io.factory.factory import Factory
import logging
import yaml

log = logging.getLogger(__name__)


class Workflow(ABC):

    DEFAULT_CONFIG_PATH = "/.config/rapidscyber"
    BACKUP_CONFIG_PATH = "/etc/rapidscyber/"

    def __init__(self, name, source=None, destination=None):
        # Initialize properties
        self._source = None
        self._destination = None
        self._name = name

        # Check to see if workflow yaml file exists. If so, set workflow configurations from file.
        config_file = "{0}/{1}/workflow.yaml"
        home_dir = os.getenv("HOME")
        default_dir = home_dir + self.DEFAULT_CONFIG_PATH
        default_filepath = config_file.format(default_dir, name)
        backup_filepath = config_file.format(self.BACKUP_CONFIG_PATH, name)
        if os.path.exists(default_filepath):
            log.info("Config file detected: {0}".format(default_filepath))
            self._set_workflow_config(default_filepath)
        elif os.path.exists(backup_filepath):
            log.info("Config file detected: {0}".format(backup_filepath))
            self._set_workflow_config(backup_filepath)
        else:
            log.info("No config file detected.")

        # If source or destination are passed in as parameters, update source and dest configurations.
        if source:
            self._source = source
            self._io_reader = Factory.get_reader(self._source["type"], self._source)
        if destination:
            self._destination = destination
            self._io_writer = Factory.get_writer(
                self._destination["type"], self._destination
            )

    def _set_workflow_config(self, yaml_file):
        # Receives a yaml file path with Workflow configurations and sets appropriate values for properties in this class
        log.info("Setting configurations from config file {0}".format(yaml_file))
        try:
            with open(yaml_file, "r") as ymlfile:
                config = yaml.load(ymlfile)
            self._source = config["source"]
            self._destination = config["destination"]
            self._io_reader = Factory.get_reader(self._source["type"], self._source)
            self._io_writer = Factory.get_writer(
                self._destination["type"], self._destination
            )
        except:
            log.error(
                "Error creating I/O reader and writer. Please check configurations in workflow config file at {0}".format(
                    yaml_file
                )
            )
            raise

    @property
    def name(self):
        """str: The name of the workflow for logging purposes."""
        return self._name

    @property
    def source(self):
        """dict: Configuration parameters for the data source"""
        return self._source

    def set_source(self, source):
        self._source = source
        self._io_reader = Factory.get_reader(self.source["type"], self.source)

    @property
    def destination(self):
        """dict: Configuration parameters for the data destination"""
        return self._destination

    def set_destination(self, destination):
        self._destination = destination
        self._io_writer = Factory.get_writer(
            self.source["destination"], self.destination
        )

    def _get_parser(self, parser_config):
        """TODO: Private helper function that fetches a specific parser based upon configuration"""
        pass

    def run_workflow(self):
        log.info("Running workflow {0}.".format(self.name))
        try:
            while (
                self._io_reader.has_data
            ):
                dataframe = (
                    self._io_reader.fetch_data()
                )
                if dataframe:
                    enriched_dataframe = self.workflow(dataframe)
                    self._io_writer.write_data(enriched_dataframe)
        except KeyboardInterrupt:
            logging.info("User aborted workflow")
            self.stop_workflow()

    def stop_workflow(self):
        log.info("Closing workflow...")
        self._io_reader.close()
        self._io_writer.close()
        log.info("Workflow {0} stopped.".format(self.name))

    @abstractmethod
    def workflow(self, dataframe):
        """The pipeline function performs the data enrichment on the data.
        Subclasses must define this function. This function will return a gpu dataframe with enriched data."""
        pass
