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

import functools
import logging
import os
import time
import yaml
from yaml import Loader
from clx.io.factory.factory import Factory
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class Workflow(ABC):

    DEFAULT_CONFIG_PATH = "/.config/clx"
    BACKUP_CONFIG_PATH = "/etc/clx/"
    CONFIG_FILE_NAME = "workflow.yaml"
    DEFAULT_PARAMS = ["name", "destination", "source"]

    def benchmark(function):
        """
           Decorator used to capture a benchmark for a given function
        """

        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            ret = function(self, *args, **kwargs)
            end = time.time()
            runtime = end - start
            log.info(
                f"Workflow benchmark for function {function.__name__!r}: {runtime:.4f} seconds"
            )
            return ret

        return wrapper

    def __init__(self, name, source=None, destination=None):
        # Initialize properties
        self._source = None
        self._destination = None
        self._name = name

        # Check to see if workflow yaml file exists. If so, set workflow configurations from file.
        default_filepath = self._get_default_filepath(name)
        backup_filepath = self._get_backup_filepath(name)
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

    def _get_default_filepath(self, workflow_name):
        home_dir = os.getenv("HOME")
        default_filepath = "{home_dir}/{default_sub_dir}/{workflow_name}/{filename}".format(
            home_dir=home_dir,
            default_sub_dir=self.DEFAULT_CONFIG_PATH,
            workflow_name=workflow_name,
            filename=self.CONFIG_FILE_NAME,
        )
        log.info("default filepath:" + default_filepath)
        return default_filepath

    def _get_backup_filepath(self, workflow_name):
        backup_filepath = "{backup_dir}/{workflow_name}/{filename}".format(
            backup_dir=self.BACKUP_CONFIG_PATH,
            workflow_name=workflow_name,
            filename=self.CONFIG_FILE_NAME,
        )
        log.info("backup filepath:" + backup_filepath)
        return backup_filepath

    def _set_workflow_config(self, yaml_file):
        # Receives a yaml file path with Workflow configurations and sets appropriate values for properties in this class
        log.info("Setting configurations from config file {0}".format(yaml_file))
        try:
            config = None
            with open(yaml_file, "r") as ymlfile:
                config = yaml.load(ymlfile, Loader=Loader)
            self._source = config["source"]
            self._destination = config["destination"]
            self._io_reader = Factory.get_reader(self._source["type"], self._source)
            self._io_writer = Factory.get_writer(
                self._destination["type"], self._destination
            )
            # Set attributes for custom workflow properties
            for key in config.keys():
                if key not in self.DEFAULT_PARAMS:
                    setattr(self, key, config[key])

        except Exception:
            log.error(
                "Error creating I/O reader and writer. Please check configurations in workflow config file at {0}".format(
                    yaml_file
                )
            )
            raise

    @property
    def name(self):
        """Name of the workflow for logging purposes."""
        return self._name

    @property
    def source(self):
        """
        Dictionary of configuration parameters for the data source (reader)
        """
        return self._source

    def set_source(self, source):
        """
        Set source.

        :param source: dict of configuration parameters for data source (reader)
        """
        self._source = source
        self._io_reader = Factory.get_reader(self.source["type"], self.source)

    @property
    def destination(self):
        """
        Dictionary of configuration parameters for the data destination (writer)
        """
        return self._destination

    def set_destination(self, destination):
        """
        Set destination.

        :param destination: dict of configuration parameters for the destination (writer)
        """
        self._destination = destination
        self._io_writer = Factory.get_writer(
            self.source["destination"], self.destination
        )

    def _get_parser(self, parser_config):
        """TODO: Private helper function that fetches a specific parser based upon configuration"""
        pass

    def run_workflow(self):
        """
        Run workflow. Reader (source) fetches data. Workflow implementation is executed. Workflow output is
        written to destination.
        """
        log.info("Running workflow {0}.".format(self.name))
        try:
            while (
                self._io_reader.has_data
            ):
                dataframe = (
                    self._io_reader.fetch_data()
                )
                enriched_dataframe = self.workflow(dataframe)
                if enriched_dataframe and not enriched_dataframe.empty:
                    self._io_writer.write_data(enriched_dataframe)
                else:
                    log.info("Dataframe is empty. Workflow processing skipped.")
        except KeyboardInterrupt:
            logging.info("User aborted workflow")
            self.stop_workflow()

    def stop_workflow(self):
        """
        Close workflow. This includes calling close() method on reader (source) and writer (destination)
        """
        log.info("Closing workflow...")
        self._io_reader.close()
        self._io_writer.close()
        log.info("Workflow {0} stopped.".format(self.name))

    @abstractmethod
    def workflow(self, dataframe):
        """The pipeline function performs the data enrichment on the data.
        Subclasses must define this function. This function will return a gpu dataframe with enriched data."""
        pass
