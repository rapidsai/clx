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

from clx.io.factory.abstract_factory import AbstractFactory
from clx.io.reader.fs_reader import FileSystemReader
from clx.io.writer.fs_writer import FileSystemWriter


class FileSystemFactory(AbstractFactory):
    def __init__(self, config):
        """
        Constructor method

        :param config: dictionary object of config values for **type**, **input_format**, **input_path** (or **output_path**), and dask reader/writer optional keyword args
        """
        self._config = config

    def get_reader(self):
        """
        Get instance of FileSystemReader
        """
        return FileSystemReader(self.config)

    def get_writer(self):
        return FileSystemWriter(self.config)
