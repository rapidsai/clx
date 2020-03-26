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

import cudf
import logging
import os

from clx.io.writer.file_writer import FileWriter

log = logging.getLogger(__name__)


class FileSystemWriter(FileWriter):
    """
    Uses cudf to write to file system based on config object.

    :param config: dictionary object of config values for **type**, **output_format**, **output_path** (or **output_path**), and cudf writer optional keyword args
    """

    def __init__(self, config):
        self._config = config

    def write_data(self, df):
        """
        Write data to file system using cudf based on provided config object
        """
        output_format = self.config["output_format"].lower()
        filepath = self.config["output_path"]
        kwargs = self.config.copy()
        del kwargs["type"]
        del kwargs["output_format"]
        del kwargs["output_path"]

        dir = os.path.dirname(filepath)
        if not os.path.isdir(dir):
            log.info("output directory { %s } not exist" % (dir))
            log.info("creating output directory { %s }..." % (dir))
            os.makedirs(dir)
            log.info("created output directory { %s }..." % (dir))
        if os.path.exists(filepath):
            raise IOError("output path { %s } already exist" % (filepath))

        log.info("writing data to location {%s}" % (filepath))

        if "csv" == output_format:
            df.to_csv(filepath, **kwargs)
        elif "parquet" == output_format:
            cudf.io.parquet.to_parquet(df, filepath, **kwargs)
        elif "orc" == output_format:
            cudf.io.orc.to_orc(df, filepath, **kwargs)
        elif "json" == output_format:
            cudf.io.json.to_json(df, filepath, **kwargs)
        else:
            raise NotImplementedError("%s is not a supported output_format" % (output_format))

    def close(self):
        """
        Close cudf writer
        """
        log.info("Closed writer")
