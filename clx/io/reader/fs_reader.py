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
from clx.io.reader.file_reader import FileReader

log = logging.getLogger(__name__)


class FileSystemReader(FileReader):
    def __init__(self, config):
        self._config = config
        self._has_data = True

    def read_text(
        self, input_path, schema, delimiter, required_cols, datatypes, header
    ):
        df = cudf.read_csv(
            input_path,
            names=schema,
            delimiter=delimiter,
            usecols=required_cols,
            dtype=datatypes,
            header=header,
            skip_blank_lines=True,
        )

        return df

    def read_parquet(self, input_path, required_cols):
        df = cudf.read_parquet(input_path, columns=required_cols)
        return df

    def read_orc(self, input_path):
        df = cudf.read_orc(input_path, engine="cudf")
        return df

    def fetch_data(self):
        df = None
        input_format = self.config["input_format"].lower()
        if "parquet" == input_format:
            required_cols = self.config.get("required_cols", None)
            df = self.read_parquet(
                self.config["input_path"], self.config["required_cols"]
            )
        elif "orc" == input_format:
            df = self.read_orc(self.config["input_path"])
        else:
            schema = self.config.get("schema", None)
            delimiter = self.config.get("delimiter", ",")
            required_cols = self.config.get("required_cols", None)
            dtype = self.config.get("dtype", None)
            header = self.config.get("header", 0)
            df = self.read_text(
                self.config["input_path"],
                self.config["schema"],
                self.config["delimiter"],
                self.config["required_cols"],
                self.config["dtype"],
                self.config["header"],
            )
        self.has_data = False
        return df

    def close(self):
        log.info("Closed fs reader")
