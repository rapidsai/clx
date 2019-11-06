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

import dask, dask_cudf
import logging
from clx.io.reader.file_reader import FileReader

log = logging.getLogger(__name__)


class DaskFileSystemReader(FileReader):
    def __init__(self, config):
        self._config = config
        self._has_data = True

    def fetch_data(self):
        df = None
        input_format = self.config["input_format"].lower()
        filepath = self.config["input_path"].lower()
        kwargs = self.config.copy()
        del kwargs["type"]
        del kwargs["input_format"]
        del kwargs["input_path"]

        if "parquet" == input_format:
            df = dask_cudf.read_parquet(filepath, **kwargs)
        elif "orc" == input_format:
            df = dask_cudf.read_orc(filepath, engine="cudf")
        else:
            df = dask_cudf.read_csv(filepath, **kwargs)

        self.has_data = False
        return df

    def close(self):
       log.info("Closed dask_fs reader")
