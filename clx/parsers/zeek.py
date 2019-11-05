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

type_dict = {
    "bool": "bool",
    "count": "int64",
    "int": "int64",
    "double": "float64",
    "time": "float64",
    "interval": "float64",
    "string": "str",
    "pattern": "str",
    "port": "int64",
    "addr": "str",
    "subnet": "str",
    "enum": "str",
    "function": "str",
    "event": "str",
    "hook": "str",
    "file": "str",
    "opaque": "str",
    "any": "str",
}


def parse_log_file(filepath):
    """Parse Zeek log file and return cuDF dataframe. Uses header comments to get column names/types and configure parser.

    Parameters
    ----------
    filepath : string
        filepath for Zeek log file

    Returns
    -------
    Zeek log dataframe : cudf.Dataframe
    """

    header_gdf = cudf.read_csv(filepath, names=["line"], nrows=8)
    lines = header_gdf["line"].str.split_record()
    column_names = lines[6][1 : len(lines[6])].to_host()
    column_types = lines[7][1 : len(lines[7])].to_host()
    column_dtypes = list(map(lambda x: type_dict.get(x, "str"), column_types))
    log_gdf = cudf.read_csv(
        filepath,
        delimiter="\t",
        dtype=column_dtypes,
        names=column_names,
        skiprows=8,
        skipfooter=1,
    )
    return log_gdf
