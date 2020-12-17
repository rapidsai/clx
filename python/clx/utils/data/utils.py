# Copyright (c) 2020, NVIDIA CORPORATION.
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

log = logging.getLogger(__name__)


def str2ascii(df, domains_len):
    """
    This function sorts domain name entries in desc order based on the length of domain and converts domain name to ascii characters.

    :param df: Domains which requires conversion.
    :type df: cudf.DataFrame
    :param domains_len: Number of entries in df.
    :type domains_len: int
    :return: Ascii character converted information.
    :rtype: cudf.DataFrame
    """
    df["len"] = df["domain"].str.len()
    df = df.sort_values("len", ascending=False)
    split_df = df["domain"].str.findall("[\w\.\-\@]")
    columns_cnt = len(split_df.columns)

    # Replace null's with ^.
    split_df = split_df.fillna("^")
    temp_df = cudf.DataFrame()
    for col in range(0, columns_cnt):
        temp_df[col] = split_df[col].str.code_points()
    del split_df

    # Replace ^ ascii value 94 with 0.
    temp_df = temp_df.replace(94, 0)
    temp_df.index = df.index
    temp_df["len"] = df["len"]
    if "type" in df.columns:
        temp_df["type"] = df["type"]
    temp_df["domain"] = df["domain"]
    return temp_df
