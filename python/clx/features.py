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
import pandas as pd


def __get_prop(row):
    if row.sum() == 0:
        return pd.Series([0.0] * row.shape[0])
    else:
        return row / row.sum()


def binary(dataframe, entity_id, feature_id):
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    # print(df_grouped)
    p_df_grouped = df_grouped.to_pandas()
    p_df_grouped = pd.pivot_table(
        p_df_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=p_df_grouped.columns[1],
    ).fillna(0)
    # print(p_df_grouped)
    p_df_grouped[p_df_grouped != 0.0] = 1
    output_dataframe = cudf.DataFrame.from_pandas(p_df_grouped)
    return output_dataframe


def frequency(dataframe, entity_id, feature_id):
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    # print(df_grouped)
    p_df_grouped = df_grouped.to_pandas()
    p_df_grouped = pd.pivot_table(
        p_df_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=p_df_grouped.columns[1],
    ).fillna(0)
    # print(p_df_grouped)
    p_df_grouped = p_df_grouped.apply(__get_prop, axis=1)
    output_dataframe = cudf.DataFrame.from_pandas(p_df_grouped)
    return output_dataframe
