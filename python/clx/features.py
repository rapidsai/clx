# Copyright (c) 2021, NVIDIA CORPORATION.
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
    """
    Create binary feature dataframe using provided dataset, entity, and feature.

    :param values: dataframe
    :type values: cudf.DataFrame
    :param values: entity_id
    :type values: str
    :param values: feature_id
    :type values: str
    :return: dataframe
    :rtype: cudf.DataFrame

    Examples
    --------
    >>> import cudf
    >>> import clx.features
    >>> df = cudf.DataFrame(
            {
                "time": [1, 2, 3],
                "user": ["u1", "u2", "u1",],
                "computer": ["c1", "c1", "c3"],
            }
        )
    >>> output = clx.features.binary(df, "user", "computer")
    >>> output
            c1	c3
        user
        u1	1.0	1.0
        u2	1.0	0.0
    """
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    p_df_grouped = df_grouped.to_pandas()
    p_df_grouped = pd.pivot_table(
        p_df_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=p_df_grouped.columns[1],
    ).fillna(0)
    p_df_grouped[p_df_grouped != 0.0] = 1
    output_dataframe = cudf.DataFrame.from_pandas(p_df_grouped)
    return output_dataframe


def frequency(dataframe, entity_id, feature_id):
    """
    Create frequency feature dataframe using provided dataset, entity, and feature.

    :param values: dataframe
    :type values: cudf.DataFrame
    :param values: entity_id
    :type values: str
    :param values: feature_id
    :type values: str
    :return: dataframe
    :rtype: cudf.DataFrame

    Examples
    --------
    >>> import cudf
    >>> import clx.features
    >>> df = cudf.DataFrame(
            {
                "time": [1, 2, 3],
                "user": ["u1", "u2", "u1",],
                "computer": ["c1", "c1", "c3"],
            }
        )
    >>> output = clx.features.binary(df, "user", "computer")
    >>> output
            c1	c3
        user
        u1	0.5	0.5
        u2	1.0	0.0
    """
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    p_df_grouped = df_grouped.to_pandas()
    p_df_grouped = pd.pivot_table(
        p_df_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=p_df_grouped.columns[1],
    ).fillna(0)
    p_df_grouped = p_df_grouped.apply(__get_prop, axis=1)
    output_dataframe = cudf.DataFrame.from_pandas(p_df_grouped)
    return output_dataframe
