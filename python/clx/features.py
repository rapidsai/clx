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
    if entity_id and feature_id not in dataframe.columns:
        raise Exception(
            "{0} and {1} must be column names in the input dataframe".format(
                entity_id, feature_id
            )
        )
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    # https://github.com/rapidsai/cudf/issues/1214
    pdf_grouped = df_grouped.to_pandas()
    pdf_pivot = pd.pivot_table(
        pdf_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=pdf_grouped.columns[1],
    ).fillna(0)
    df_output = cudf.DataFrame.from_pandas(pdf_pivot)
    df_output[df_output != 0.0] = 1
    return df_output


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
    if entity_id and feature_id not in dataframe.columns:
        raise Exception(
            "{0} and {1} must be column names in the input dataframe".format(
                entity_id, feature_id
            )
        )
    df_grouped = (
        dataframe.groupby([entity_id, feature_id])
        .count()
        .reset_index()
        .set_index(entity_id)
    )
    # https://github.com/rapidsai/cudf/issues/1214
    pdf_grouped = df_grouped.to_pandas()
    pdf_pivot = pd.pivot_table(
        pdf_grouped,
        index=[entity_id],
        columns=[feature_id],
        values=pdf_grouped.columns[1],
    ).fillna(0)
    output_df = cudf.DataFrame.from_pandas(pdf_pivot)
    sum_col = output_df.sum(axis=1)
    for col in output_df.columns:
        output_df[col] = output_df[col] / sum_col
    return output_df
