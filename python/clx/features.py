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


def binary(dataframe, entity_id, feature_id):
    """
    Create binary feature dataframe using provided dataset, entity, and feature.

    :param dataframe: Input dataframe to create binary features
    :type dataframe: cudf.DataFrame
    :param entity_id: Entity ID. Must be a column within `dataframe`
    :type entity_id: str
    :param feature_id: Feature ID. Must be a column within `dataframe`
    :type feature_id: str
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
    df_grouped = dataframe.groupby([entity_id, feature_id]).count().reset_index()
    df_output = df_grouped.pivot(index=entity_id, columns=feature_id).fillna(0)
    df_output[df_output != 0] = 1
    return df_output


def frequency(dataframe, entity_id, feature_id):
    """
    Create frequency feature dataframe using provided dataset, entity, and feature.

    :param dataframe: Input dataframe to create binary features
    :type dataframe: cudf.DataFrame
    :param entity_id: Entity ID. Must be a column within `dataframe`
    :type entity_id: str
    :param feature_id: Feature ID. Must be a column within `dataframe`
    :type feature_id: str
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
    df_grouped = dataframe.groupby([entity_id, feature_id]).count().reset_index()
    df_output = df_grouped.pivot(index=entity_id, columns=feature_id).fillna(0)
    sum_col = df_output.sum(axis=1)
    for col in df_output.columns:
        df_output[col] = df_output[col] / sum_col
    return df_output
