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

import logging
import math

log = logging.getLogger(__name__)


def rzscore(series, window):
    """
    Calculates rolling z-score

    Parameters
    ----------
    series : cudf.Series
        Series for which to calculate rolling z-score
    window : int
        Window size

    Returns
    -------
    cudf.Series
        Series with rolling z-score values

    Examples
    --------
    >>> import clx.analytics.stats
    >>> import cudf
    >>> sequence = [3,4,5,6,1,10,34,2,1,11,45,34,2,9,19,43,24,13,23,10,98,84,10]
    >>> series = cudf.Series(sequence)
    >>> zscores_df = cudf.DataFrame()
    >>> zscores_df['zscore'] = clx.analytics.stats.rzscore(series, 7)
    >>> zscores_df
                zscore
    0           null
    1           null
    2           null
    3           null
    4           null
    5           null
    6    2.374423424
    7   -0.645941275
    8   -0.683973734
    9    0.158832461
    10   1.847751909
    11   0.880026019
    12  -0.950835449
    13  -0.360593742
    14   0.111407599
    15   1.228914145
    16  -0.074966331
    17  -0.570321249
    18   0.327849973
    19  -0.934372308
    20   2.296828498
    21   1.282966989
    22  -0.795223674
    """
    rolling = series.rolling(window=window)
    mean = rolling.mean()
    std = rolling.apply(__std_func)

    zscore = (series - mean) / std
    return zscore


def __std_func(A):
    """
    Current implementation assumes ddof = 0
    """
    sum_of_elem = 0
    sum_of_square_elem = 0

    for a in A:
        sum_of_elem += a
        sum_of_square_elem += a * a

    s = (sum_of_square_elem - ((sum_of_elem * sum_of_elem) / len(A))) / len(A)
    return math.sqrt(s)
