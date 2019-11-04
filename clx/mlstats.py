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
