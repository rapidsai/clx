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

import logging
from clx.utils.data.dataset import Dataset
from clx.utils.data import utils

log = logging.getLogger(__name__)


class DGADataset(Dataset):
    """Constructor to create DGADataset instance.

        :param df: Input dataframe.
        :type df: cudf.DataFrame
        :param truncate: Truncate string to n number of characters.
        :type truncate: int
    """

    def __init__(self, df, truncate):
        df = self.__preprocess(df, truncate)
        super().__init__(df)

    def __preprocess(self, df, truncate):
        df['domain'] = df['domain'].str.slice_replace(truncate, repl='')
        df = utils.str2ascii(df, 'domain')
        return df
