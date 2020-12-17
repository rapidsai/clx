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
from clx.utils.data.dataset import Dataset
from clx.utils.data.dataloader import DataLoader

test_batchsize = 2
test_df = cudf.DataFrame(
    {
        "domain": [
            "studytour.com.tw",
            "cnn.com",
            "bakercityherald.com",
            "bankmobile.com",
        ],
        "type": [1, 1, 0, 1],
    }
)
expected_part_df1 = cudf.DataFrame(
    {
        "domain": [
            "studytour.com.tw",
            "cnn.com",
        ],
        "type": [1, 1],
    }
)

expected_part_df2 = cudf.DataFrame(
    {
        "domain": [
            "bakercityherald.com",
            "bankmobile.com",
        ],
        "type": [0, 1],
    }
)
dataset = Dataset(test_df)
dataloader = DataLoader(dataset, batchsize=test_batchsize)


def test_get_chunks():
    df_parts = []
    for df_part in dataloader.get_chunks():
        df_parts.append(df_part)
    assert len(df_parts) == 2
    assert df_parts[0].reset_index(drop=True).equals(expected_part_df1)
    assert df_parts[1].reset_index(drop=True).equals(expected_part_df2)
