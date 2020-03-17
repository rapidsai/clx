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
import pytest
from clx.analytics.detector_dataset import DetectorDataset

test_domains_len = 2
test_batchsize = 1
test_input_df = cudf.DataFrame(
    {"domain": ["studytour.com.tw", "cnn.com"], "type": [1, 1]}
)

expected_output_df1 = cudf.DataFrame(
    {
        "domain": ["studytour.com.tw"],
        "type": [1],
        0: [115],
        1: [116],
        2: [117],
        3: [100],
        4: [121],
        5: [116],
        6: [111],
        7: [117],
        8: [114],
        9: [46],
        10: [99],
        11: [111],
        12: [109],
        13: [46],
        14: [116],
        15: [119],
        "len": [16],
    }
)

expected_output_df2 = cudf.DataFrame(
    {
        "domain": ["cnn.com"],
        "type": [1],
        0: [99],
        1: [110],
        2: [110],
        3: [46],
        4: [99],
        5: [111],
        6: [109],
        7: [0],
        8: [0],
        9: [0],
        10: [0],
        11: [0],
        12: [0],
        13: [0],
        14: [0],
        15: [0],
        "len": [7],
    }
)


def test_detector_dataset():
    dataset = DetectorDataset(test_input_df, test_batchsize)
    assert len(dataset.partitioned_dfs) == 2
    assert dataset.partitioned_dfs[0].equals(expected_output_df1)
    assert (
        dataset.partitioned_dfs[1]
        .reset_index(drop=True)["len"]
        .equals(expected_output_df2.reset_index(drop=True)["len"])
    )
