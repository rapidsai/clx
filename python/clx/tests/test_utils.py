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
from clx.utils.data import utils

test_domains_len = 2
test_input_df = cudf.DataFrame(
    {"domain": ["studytour.com.tw", "cnn.com"], "type": [1, 1]}
)
expected_output_df = cudf.DataFrame(
    {
        0: [115, 99],
        1: [116, 110],
        2: [117, 110],
        3: [100, 46],
        4: [121, 99],
        5: [116, 111],
        6: [111, 109],
        7: [117, 0],
        8: [114, 0],
        9: [46, 0],
        10: [99, 0],
        11: [111, 0],
        12: [109, 0],
        13: [46, 0],
        14: [116, 0],
        15: [119, 0],
        "len": [16, 7]
    },
    dtype="int32"
)
expected_output_df["type"] = [1, 1]
expected_output_df["domain"] = ["studytour.com.tw", "cnn.com"]


def test_str2ascii():
    actual_output_df = utils.str2ascii(test_input_df, 'domain')
    assert actual_output_df.equals(expected_output_df)
