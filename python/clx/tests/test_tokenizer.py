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
import pytest
import torch
from clx.analytics import tokenizer

input_sentence = "Key length indicates the length of the generated session key."


def get_expected():

    tokens = torch.tensor([[3145, 3091, 7127, 1996, 3091, 1997, 1996, 7013, 5219, 3145, 1012, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0]], device="cuda")

    masks = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda')

    metadata = torch.tensor([[0, 0, 10]], device="cuda")

    return tokens, masks, metadata


@pytest.mark.parametrize("input_sentence", [(input_sentence)])
def test_tokenize_file(tmpdir, input_sentence):
    if torch.cuda.is_available():
        expected_tokens, expected_masks, expected_metadata = get_expected()
        fname = tmpdir.mkdir("tmp_test_tokenizer").join("test1.txt")
        fname.write(input_sentence)

        assert fname.read() == input_sentence

        actual_tokens, actual_masks, actual_metadata = tokenizer.tokenize_file(str(fname))

        assert actual_tokens.equal(expected_tokens)
        assert actual_masks.equal(expected_masks)
        assert actual_metadata.equal(expected_metadata)


@pytest.mark.parametrize("input_sentence", [(input_sentence)])
def test_tokenize_df(tmpdir, input_sentence):
    if torch.cuda.is_available():
        expected_tokens, expected_masks, expected_metadata = get_expected()
        fname = tmpdir.mkdir("tmp_test_tokenizer").join("test1.txt")
        fname.write(input_sentence)

        assert fname.read() == input_sentence

        df = cudf.read_csv(fname, header=None)
        actual_tokens, actual_masks, actual_metadata = tokenizer.tokenize_df(df)

        assert actual_tokens.equal(expected_tokens)
        assert actual_masks.equal(expected_masks)
        assert actual_metadata.equal(expected_metadata)
