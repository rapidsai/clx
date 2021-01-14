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
import cupy
import pandas as pd
import numpy as np
import torch
import s3fs
import transformers
from clx.analytics.cybert import Cybert

S3_BASE_PATH = "models.huggingface.co/bert/raykallen/cybert_apache_parser"
CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "pytorch_model.bin"

fs = s3fs.S3FileSystem(anon=True)
fs.get(S3_BASE_PATH + "/" + MODEL_FILENAME, MODEL_FILENAME)
fs.get(S3_BASE_PATH + "/" + CONFIG_FILENAME, CONFIG_FILENAME)

cyparse = Cybert()

input_logs = cudf.Series(['109.169.248.247 - -',
                          'POST /administrator/index.php HTTP/1.1 200 4494'])


def get_expected_preprocess():
    tokens = torch.tensor(
        [[11523, 119, 20065, 119, 27672, 119, 26049, 118, 118, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [153, 9025, 1942, 120, 11065, 120, 7448, 119, 185, 16194, 145, 20174,
          2101, 120, 122, 119, 122, 2363, 3140, 1580, 1527, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0]], device='cuda:0'
    )

    masks = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0'
    )

    metadata = cupy.array([[0, 0, 8], [1, 0, 20]], dtype='uint32')
    return tokens, masks, metadata


def get_expected_inference():
    expected_parsed_df = pd.DataFrame({
        'remote_host': ['109.169.248.247', np.NaN],
        'other': ['-', np.NaN],
        'request_method': [np.NaN, 'POST'],
        'request_url': [np.NaN, "/administrator/index.php"],
        'request_http_ver': [np.NaN, 'HTTP/1.1'],
        'status': [np.NaN, '200'],
        'response_bytes_clf': [np.NaN, '449']
    })

    expected_confidence_df = pd.DataFrame({
        'remote_host': [0.999628, np.NaN], 'other': [0.999579, np.NaN],
        'request_method': [np.NaN, 0.99822], 'request_url': [np.NaN, 0.999629],
        'request_http_ver': [np.NaN, 0.999936], 'status': [np.NaN, 0.999866],
        'response_bytes_clf': [np.NaN, 0.999751]
    })
    return expected_parsed_df, expected_confidence_df


def test_load_model():
    cyparse.load_model(MODEL_FILENAME, CONFIG_FILENAME)
    assert isinstance(cyparse._label_map, dict)
    assert isinstance(cyparse._model.module,
                      transformers.models.bert.modeling_bert.BertForTokenClassification)


def test_preprocess():
    expected_tokens, expected_masks, expected_metadata = get_expected_preprocess()
    actual_tokens, actual_masks, actual_metadata = cyparse.preprocess(input_logs)
    assert actual_tokens.equal(expected_tokens)
    assert actual_masks.equal(expected_masks)
    assert cupy.equal(actual_metadata, expected_metadata).all()


def test_inference():
    if torch.cuda.is_available():
        expected_parsed_df, expected_confidence_df = get_expected_inference()
        actual_parsed_df, actual_confidence_df = cyparse.inference(input_logs)
        pd._testing.assert_frame_equal(actual_parsed_df, expected_parsed_df)
        pd._testing.assert_frame_equal(actual_confidence_df, expected_confidence_df)
