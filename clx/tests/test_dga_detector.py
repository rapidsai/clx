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

import os
import cudf
import pytest
import torch
import torch.nn as nn
from cudf import DataFrame
from mockito import when, mock, verify
from clx.analytics.detector_utils import DetectorDataset
from clx.analytics.dga_detector import DGADetector
from clx.analytics.model.rnn_classifier import RNNClassifier

test_dataset_len = 4
test_df1 = cudf.DataFrame(
    {
        "domain": ["studytour.com.tw", "cnn.com"],
        "type": [1, 1],
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
        "len": [16, 7],
    }
)

test_df2 = cudf.DataFrame(
    {
        "domain": ["bakercityherald.com", "bankmobile.com"],
        "type": [0, 1],
        0: [98, 98],
        1: [97, 97],
        2: [107, 110],
        3: [101, 107],
        4: [114, 109],
        5: [99, 111],
        6: [105, 98],
        7: [116, 105],
        8: [121, 108],
        9: [104, 101],
        10: [101, 46],
        11: [114, 99],
        12: [97, 111],
        13: [108, 109],
        14: [100, 0],
        15: [46, 0],
        16: [99, 0],
        17: [111, 0],
        18: [109, 0],
        "len": [19, 14],
    }
)

test_partitioned_dfs = [test_df1, test_df2]
dataset = DetectorDataset(test_partitioned_dfs, test_dataset_len)
model_filepath = "%s/input/rnn_classifier_2020-01-20_22_50_26.pth" % os.path.dirname(
    os.path.realpath(__file__)
)


def test_load_model():
    dd = DGADetector()
    dd.load_model(model_filepath)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        assert isinstance(dd.model, nn.DataParallel)
    else:
        assert isinstance(dd.model, RNNClassifier)


def test_predict():
    dd = DGADetector()
    test_domains = cudf.Series(["nvidia.com", "dfsdfsdf"])
    dd.load_model(model_filepath)
    actual_output = dd.predict(test_domains)
    expected_output = cudf.Series([1, 0])
    assert actual_output.equals(actual_output)


def test_train_model():
    dd = DGADetector()
    dd.init_model()
    total_loss = dd.train_model(dataset)
    assert isinstance(total_loss, (int, float))


def test_evaluate_model():
    dd = DGADetector()
    dd.init_model()
    accuracy = dd.evaluate_model(dataset)
    assert isinstance(accuracy, (int, float))
