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
from clx.analytics.detector_dataset import DetectorDataset
from clx.analytics.dga_detector import DGADetector
from clx.analytics.model.rnn_classifier import RNNClassifier

test_dataset_len = 4
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
dataset = DetectorDataset(test_df, test_batchsize)
model_filepath = "%s/input/rnn_classifier_2020-06-08_20_48_03.pth" % os.path.dirname(
    os.path.realpath(__file__)
)


def test_load_model():
    dd = DGADetector()
    dd.load_model(model_filepath)
    assert isinstance(dd.model, RNNClassifier)


def test_predict():
    dd = DGADetector()
    test_domains = cudf.Series(["nvidia.com", "dfsdfsdf"])
    dd.load_model(model_filepath)
    actual_output = dd.predict(test_domains)
    expected_output = cudf.Series([1, 0])
    assert actual_output.equals(expected_output)


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
