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
from clx.utils.data.dataloader import DataLoader
from clx.analytics.dga_detector import DGADetector
from clx.analytics.dga_dataset import DGADataset
from clx.analytics.model.rnn_classifier import RNNClassifier
import torch
from os import path

gpu_count = torch.cuda.device_count()

train_data = cudf.Series(
    ["studytour.com.tw", "cnn.com", "bakercityherald.com", "bankmobile.com"]
)
labels = cudf.Series([1, 1, 0, 1])

test_df = cudf.DataFrame({"domain": ["cnn.com", "bakercityherald.com"], "type": [1, 0]})

dd = DGADetector()
dd.init_model()


def test_train_model():
    if torch.cuda.is_available():
        # train model
        dd.train_model(train_data, labels, batch_size=2)
        if gpu_count > 1:
            assert isinstance(dd.model.module, RNNClassifier)
        else:
            assert isinstance(dd.model, RNNClassifier)


def test_evaluate_model():
    if torch.cuda.is_available():
        dataset = DGADataset(test_df)
        dataloader = DataLoader(dataset, batchsize=2)
        # evaluate model
        accuracy = dd.evaluate_model(dataloader)
        assert isinstance(accuracy, (int, float))


def test_predict():
    if torch.cuda.is_available():
        test_domains = cudf.Series(["nvidia.com", "dfsdfsdf"])
        # predict
        preds = dd.predict(test_domains)
        assert len(preds) == 2
        assert preds.dtype == int
        assert isinstance(preds, cudf.core.series.Series)


def test2_predict():
    if torch.cuda.is_available():
        test_domains = cudf.Series(["nvidia.com", "dfsdfsdf"])
        # predict
        preds = dd.predict(test_domains, probability=True)
        assert len(preds) == 2
        assert preds.dtype == float
        assert isinstance(preds, cudf.core.series.Series)


def test_save_model(tmpdir):
    if torch.cuda.is_available():
        # save model
        dd.save_model(str(tmpdir.join("clx_dga.mdl")))
        assert path.exists(str(tmpdir.join("clx_dga.mdl")))


def test_load_model(tmpdir):
    if torch.cuda.is_available():
        # save model
        dd.save_model(str(tmpdir.join("clx_dga.mdl")))
        assert path.exists(str(tmpdir.join("clx_dga.mdl")))
        # load model
        dd2 = DGADetector()
        dd2.init_model()
        dd2.load_model(str(tmpdir.join("clx_dga.mdl")))
        if gpu_count > 1:
            assert isinstance(dd2.model.module, RNNClassifier)
        else:
            assert isinstance(dd2.model, RNNClassifier)
