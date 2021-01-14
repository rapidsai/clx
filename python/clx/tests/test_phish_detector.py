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
import random
from os import path

import cudf
import torch
import transformers
from cuml.preprocessing.model_selection import train_test_split
from faker import Faker

from clx.analytics.phishing_detector import PhishingDetector

phish_detect = PhishingDetector()
if torch.cuda.is_available():
    phish_detect.init_model()


def test_train_model():
    if torch.cuda.is_available():
        fake = Faker()
        email_col = [fake.text() for _ in range(200)]
        label_col = [random.randint(0, 1) for _ in range(200)]
        emails_gdf = cudf.DataFrame(list(zip(email_col, label_col)), columns=["email", "label"])
        X_train, X_test, y_train, y_test = train_test_split(
            emails_gdf, "label", train_size=0.8, random_state=10
        )
        phish_detect.train_model(
            X_train,
            y_train,
            learning_rate=3e-5,
            max_seq_len=128,
            batch_size=32,
            epochs=1,
        )
        assert isinstance(
            phish_detect._model.module,
            transformers.models.bert.modeling_bert.BertForSequenceClassification
        )


def test_evaluate_model():
    if torch.cuda.is_available():
        X_test = cudf.DataFrame({"email": ["email 1", "email 2"]})
        y_test = cudf.Series([0, 0])
        accuracy = phish_detect.evaluate_model(
            X_test, y_test, max_seq_len=128, batch_size=32
        )
        assert accuracy >= 0.0 and accuracy <= 1.0


def test_predict():
    if torch.cuda.is_available():
        X_test = cudf.Series(["email 1", "email 2"])
        preds = phish_detect.predict(X_test, max_seq_len=128)
        assert preds[0].isin([False, True]).equals(cudf.Series([True, True]))


def test_save_model(tmpdir):
    if torch.cuda.is_available():
        phish_detect.save_model(tmpdir)
        assert path.exists(str(tmpdir.join("config.json")))
        assert path.exists(str(tmpdir.join("pytorch_model.bin")))
