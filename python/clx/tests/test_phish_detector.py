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
from clx.analytics.phishing_detector import PhishingDetector
from cuml.preprocessing.model_selection import train_test_split
import torch
import s3fs
import transformers
from os import path

S3_BASE_PATH = "rapidsai-data/cyber/clx"
EMAILS_TSV = "spam_assassin_hardham_200_20021010.tsv"

phish_detect = PhishingDetector()
if torch.cuda.is_available():
    phish_detect.init_model()


def test_train_model(tmpdir):
    if torch.cuda.is_available():
        fname = str(tmpdir.mkdir("tmp_test_phish_detector").join(EMAILS_TSV))
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + EMAILS_TSV, fname)
        emails_gdf = cudf.read_csv(fname, delimiter='\t', header=None, names=['label', 'email'])
        X_train, X_test, y_train, y_test = train_test_split(emails_gdf, 'label', train_size=0.8, random_state=10)
        phish_detect.train_model(X_train, y_train, epochs=1)
        assert isinstance(phish_detect._model, transformers.modeling_bert.BertForSequenceClassification)


def test_evaluate_model():
    if torch.cuda.is_available():
        X_test = cudf.DataFrame({"email": ["email 1", "email 2"]})
        y_test = cudf.Series([0, 0])
        accuracy = phish_detect.evaluate_model(X_test, y_test)
        assert accuracy >= 0.0 and accuracy <= 1.0


def test_predict():
    if torch.cuda.is_available():
        X_test = cudf.DataFrame({"email": ["email 1", "email 2"]})
        preds = phish_detect.predict(X_test)
        assert preds.isin([0, 1]).equals(cudf.Series([True, True]))


def test_save_model(tmpdir):
    if torch.cuda.is_available():
        phish_detect.save_model(tmpdir)
        assert path.exists(str(tmpdir.join("config.json")))
        assert path.exists(str(tmpdir.join("pytorch_model.bin")))
