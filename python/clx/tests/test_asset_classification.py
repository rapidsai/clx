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

import pytest
import cudf
import clx
from clx.analytics.asset_classification import AssetClassification
import torch
from os import path

import random
import pandas as pd

column1 = [random.randint(1, 24) for _ in range(9000)]
column2 = [random.randint(1, 4) for _ in range(9000)]
column3 = [random.randint(1, 9) for _ in range(9000)]
column4 = [random.randint(1, 26) for _ in range(9000)]
column5 = [random.randint(1, 3) for _ in range(9000)]
column6 = [random.randint(1, 9) for _ in range(9000)]
column7 = [random.randint(1, 37) for _ in range(9000)]
column8 = [random.randint(1, 8) for _ in range(9000)]
column9 = [random.randint(1, 4) for _ in range(9000)]
column10 = [random.randint(1, 11) for _ in range(9000)]
label = [random.randint(0, 6) for _ in range(9000)]

train_pd = pd.DataFrame(list(zip(column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, label)), columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "label"])
train_gdf = cudf.from_pandas(train_pd)

batch_size = 6
epochs = 15


@pytest.mark.parametrize("train_gdf", [train_gdf])
def test_train_model_mixed_cat_cont(tmpdir, train_gdf):
    train_gdf = train_gdf.copy()
    cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8"]
    cont_cols = ["9", "10"]
    train_gdf[cont_cols] = normalize_conts(train_gdf[cont_cols])
    ac = AssetClassification()
    ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
    if torch.cuda.is_available():
        assert isinstance(ac._model, clx.analytics.model.tabular_model.TabularModel)


@pytest.mark.parametrize("train_gdf", [train_gdf])
def test_train_model_all_cat(tmpdir, train_gdf):
    train_gdf = train_gdf.copy()
    cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    cont_cols = []
    ac = AssetClassification()
    ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
    if torch.cuda.is_available():
        assert isinstance(ac._model, clx.analytics.model.tabular_model.TabularModel)


@pytest.mark.parametrize("train_gdf", [train_gdf])
def test_train_model_all_cont(tmpdir, train_gdf):
    train_gdf = train_gdf.copy()
    cont_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    cat_cols = []
    train_gdf[cont_cols] = normalize_conts(train_gdf[cont_cols])
    ac = AssetClassification()
    ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
    if torch.cuda.is_available():
        assert isinstance(ac._model, clx.analytics.model.tabular_model.TabularModel)


@pytest.mark.parametrize("train_gdf", [train_gdf])
def test_predict(tmpdir, train_gdf):
    if torch.cuda.is_available():
        cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        cont_cols = []
        ac = AssetClassification()
        ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
        # predict
        test_gdf = train_gdf.head()
        test_gdf.drop("label", axis=1)
        preds = ac.predict(test_gdf, cat_cols, cont_cols)
        assert isinstance(preds, cudf.core.series.Series)
        assert len(preds) == len(test_gdf)
        assert preds.dtype == int


def test_save_model(tmpdir):
    if torch.cuda.is_available():
        cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        cont_cols = []
        ac = AssetClassification()
        ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
        # save model
        ac.save_model(str(tmpdir.join("clx_ac.mdl")))
        assert path.exists(str(tmpdir.join("clx_ac.mdl")))


def test_load_model(tmpdir):
    if torch.cuda.is_available():
        cat_cols = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        cont_cols = []
        ac = AssetClassification()
        ac.train_model(train_gdf, cat_cols, cont_cols, "label", batch_size, epochs)
        # save model
        ac.save_model(str(tmpdir.join("clx_ac.mdl")))
        assert path.exists(str(tmpdir.join("clx_ac.mdl")))
        # load model
        ac2 = AssetClassification()
        ac2.load_model(str(tmpdir.join("clx_ac.mdl")))
        assert isinstance(ac2._model, clx.analytics.model.tabular_model.TabularModel)


def normalize_conts(gdf):
    means, stds = (gdf.mean(0), gdf.std(ddof=0))
    gdf = (gdf - means) / stds
    return gdf
