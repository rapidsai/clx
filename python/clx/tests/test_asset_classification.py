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

from faker import Faker
from random import shuffle, seed
from faker.providers.person.en import Provider
import random
import pandas as pd

fake = Faker()
dict_items = {'1': 24, '2': 4, '3': 9, '4': 26, '5': 3, '6': 9, '7': 37, '8': 8, '9': 4, '10': 11, '11': 5, '12': 8, '13': 1408, '14': 8990, '15': 5, '16': 8, '17': 4, '18': 4}
computer_names13 = list(set(Provider.first_names))
seed(4)
shuffle(computer_names13)
computer_names13 = computer_names13 + computer_names13[:9000 - len(set(set(Provider.first_names)))]
computer_names14 = list(set(Provider.first_names)) + list(set(Provider.first_names))[:9000 - len(set(Provider.first_names))]
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
column11 = [random.randint(1, 5) for _ in range(9000)]
column12 = [random.randint(1, 8) for _ in range(9000)]
column15 = [random.randint(1, 5) for _ in range(9000)]
column16 = [random.randint(1, 6) for _ in range(9000)]
column17 = [random.randint(1, 8) for _ in range(9000)]
column18 = [random.randint(1, 4) for _ in range(9000)]
label = [random.randint(1, 6) for _ in range(9000)]

train_pd = pd.DataFrame(list(zip(column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11, column12, computer_names13, computer_names14, column15, column16, column17, column18, label)), columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
train_gdf = cudf.from_pandas(train_pd)


@pytest.mark.parametrize("train_gdf", [train_gdf])
def test_all(tmpdir, train_gdf):
    if torch.cuda.is_available():
        # train
        label_col = "19"
        batch_size = 1000
        epochs = 15
        ac = AssetClassification()
        train_gdf = ac.categorize_columns(train_gdf)
        ac.train_model(train_gdf, label_col, batch_size, epochs)
        assert isinstance(ac._model, clx.analytics.model.tabular_model.TabularModel)

        # predict
        test_gdf = train_gdf.head()
        test_gdf.drop_column("19")
        preds = ac.predict(test_gdf)
        assert isinstance(preds, cudf.core.series.Series)

        # save model
        ac.save_model(str(tmpdir.join("clx_ac.mdl")))
        assert path.exists(str(tmpdir.join("clx_ac.mdl")))

        # load model
        ac = AssetClassification()
        ac.load_model(str(tmpdir.join("clx_ac.mdl")))
        assert isinstance(ac._model, clx.analytics.model.tabular_model.TabularModel)
