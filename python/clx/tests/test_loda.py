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
import cupy
from clx.analytics.loda import Loda
from os import path


def test_fit():
    ld = Loda(n_random_cuts=10, n_bins=None)
    x = cupy.random.randint(0, 100, size=(200, 10))
    ld.fit(x)
    assert ld._histograms is not None
    assert isinstance(
        ld._histograms,
        cupy.ndarray
    )
    assert cupy.all(ld._histograms > 0)


def test_score():
    ld = Loda(n_random_cuts=10, n_bins=None)
    x = cupy.random.randint(0, 100, size=(200, 10))
    ld.fit(x)
    scores = ld.score(x)
    assert scores is not None
    assert isinstance(
        scores,
        cupy.ndarray
    )
    assert cupy.all(scores > 0)


def test_explain():
    ld = Loda(n_random_cuts=10, n_bins=None)
    x = cupy.random.randint(0, 100, size=(200, 10))
    ld.fit(x)
    explanation = ld.explain(x[0])
    assert explanation is not None
    assert isinstance(
        explanation,
        cupy.ndarray
    )


def test_save_model(tmpdir):
    ld = Loda(n_random_cuts=10, n_bins=None)
    x = cupy.random.randint(0, 100, size=(200, 10))
    ld.fit(x)
    ipath = path.join(tmpdir, "clx_loda")
    opath = path.join(tmpdir, "clx_loda.npz")
    ld.save_model(ipath)
    assert path.exists(opath)


def test_load_model(tmpdir):
    ld = Loda(n_random_cuts=10, n_bins=None)
    x = cupy.random.randint(0, 100, size=(200, 10))
    ld.fit(x)
    ipath = path.join(tmpdir, "clx_loda")
    opath = path.join(tmpdir, "clx_loda.npz")
    ld.save_model(ipath)
    assert path.exists(opath)

    # load model
    ld = Loda.load_model(opath)
    assert isinstance(ld, Loda)
