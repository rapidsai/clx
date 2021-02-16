import cudf
import pytest

import clx.features

df = cudf.DataFrame(
    {
        "time": [1, 2, 3, 4, 5, 6, 7],
        "user": ["u1", "u2", "u3", "u1", "u1", "u2", "u1"],
        "computer": ["c1", "c2", "c3", "c1", "c2", "c3", "c1"],
    }
)


def test_binary_features():
    actual = clx.features.binary(df, "user", "computer")
    expected = cudf.DataFrame(
        {"user": ["u1", "u2", "u3"], "c1": [1, 0, 0], "c2": [1, 1, 0], "c3": [0, 1, 1]}
    )
    expected = expected.set_index("user")
    expected["c1"] = expected["c1"].astype("int32")
    expected["c2"] = expected["c2"].astype("int32")
    expected["c3"] = expected["c3"].astype("int32")
    expected.columns = cudf.MultiIndex(
        names=[None, "computer"],
        codes=[[0, 0, 0], [0, 1, 2]],
        levels=[["time"], ["c1", "c2", "c3"]],
    )
    assert expected.equals(actual)


def test_binary_exception():
    with pytest.raises(Exception):
        clx.features.binary(df, "user", "a")


def test_frequency_features():
    actual = clx.features.frequency(df, "user", "computer")
    expected = cudf.DataFrame(
        {
            "user": ["u1", "u2", "u3"],
            "c1": [0.75, 0.00, 0.00],
            "c2": [0.25, 0.50, 0.0],
            "c3": [0.0, 0.5, 1.0],
        }
    )
    expected = expected.set_index("user")
    expected.columns = cudf.MultiIndex(
        names=[None, "computer"],
        codes=[[0, 0, 0], [0, 1, 2]],
        levels=[["time"], ["c1", "c2", "c3"]],
    )
    assert expected.equals(actual)


def test_frequency_exception():
    with pytest.raises(Exception):
        clx.features.frequency(df, "a", "computer")
