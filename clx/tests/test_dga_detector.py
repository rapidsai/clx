import os
import cudf
import pytest
from cudf import DataFrame
from mockito import when, mock, verify
from clx.ml.provider.dga_detector import DGADetector

test_dataset_len = 4
test_df1 = DataFrame()
test_df1["domain"] = ["cnn.com", "studytour.com.tw"]
test_df1["type"] = ["1", "1"]

test_df2 = DataFrame()
test_df2["domain"] = ["bankmobile.com", "bakercityherald.com"]
test_df2["type"] = ["1", "0"]
test_partitioned_dfs = [test_df1, test_df2]

model_filepath = "%s/input/rnn_classifier_2019-10-15_20_33_04.pth" % os.path.dirname(
    os.path.realpath(__file__)
)


def test_load_model():
    dd = DGADetector()
    model = mock()
    when(dd).load_model(model_filepath).thenReturn(model)
    dd.load_model(model_filepath)
    verify(dd, times=1).model


def test_predict():
    dd = DGADetector()
    test_domains = cudf.Series(["nvidia.com", "dfsdfsdf"])
    dd.load_model(model_filepath)
    actual_output = dd.predict(test_domains)
    expected_output = cudf.Series([1, 0])
    assert actual_output.equals(actual_output)


def test_train_model():
    dd = DGADetector()
    model = mock()
    when(dd).train_model(test_partitioned_dfs, test_dataset_len).thenReturn(1)
    dd.train_model(test_partitioned_dfs, test_dataset_len)
    verify(dd, times=2).model


def test_evaluate_model():
    dd = DGADetector()
    model = mock()
    when(dd).evaluate_model(test_partitioned_dfs, test_dataset_len).thenReturn(1)
    dd.evaluate_model(test_partitioned_dfs, test_dataset_len)
    verify(dd, times=2).model
