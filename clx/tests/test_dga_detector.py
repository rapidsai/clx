import os
import cudf
import torch
import pytest
from cudf import DataFrame
from mockito import when, mock, verify
from clx.ml.provider.dga_detector import DGADetector
from clx.ml.model.rnn_classifier import RNNClassifier
import logging

log = logging.getLogger(__name__)

test_dataset_len = 4
test_df1 = cudf.DataFrame(
    [
        ("domain", ["studytour.com.tw", "cnn.com"]),
        ("type", [1, 1]),
        (0, [115, 99]),
        (1, [116, 110]),
        (2, [117, 110]),
        (3, [100, 46]),
        (4, [121, 99]),
        (5, [116, 111]),
        (6, [111, 109]),
        (7, [117, 0]),
        (8, [114, 0]),
        (9, [46, 0]),
        (10, [99, 0]),
        (11, [111, 0]),
        (12, [109, 0]),
        (13, [46, 0]),
        (14, [116, 0]),
        (15, [119, 0]),
        ("len", [16, 7]),
    ]
)

test_df2 = cudf.DataFrame(
    [
        ("domain", ["bakercityherald.com", "bankmobile.com"]),
        ("type", [0, 1]),
        (0, [98, 98]),
        (1, [97, 97]),
        (2, [107, 110]),
        (3, [101, 107]),
        (4, [114, 109]),
        (5, [99, 111]),
        (6, [105, 98]),
        (7, [116, 105]),
        (8, [121, 108]),
        (9, [104, 101]),
        (10, [101, 46]),
        (11, [114, 99]),
        (12, [97, 111]),
        (13, [108, 109]),
        (14, [100, 0]),
        (15, [46, 0]),
        (16, [99, 0]),
        (17, [111, 0]),
        (18, [109, 0]),
        ("len", [19, 14]),
    ]
)

test_partitioned_dfs = [test_df1, test_df2]

model_filepath = "%s/input/rnn_classifier_2019-10-21_22_40_57.pth" % os.path.dirname(
    os.path.realpath(__file__)
)


def test_load_model():
    log.info('>>>>>>>>>>>device count %s <<<<<<<<' %(torch.cuda.device_count()))
    print('>>>>>>>>>>>device count %s <<<<<<<<' %(torch.cuda.device_count()))
    print(">>>>>>>>> current device %s<<<<<<<<<" %(torch.cuda.current_device()))
    log.info(">>>>>>>>> current device %s<<<<<<<<<" %(torch.cuda.current_device()))
    dd = DGADetector()
    dd.load_model(model_filepath)
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
    total_loss = dd.train_model(test_partitioned_dfs, test_dataset_len)
    assert isinstance(total_loss, (int, float))


def test_evaluate_model():
    dd = DGADetector()
    dd.init_model()
    accuracy = dd.evaluate_model(test_partitioned_dfs, test_dataset_len)
    assert isinstance(accuracy, (int, float))
