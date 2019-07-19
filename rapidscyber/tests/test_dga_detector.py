import os
import pytest
from mockito import when, mock, verify
from rapidscyber.ml.dataset_api.dga_dataset import DGADataset
from rapidscyber.ml.provider.dga_detector import DGADetector
from torch.utils.data import DataLoader


input_rows = [
    ["cnn.com", "1"],
    ["studytour.com.tw", "1"],
    ["bankmobile.com", "1"],
    ["bakercityherald.com", "0"],
]
dataset = DGADataset(input_rows)
data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

model_filepath = "%s/input/rnn_classifier_2019-07-18_05_13_12.pth" % os.path.dirname(
    os.path.realpath(__file__)
)
dd = DGADetector()


@pytest.mark.parametrize("dd", [dd])
@pytest.mark.parametrize("model_filepath", [model_filepath])
def test_load_model(dd, model_filepath):
    model = mock()
    when(dd).load_model(model_filepath).thenReturn(model)
    dd.load_model(model_filepath)
    verify(dd, times=1).model


@pytest.mark.parametrize("dd", [dd])
def test_predict(dd):
    domains = ["nvidia.com", "dfsdfsdf"]
    when(dd).predict(domains).thenReturn([1, 0])
    dd.predict(domains)
    verify(dd, times=2).model


@pytest.mark.parametrize("dd", [dd])
@pytest.mark.parametrize("data_loader", [data_loader])
def test_train_model(dd, data_loader):
    model = mock()
    when(dd).train_model(data_loader).thenReturn(1)
    dd.train_model(data_loader)
    verify(dd, times=2).model
