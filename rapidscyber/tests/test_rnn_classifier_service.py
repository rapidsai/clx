import os
import pytest
import torch
from mockito import when, mock, verify
from rapidscyber.ml.dataset_api.dga_dataset import DGADataset
from rapidscyber.ml.manager.rnn_classifier_service import RNNClassifierService
from torch.utils.data import DataLoader


input_rows = [
    ["cnn.com", "1"],
    ["studytour.com.tw", "1"],
    ["bankmobile.com", "1"],
    ["bakercityherald.com", "0"],
]
dataset = DGADataset()
dataset.set_attributes(input_rows)
data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

model_filepath = "%s/input/rnn_classifier_2019-07-03_03_46_53.pth" % os.path.dirname(os.path.realpath(__file__))

model = mock()
optimizer = mock()
criterion = mock()


@pytest.mark.parametrize("model", [model])
@pytest.mark.parametrize("optimizer", [optimizer])
@pytest.mark.parametrize("criterion", [criterion])
@pytest.mark.parametrize("data_loader", [data_loader])
def test_train_model(model, optimizer, criterion, data_loader):
    service = RNNClassifierService(
        model, optimizer, criterion, train_loader=data_loader
    )
    when(service).get_item(..., ...).thenReturn(0.5)
    service.train_model(1)
    # since given batch size is 2 and input record count is 4 it calls classifier for 2 times.
    verify(service.classifier, times=2)


@pytest.mark.parametrize("model_filepath", [model_filepath])
@pytest.mark.parametrize("optimizer", [optimizer])
@pytest.mark.parametrize("criterion", [criterion])
def test_predict(model_filepath, optimizer, criterion):
    model = torch.load(model_filepath)
    model.eval()
    if torch.cuda.is_available():
       model.cuda()
    service = RNNClassifierService(model, optimizer, criterion)
    domains = ["nvidia.com", "dfsdfsdf"]
    actual = service.predict(domains)
    expected = [1,0]
    assert actual == expected