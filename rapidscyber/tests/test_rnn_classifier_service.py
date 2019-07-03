import os
import pytest
import torch
from mockito import when, mock, verify
from rapidscyber.ml.manager.rnn_classifier_service import RNNClassifierService


model_filepath = "%s/input/rnn_classifier_2019-07-03_03_46_53.pth" % os.path.dirname(os.path.realpath(__file__))
model = mock()
optimizer = mock()
criterion = mock()


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