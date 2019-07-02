import pytest
import os
from rapidscyber.ml.model.rnn_classifier import RNNClassifier
from rapidscyber.ml.manager.rnn_classifier_builder import RNNClassifierBuilder

char_vocab = 128
hidden_size = 100
n_domain_type = 2
n_layers = 3
model_filepath = "%s/input/rnn_classifier_2019-07-01_17_01_39.pth" % os.path.dirname(
    os.path.realpath(__file__)
)
expected_module = "rapidscyber.ml.model.rnn_classifier"


@pytest.mark.parametrize("char_vocab", [char_vocab])
@pytest.mark.parametrize("hidden_size", [hidden_size])
@pytest.mark.parametrize("n_domain_type", [n_domain_type])
@pytest.mark.parametrize("n_layers", [n_layers])
@pytest.mark.parametrize("expected_module", [expected_module])
def test_build_model(char_vocab, hidden_size, n_domain_type, n_layers, expected_module):
    builder = RNNClassifierBuilder(char_vocab, hidden_size, n_domain_type, n_layers)
    actual_module = builder.build_model().module.__module__
    assert actual_module == expected_module


@pytest.mark.parametrize("model_filepath", [model_filepath])
@pytest.mark.parametrize("expected_module", [expected_module])
def test_load_model(model_filepath, expected_module):
    nn_wrapped_model = RNNClassifierBuilder.load_model(model_filepath)
    actual_model = nn_wrapped_model.module.__module__
    assert actual_model == expected_module
