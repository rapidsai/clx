import torch
import logging
from rapidscyber.ml.manager.builder import Builder
from rapidscyber.ml.model.rnn_classifier import RNNClassifier

log = logging.getLogger("RNNClassifierBuilder")


class RNNClassifierBuilder(Builder):
    def __init__(
        self, char_vocab=None, hidden_size=None, n_domain_type=None, n_layers=None
    ):
        self.char_vocab = char_vocab
        self.hidden_size = hidden_size
        self.n_domain_type = n_domain_type
        self.n_layers = n_layers

    def build_model(self):
        if all([self.char_vocab, self.hidden_size, self.n_domain_type, self.n_layers]):
            model = RNNClassifier(
                self.char_vocab, self.hidden_size, self.n_domain_type, self.n_layers
            )
            model = RNNClassifierBuilder.parallelize(model)
            return model
        else:
            log.error(
                "Please check the attributes >>> char_vocab, hidden_size, n_domain_type, n_layers attributes."
            )
            raise Exception("One or more given attributes are none.")

    @staticmethod
    def load_model(model_filepath):
        model = torch.load(model_filepath)
        model.eval()
        if torch.cuda.is_available():
            log.info("Setting cuda")
            model.cuda()
        return model
