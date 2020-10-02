import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from clx.analytics.model.rnn_classifier import RNNClassifier

log = logging.getLogger(__name__)

GPU_COUNT = torch.cuda.device_count()


class Detector(ABC):
    def __init__(self, lr=0.001):
        self.lr = lr
        self.__model = None
        self.__optimizer = None
        self.__criterion = nn.CrossEntropyLoss()

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def criterion(self):
        return self.__criterion

    @abstractmethod
    def init_model(self, char_vocab, hidden_size, n_domain_type, n_layers):
        pass

    @abstractmethod
    def train_model(self, epoch, train_dataset):
        pass

    @abstractmethod
    def predict(self, epoch, train_dataset):
        pass

    def load_model(self, file_path):
        """ This function load already saved model and sets cuda parameters.

        :param file_path: File path of a model to loaded.
        :type file_path: string
        """
        model_dict = torch.load(file_path)
        model = RNNClassifier(
            model_dict["input_size"],
            model_dict["hidden_size"],
            model_dict["output_size"],
            model_dict["n_layers"],
        )
        model.load_state_dict(model_dict["state_dict"])
        model.eval()
        self.leverage_model(model)

    def save_model(self, file_path):
        """ This function saves model to given location.

        :param file_path: File path to save model.
        :type file_path: string
        """
        if GPU_COUNT > 1:
            rnn_model = self.model.module
        else:
            rnn_model = self.model
        checkpoint = {
            "state_dict": rnn_model.state_dict(),
            "input_size": rnn_model.input_size,
            "hidden_size": rnn_model.hidden_size,
            "n_layers": rnn_model.n_layers,
            "output_size": rnn_model.output_size,
        }
        torch.save(checkpoint, file_path)

    def __set_parallelism(self):
        if GPU_COUNT > 1:
            log.info("%s GPUs!" % (GPU_COUNT))
            self.__model = nn.DataParallel(self.model)
            self.__set_model2cuda()
        else:
            self.__set_model2cuda()

    def __set_optimizer(self):
        self.__optimizer = torch.optim.RMSprop(
            self.model.parameters(), self.lr, weight_decay=0.0
        )

    def __set_model2cuda(self):
        if torch.cuda.is_available():
            log.info("Setting cuda")
            self.model.cuda()

    def leverage_model(self, model):
        """This function leverages model by setting parallelism parameters.

        :param model: Model instance.
        :type model: RNNClassifier
        """
        self.__model = model
        self.__set_parallelism()
        self.__set_optimizer()
        