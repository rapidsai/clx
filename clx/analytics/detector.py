import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from clx.analytics.model.rnn_classifier import RNNClassifier

log = logging.getLogger(__name__)


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
        model = torch.load(file_path)
        model.eval()
        self.__model = model
        self.__set_model2cuda()
        self.__set_optimizer()

    def save_model(self, file_path):
        torch.save(self.model, file_path)

    def __set_parallelism(self):
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            log.info("%s GPUs!" % (gpu_count))
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
        self.__model = model
        self.__set_parallelism()
        self.__set_optimizer()
