import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

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
    def load_model(self, file_path):
        pass

    @abstractmethod
    def save_model(self, file_path):
        pass

    @abstractmethod
    def train_model(self, epoch, train_dataset):
        pass

    @abstractmethod
    def predict(self, epoch, train_dataset):
        pass

    def _load_model(self, model):
        model.eval()
        self.leverage_model(model)

    def _save_model(self, checkpoint, file_path):
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

    def _get_unwrapped_model(self):
        if GPU_COUNT > 1:
            model = self.model.module
        else:
            model = self.model
        return model
