import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod, abstractclassmethod

log = logging.getLogger("Builder")

class Builder(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass
    
    @staticmethod
    def parallelize(model):
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            log.info("%s GPUs!" % (gpu_count))
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()
        return model
