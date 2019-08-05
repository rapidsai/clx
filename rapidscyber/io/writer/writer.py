from abc import ABC, abstractmethod

class Writer(ABC):
    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def write_data(self):
        pass
