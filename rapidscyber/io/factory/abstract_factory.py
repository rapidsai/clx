from abc import ABC, abstractmethod


class AbstractFactory(ABC):
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    @abstractmethod
    def get_reader(self):
        pass

    @abstractmethod
    def get_writer(self):
        pass
