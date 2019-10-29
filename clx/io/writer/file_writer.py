from abc import ABC, abstractmethod
from clx.io.writer.writer import Writer


class FileWriter(Writer):

    @property
    def config(self):
        return self._config

    @abstractmethod
    def write_data(self):
        pass
