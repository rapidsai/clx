from clx.io.factory.abstract_factory import AbstractFactory
from clx.io.reader.fs_reader import DaskFileSystemReader
from clx.io.writer.fs_writer import FileSystemWriter


class DaskFileSystemFactory(AbstractFactory):
    def __init__(self, config):
        self._config = config

    def get_reader(self):
        return DaskFileSystemReader(self.config)

    def get_writer(self):
        raise NotImplementedError
