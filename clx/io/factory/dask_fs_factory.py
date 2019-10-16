from clx.io.factory.abstract_factory import AbstractFactory
from clx.io.reader.dask_fs_reader import DaskFileSystemReader


class DaskFileSystemFactory(AbstractFactory):
    def __init__(self, config):
        self._config = config

    def get_reader(self):
        return DaskFileSystemReader(self.config)

    def get_writer(self):
        raise NotImplementedError
