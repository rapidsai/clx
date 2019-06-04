from factory.abstract_factory import AbstractFactory
from reader.nfs_reader import NFSReader
from writer.nfs_writer import NFSWriter


class NFSFactory(AbstractFactory):
    def __init__(self, config):
        self._config = config

    def getReader(self):
        return NFSReader(self.config)

    def getWriter(self):
        return NFSWriter(self.config["output_path"], self.config["output_format"])
