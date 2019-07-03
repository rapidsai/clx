from rapidscyber.io.factory.abstract_factory import AbstractFactory
from rapidscyber.io.reader.fs_reader import FileSystemReader
from rapidscyber.io.writer.fs_writer import FileSystemWriter


class FileSystemFactory(AbstractFactory):
    def __init__(self, config):
        self._config = config

    def get_reader(self):
        return FileSystemReader(self.config)

    def get_writer(self):
        return FileSystemWriter(
            self.config["output_path"], self.config["output_format"]
        )
