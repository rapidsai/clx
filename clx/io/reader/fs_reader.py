import cudf
import logging
from clx.io.reader.file_reader import FileReader

log = logging.getLogger(__name__)

class FileSystemReader(FileReader):
    def __init__(self, config):
        self._config = config
        self._has_data = True

    
    def fetch_data(self):
        df = None
        input_format = self.config["input_format"].lower()
        filepath = self.config["filepath"].lower()
        del self.config["type"]
        del self.config["input_format"]
        del self.config["filepath"]

        if "parquet" == input_format:
            df = cudf.read_parquet(filepath, **self.config)
        elif "orc" == input_format:
            df = cudf.read_orc(filepath, engine="cudf")
        else:
            df = cudf.read_csv(filepath, **self.config)
        
        self.has_data = False
        return df

    def close(self):
       log.info("Closed fs reader")
