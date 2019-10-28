import dask, dask_cudf
import logging
from clx.io.reader.file_reader import FileReader

log = logging.getLogger(__name__)

class DaskFileSystemReader(FileReader):
    def __init__(self, config):
        self._config = config
        self._has_data = True

    def fetch_data(self):
        df = None
        input_format = self.config["input_format"].lower()
        filepath = self.config["input_path"].lower()
        del self.config["type"]
        del self.config["input_format"]
        del self.config["input_path"]

        if "parquet" == input_format:
            df = dask_cudf.read_parquet(filepath, **self.config)
        elif "orc" == input_format:
            df = dask_cudf.read_orc(filepath, engine="cudf")
        else:
            df = dask_cudf.read_csv(filepath, **self.config)

        self.has_data = False
        return df

    def close(self):
       log.info("Closed dask_fs reader")
