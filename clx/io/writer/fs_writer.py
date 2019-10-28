import cudf
import logging
import os

from clx.io.writer.file_writer import FileWriter

log = logging.getLogger(__name__)

class FileSystemWriter(FileWriter):
 
    def __init__(self, config):
        self._config = config

    def write_data(self, df):
        output_format = self.config["output_format"]
        filepath = self.config["output_path"]
        kwargs = self.config.copy()
        del kwargs["type"]
        del kwargs["output_format"]
        del kwargs["output_path"]


        dir = os.path.dirname(filepath)
        if not os.path.isdir(dir):
            log.info("output directory { %s } not exist" % (dir))
            log.info("creating output directory { %s }..." % (dir))
            os.makedirs(dir)
            log.info("created output directory { %s }..." % (dir))
        if os.path.exists(filepath):
            raise IOError("output path { %s } already exist" % (filepath))

        log.info("writing data to location {%s}" % (filepath))

        if "parquet" == output_format.lower():
            cudf.io.parquet.to_parquet(df, filepath, **kwargs)
        else:
            df.to_csv(filepath, **kwargs)

    def close(self):
        log.info("Closed writer")
