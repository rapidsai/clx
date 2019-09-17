import cudf
import logging
import os

from clx.io.writer.file_writer import FileWriter

log = logging.getLogger(__name__)

class FileSystemWriter(FileWriter):
    def __init__(self, output_path, output_format="text"):
        self._output_path = output_path
        self._output_format = output_format

    def is_valid_path(fun):
        def wrapper(self, df, output_path):
            dir = os.path.dirname(output_path)
            if not os.path.isdir(dir):
                log.info("output directory { %s } not exist" % (dir))
                log.info("creating output directory { %s }..." % (dir))
                os.makedirs(dir)
                log.info("created output directory { %s }..." % (dir))
            if os.path.exists(output_path):
                raise IOError("output path { %s } already exist" % (output_path))
            log.info("writing data to location {%s}" % (output_path))
            fun(self, df, output_path)

        return wrapper

    @is_valid_path
    def write_as_text(self, df, output_path):
        df.to_pandas().to_csv(output_path, index=False)

    @is_valid_path
    def write_as_parquet(self, df, output_path):
        cudf.io.parquet.to_parquet(df, output_path)

    def write_data(self, df):
        if "parquet" == self._output_format.lower():
            self.write_as_parquet(df, self._output_path)
        else:
            self.write_as_text(df, self._output_path)

    def close(self):
        log.info("Closed writer")
