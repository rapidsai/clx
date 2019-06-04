import cudf
import logging
from rapidscyber.reader.file_reader import FileReader


class NFSReader(FileReader):
    def __init__(self, config):
        self._config = config
        self._has_data = True

    def read_text(
        self, input_path, schema, delimiter, required_cols, datatypes, header
    ):
        df = cudf.read_csv(
            input_path,
            names=schema,
            delimiter=delimiter,
            usecols=required_cols,
            dtype=datatypes,
            header=header,
            skip_blank_lines=True,
        )

        return df

    def read_parquet(self, input_path, required_cols):
        df = cudf.read_parquet(input_path, columns=required_cols)
        return df

    def read_orc(self, input_path):
        df = cudf.read_orc(input_path, engine="cudf")
        return df

    def fetch_data(self):
        df = None
        input_format = self.config["input_format"].lower()
        if "parquet" == input_format:
            df = self.read_parquet(
                self.config["input_path"], self.config["required_cols"]
            )
        elif "orc" == input_format:
            df = self.read_orc(self.config["input_path"])
        else:
            df = self.read_text(
                self.config["input_path"],
                self.config["schema"],
                self.config["delimiter"],
                self.config["required_cols"],
                self.config["dtype"],
                self.config["header"],
            )
        self.has_data = False
        return df
