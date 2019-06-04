import json
import logging
import os

import cudf
import yaml


class ParserHelper:
    @staticmethod
    def load_regex_yaml(yaml_file):
        logging.info("reading yaml file : %s" % (yaml_file))
        with open(yaml_file) as f:
            regex_dict = yaml.safe_load(f)
            regex_dict = {k: v[0] for k, v in regex_dict.items()}
        return regex_dict

    @staticmethod
    def create_regex_dictionaries(yaml_directory):
        regex_dict = {}
        for f in os.listdir(yaml_directory):
            yaml_file = "%s/%s" % (yaml_directory, f)
            temp_regex = ParserHelper.load_regex_yaml(yaml_file)
            regex_dict[f[:-5]] = temp_regex

        return regex_dict

    @staticmethod
    def get_dummy_gdf(superset_col_filepath, dtypes):
        gdf = cudf.read_csv(superset_col_filepath, dtype=dtypes)
        gdf = gdf[:0]
        return gdf

    @staticmethod
    def construct_csv_record(rec, output_cols_superset):
        csv_rec = ",".join(
            [
                (str(rec[col]).rstrip() if rec[col] is not None else "")
                for col in output_cols_superset
            ]
        )
        return csv_rec

    @staticmethod
    def generate_delimited_ouput_col(gdf, separator):
        first_col = gdf.columns[0]
        gdf[first_col] = gdf[first_col].data.fillna("")
        gdf["delimited_ouput"] = gdf[first_col].str.rstrip()
        for col in gdf.columns[1:-1]:
            gdf[col] = gdf[col].data.fillna("")
            gdf[col] = gdf[col].str.rstrip()
            gdf["delimited_ouput"] = gdf.delimited_ouput.str.cat(
                gdf[col], sep=separator
            )
        return gdf

    @staticmethod
    def post_parsing_cleanup(out_df, cleanup_required_cols, pattern, replacement):
        if cleanup_required_cols is None:
            logging.warn(
                "null value appeared. Please check cleanup required columns values!!!"
            )
        else:
            for col in cleanup_required_cols:
                out_df[col] = out_df[col].str.replace(pattern, replacement)
        return out_df
