import rmm
import cudf
import logging
import numpy as np

log = logging.getLogger(__name__)


class DetectorDataset(object):
    """
    Wrapper class is used to hold the partitioned datframes and number of the records in all partitions.
    """

    def __init__(self, partitioned_dfs, dataset_len):
        """This function instantiates partitioned datframes and number of the records in all partitions.
        
        :param partitioned_dfs: Partitioned dataframes.
        :type partitioned_dfs: list(dataframes)
        :param dataset_len: Number of records in the list of partitioned dataframes.
        :type dataset_len: int
        """
        self.__partitioned_dfs = partitioned_dfs
        self.__dataset_len = dataset_len

    @property
    def partitioned_dfs(self):
        return self.__partitioned_dfs

    @property
    def dataset_len(self):
        return self.__dataset_len


# https://github.com/rapidsai/cudf/issues/2861
# Workaround for partitioning dataframe to small batches
def prepare_detector_dataset(df, batch_size):
    """Partition one dataframe to multiple small dataframes based on a given batch size.
    :param df: Contains domains and it's types.
    :type df: cudf.DataFrame
    :param batch_size: Number of records has to be in each partitioned dataframe.
    :type batch_size: int
    """
    dataset_len = df["domain"].count()
    df = str2ascii(df, dataset_len)
    prev_chunk_offset = 0
    partitioned_dfs = []
    while prev_chunk_offset < dataset_len:
        curr_chunk_offset = prev_chunk_offset + batch_size
        chunk = df.iloc[prev_chunk_offset:curr_chunk_offset:1]
        partitioned_dfs.append(chunk)
        prev_chunk_offset = curr_chunk_offset
    return DetectorDataset(partitioned_dfs, dataset_len)


def str2ascii(df, domains_len):
    """This function sorts domain name entries in desc order based on the length of domain and converts domain name to ascii characters.
    
    :param df: Domains which requires conversion.
    :type df: cudf.DataFrame
    :param domains_len: Number of entries in df.
    :type domains_len: int
    :return: Ascii character converted information.
    :rtype: cudf.DataFrame
    """
    df["len"] = df["domain"].str.len()
    df = df.sort_values("len", ascending=False)
    splits = df["domain"].str.findall("[\w\.\-\@]")
    split_df = cudf.DataFrame()
    columns_cnt = len(splits)

    for i in range(0, columns_cnt):
        split_df[i] = splits[i]

    # https://github.com/rapidsai/cudf/issues/3123
    # Replace null's with ^.
    split_df = split_df.fillna("^")
    temp_df = cudf.DataFrame()
    for col in range(0, columns_cnt):
        ascii_darr = rmm.device_array(domains_len, dtype=np.int32)
        split_df[col].str.code_points(ascii_darr.device_ctypes_pointer.value)
        temp_df[col] = ascii_darr
    del split_df
    # https://github.com/rapidsai/cudf/issues/3123
    # Replace ^ ascii value 94 with 0.
    temp_df = temp_df.replace(94, 0)
    temp_df["len"] = df["len"]
    if "type" in df.columns:
        temp_df["type"] = df["type"]
    temp_df["domain"] = df["domain"]
    temp_df.index = df.index
    return temp_df
