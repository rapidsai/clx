import cudf
import logging
from clx.analytics import detector_utils as du

log = logging.getLogger(__name__)


class DetectorDataset(object):
    """
    Wrapper class is used to hold the partitioned datframes and number of the records in all partitions.
    """

    def __init__(self, df, batch_size):
        """This function instantiates partitioned datframes and number of the records in all partitions.
        
        :param df: domains dataframe.
        :type df: cudf.DataFrame
        :param batch_size: Number of records in the dataframe.
        :type batch_size: int
        """
        self.__partitioned_dfs, self.__dataset_len = self.__get_partitioned_dfs(
            df, batch_size
        )

    @property
    def partitioned_dfs(self):
        return self.__partitioned_dfs

    @property
    def dataset_len(self):
        return self.__dataset_len

    # https://github.com/rapidsai/cudf/issues/2861
    # https://github.com/rapidsai/cudf/issues/1473
    # Workaround for partitioning dataframe into small batches
    def __get_partitioned_dfs(self, df, batch_size):
        """Partition one dataframe to multiple small dataframes based on a given batch size.
        :param df: Contains domains and it's types.
        :type df: cudf.DataFrame
        :param batch_size: Number of records has to be in each partitioned dataframe.
        :type batch_size: int
        """
        dataset_len = df["domain"].count()
        df = du.str2ascii(df, dataset_len)
        prev_chunk_offset = 0
        partitioned_dfs = []
        while prev_chunk_offset < dataset_len:
            curr_chunk_offset = prev_chunk_offset + batch_size
            chunk = df.iloc[prev_chunk_offset:curr_chunk_offset:1]
            partitioned_dfs.append(chunk)
            prev_chunk_offset = curr_chunk_offset
        return partitioned_dfs, dataset_len
