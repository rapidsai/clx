import logging
from clx.analytics import detector_utils as du

log = logging.getLogger(__name__)

class DetectorDataset(object):
    """
    Wrapper class is used to hold the partitioned datframes and number of the records in all partitions.
    """

    def __init__(self, df, batchsize=1000):
        """Constructor to create dataframe partitions.

        :param df: Input dataframe.
        :type df: cudf.DataFrame
        :param batch_size: Number of records in the dataframe.
        :type batch_size: int
        """
        self.__df = df.reset_index(drop=True)
        self.__dataset_len = df.shape[0]
        self.__batchsize = batchsize

    @property
    def dataset_len(self):
        return self.__dataset_len
    
    @property
    def data(self):
        return self.__df
        
    def get_chunks(self):
        """Returns a chunks of original input dataframe based on batchsize
        """
        prev_chunk_offset = 0
        while prev_chunk_offset < self.__dataset_len:
            curr_chunk_offset = prev_chunk_offset + self.__batchsize
            chunk = self.__df[prev_chunk_offset:curr_chunk_offset:1]
            prev_chunk_offset = curr_chunk_offset
            yield chunk
            