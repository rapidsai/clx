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
        self.__offset_boundries = self.__get_boundaries()

    @property
    def offset_boundaries(self):
        return self.__offset_boundries

    @property
    def dataset_len(self):
        return self.__dataset_len
    
    @property
    def data(self):
        return self.__df
        
    def __get_boundaries(self):
        """Creates dataframe boundries
        """
        prev_chunk_offset = 0
        offset_boundaries = []
        while prev_chunk_offset < self.__dataset_len:
            curr_chunk_offset = prev_chunk_offset + self.__batchsize
            offset_boundary = (prev_chunk_offset, curr_chunk_offset)
            offset_boundaries.append(offset_boundary)
            prev_chunk_offset = curr_chunk_offset
        return offset_boundaries
    
    def get_chunk(self, idx_boundary):
        """Returns a chunck of original input dataframe based on index boundary
        :param idx_boundary: Start and Stop index
        :type idx_boundary: Tuple(int, int)
        :return: Part of original input dataframe
        :rtype: cudf.DataFrame
        """
        df_chunck = self.__df[idx_boundary[0]:idx_boundary[1]:1]
        return df_chunck