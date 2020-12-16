import logging

log = logging.getLogger(__name__)

class DataLoader(object):
    """
    Wrapper class is used to return dataframe partitions based on batchsize.
    """

    def __init__(self, dataset, batchsize=1000):
        """Constructor to create dataframe partitions.

        :param df: Input dataframe.
        :type df: cudf.DataFrame
        :param batch_size: Number of records in the dataframe.
        :type batch_size: int
        """
        self.__dataset = dataset
        self.__batchsize = batchsize

    @property
    def dataset_len(self):
        return self.__dataset.length
    
    @property
    def dataset(self):
        return self.__dataset
        
    def get_chunks(self):
        """Returns chunks of original input dataframe based on batchsize
        """
        prev_chunk_offset = 0
        while prev_chunk_offset < self.__dataset.length:
            curr_chunk_offset = prev_chunk_offset + self.__batchsize
            chunk = self.__dataset.data[prev_chunk_offset:curr_chunk_offset:1]
            prev_chunk_offset = curr_chunk_offset
            yield chunk
            