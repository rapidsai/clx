from libcpp cimport bool
from libcpp.string cimport string
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
import cudf
import torch
from torch.utils.dlpack import from_dlpack
    
cdef extern from "../../../cpp/src/for_cython.h":
    struct TokenizerResult:
        unsigned int nrows_tensor
        unsigned int* device_tensor_tokenIDS
        unsigned int* device_attention_mask
        unsigned int* device_tensor_metadata

    cdef void cuda_tokenizer_file(string input_file_name, string hash_file, unsigned int max_sequence_length, unsigned int stride, bool do_lower, bool do_truncate,
                                  unsigned int max_num_sentences_, unsigned int max_num_chars_, unsigned int max_rows_tensor_, TokenizerResult* result) except+
    cdef void cuda_tokenizer_cudf(char* device_sentences, unsigned int* offsets, unsigned int offset_size, string hash_file, unsigned int max_sequence_length, unsigned int stride,
                                  bool do_lower, bool do_truncate, unsigned int max_num_sentences_, unsigned int max_num_chars_, unsigned int max_rows_tensor_,
                                  TokenizerResult* result) except+

import numpy as np
import cupy
def device_array_from_ptr(ptr, shape, dtype):
    dtype=np.dtype(dtype)
    elemsize = dtype.itemsize
    datasize = elemsize * shape[0] * shape[1]
    strides = (elemsize*shape[1], elemsize)
    base_mem = cupy.cuda.memory.UnownedMemory(ptr, datasize, None)
    mem = cupy.cuda.MemoryPointer(base_mem, 0)
    return cupy.ndarray(shape, dtype, mem, strides)


def tokenize_file(input_file, hash_file, max_sequence_length=64, stride=48, do_lower=True, do_truncate=False, max_num_sentences=100, max_num_chars=100000, max_rows_tensor=500):

    cdef TokenizerResult *result
    result = <TokenizerResult *>calloc(1, sizeof(TokenizerResult))

    cuda_tokenizer_file(input_file.encode(), hash_file.encode(), max_sequence_length, stride, do_lower, do_truncate,
                        max_num_sentences, max_num_chars, max_rows_tensor, result)

    device_tokenIDS = device_array_from_ptr(<uintptr_t>result.device_tensor_tokenIDS,
                                            shape=(result.nrows_tensor,max_sequence_length),
                                            dtype=np.int32)
    device_mask = device_array_from_ptr(<uintptr_t>result.device_attention_mask,
                                        shape=(result.nrows_tensor,max_sequence_length),
                                        dtype=np.int32)
    device_metadata = device_array_from_ptr(<uintptr_t>result.device_tensor_metadata,
                                            shape=(result.nrows_tensor,3),
                                            dtype=np.int32)

    token = from_dlpack(device_tokenIDS.toDlpack())
    mask = from_dlpack(device_mask.toDlpack())
    metadata = from_dlpack(device_metadata.toDlpack())
    return token.type(torch.long), mask.type(torch.long), metadata.type(torch.long)


def tokenize_df(input_df, hash_file, max_sequence_length=64, stride=48, do_lower=True, do_truncate=False, max_num_sentences=100, max_num_chars=100000, max_rows_tensor=500):

    if isinstance(input_df, cudf.DataFrame):
        col = input_df.iloc[:,0]
    elif isinstance(input_df, cudf.Series):
        col = input_df
    else:
        raise ValueError("Input must be a cudf.DataFrame or cudf.Series")

    d_arr=cupy.empty(len(input_df), dtype=np.uint32)
    col.str.byte_count(d_arr.data.ptr,True)
    offsets = cupy.asnumpy(d_arr)

    cdef TokenizerResult *result
    result = <TokenizerResult *>calloc(1,sizeof(TokenizerResult))

    cdef uintptr_t offsets_array = <uintptr_t>offsets.__array_interface__['data'][0]
    cdef unsigned int* offsets_ptr = <unsigned int*>offsets_array
    cdef uintptr_t data_device_array = <uintptr_t>col._column.children[1].serialize()[0]['data']['desc']['data'][0]
    cdef char* data_ptr = <char*> data_device_array

    cuda_tokenizer_cudf(data_ptr, offsets_ptr, len(offsets), hash_file.encode(), max_sequence_length, stride, do_lower, do_truncate,
                        max_num_sentences, max_num_chars, max_rows_tensor, result)

    device_tokenIDS = device_array_from_ptr(<uintptr_t>result.device_tensor_tokenIDS,
                                            shape=(result.nrows_tensor,max_sequence_length),
                                            dtype=np.int32)
    device_mask = device_array_from_ptr(<uintptr_t>result.device_attention_mask,
                                        shape=(result.nrows_tensor,max_sequence_length),
                                        dtype=np.int32)
    device_metadata = device_array_from_ptr(<uintptr_t>result.device_tensor_metadata,
                                            shape=(result.nrows_tensor,3),
                                            dtype=np.int32)

    token = from_dlpack(device_tokenIDS.toDlpack())
    mask = from_dlpack(device_mask.toDlpack())
    metadata = from_dlpack(device_metadata.toDlpack())
    return token.type(torch.long), mask.type(torch.long), metadata.type(torch.long)
