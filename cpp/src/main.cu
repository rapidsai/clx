#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>

#include "cuda_profiler_api.h"
#include "tokenizers.cuh"
#include "nvToolsExt.h"
#include "for_cython.h"
using namespace std::chrono;

#define MAX_NUM_SENTENCES 100
#define MAX_NUM_CHARS 100000
#define MAX_ROWS_TENSOR 500


void cuda_tokenizer_cudf(char* device_sentences, uint32_t* offsets, uint32_t offset_size, std::string hash_file, uint32_t max_sequence_length, uint32_t stride, bool do_lower,
                         bool do_truncate, uint32_t max_num_sentences_, uint32_t max_num_chars_, uint32_t max_rows_tensor_, TokenizerResult *result)  {

  // Create tokenizer
  nvtxRangePushA("create Tokenizer");
  GpuFullTokenizer tokenizer(hash_file, max_num_sentences_, max_num_chars_, max_rows_tensor_,
                                   max_sequence_length, stride, do_truncate, do_lower);
  nvtxRangePop();

  // Run GPU tokenizer
  nvtxRangePushA("Tokenize");
  tokenizer.tokenize(device_sentences, offsets, offset_size);
  nvtxRangePop();

  // Get output from tokenizer
  result->nrows_tensor = tokenizer.get_nrows_tensor_tokenIDS();
  cudaMalloc((void**)&result->device_tensor_tokenIDS, result->nrows_tensor*max_sequence_length*sizeof(uint32_t));
  cudaMalloc((void**)&result->device_attention_mask, result->nrows_tensor*max_sequence_length*sizeof(uint32_t));
  cudaMalloc((void**)&result->device_tensor_metadata, result->nrows_tensor*3*sizeof(uint32_t));
  cudaMemcpy(result->device_tensor_tokenIDS, tokenizer.get_tensor_tokenIDS(), result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_attention_mask, tokenizer.get_attention_mask(), result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_tensor_metadata, tokenizer.get_tensor_metadata(), result->nrows_tensor*3*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

}

void cuda_tokenizer_file(std::string input_file_name, std::string hash_file, uint32_t max_sequence_length, uint32_t stride, bool do_lower, bool do_truncate,
                         uint32_t max_num_sentences_, uint32_t max_num_chars_, uint32_t max_rows_tensor_, TokenizerResult *result)  {
  // Create tokenizer
  nvtxRangePushA("create Tokenizer");
  GpuFullTokenizer tokenizer(hash_file, max_num_sentences_, max_num_chars_, max_rows_tensor_, 
                                   max_sequence_length, stride, do_truncate, do_lower);
  nvtxRangePop();

  // Load Logs
  nvtxRangePushA("Load Logs");
  std::ifstream input_file(input_file_name);
  if(!input_file.good()){
    std::cerr << "File " << input_file_name << " not found." << std::endl;
    exit(1);
  }
  std::vector<std::string> logs;
  std::string line;
  while (std::getline(input_file, line)) {
    logs.push_back(line);
  }
  nvtxRangePop();


  // Run GPU tokenizer
  nvtxRangePushA("Tokenize");
  tokenizer.tokenize(logs);
  nvtxRangePop();


  // Get output from tokenizer
  result->nrows_tensor = tokenizer.get_nrows_tensor_tokenIDS();
  cudaMalloc((void**)&result->device_tensor_tokenIDS, result->nrows_tensor*max_sequence_length*sizeof(uint32_t));
  cudaMalloc((void**)&result->device_attention_mask, result->nrows_tensor*max_sequence_length*sizeof(uint32_t));
  cudaMalloc((void**)&result->device_tensor_metadata, result->nrows_tensor*3*sizeof(uint32_t));
  cudaMemcpy(result->device_tensor_tokenIDS, tokenizer.get_tensor_tokenIDS(), result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_attention_mask, tokenizer.get_attention_mask(), result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(result->device_tensor_metadata, tokenizer.get_tensor_metadata(), result->nrows_tensor*3*sizeof(uint32_t), cudaMemcpyDeviceToDevice);

}


int main(int argc, char *argv[]) {

  uint32_t max_sequence_length = 64;
  uint32_t stride = 48;
  bool do_truncate = false;
  bool do_lower = true;
  std::string input_file_name;
  std::string hash_file;

  if (argc < 7) {
    std::cout << "Usage:" << std::endl;
    std::cout << "    " << argv[0] << "  input_file  hash_table_path  max_sequence_length  stride  do_truncate  do_lower" << std::endl;
    return 1;
  }else{
    input_file_name = argv[1];
    hash_file = argv[2];
    max_sequence_length = std::stoi(argv[3]);
    stride = std::stoi(argv[4]);
    do_truncate = std::stoi(argv[5]);
    do_lower = std::stoi(argv[6]);
  }
  if (max_sequence_length < stride) {
    std::cout << "Error: max_sequence_length must be larger than  stride" << std::endl;
    return 1;
  }

  TokenizerResult* result = new TokenizerResult();

  cuda_tokenizer_file(input_file_name, hash_file, max_sequence_length, stride, do_lower, do_truncate,
                      MAX_NUM_SENTENCES, MAX_NUM_CHARS, MAX_ROWS_TENSOR, result);


  std::vector<uint32_t> host_final_tensor;
  std::vector<uint32_t> host_attn_mask;
  std::vector<uint32_t> host_metadata;
  host_final_tensor.resize(result->nrows_tensor*max_sequence_length);
  host_attn_mask.resize(result->nrows_tensor*max_sequence_length);
  host_metadata.resize(result->nrows_tensor*3);

  cudaMemcpy(host_final_tensor.data(), result->device_tensor_tokenIDS, result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_attn_mask.data(), result->device_attention_mask, result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_metadata.data(), result->device_tensor_metadata, result->nrows_tensor*3*sizeof(uint32_t), cudaMemcpyDeviceToHost);

  printf("\n --- TENSOR --- \n");
  for (int i=0; i<host_final_tensor.size(); i++){
    if (i!=0 && i%max_sequence_length==0) printf("\n\n");
    printf("%u ",host_final_tensor[i]);
  }
  printf("\n");
  printf("\n\n --- MASK ---- \n");
  for (int i=0; i<host_attn_mask.size(); i++){
    if (i!=0 &&  i%max_sequence_length==0) printf("\n\n");
    printf("%u ", host_attn_mask[i]);
  }
  printf("\n\n --- METADATA ---- \n");
  for (int i=0; i<result->nrows_tensor; i++) printf("%u %u %u\n", host_metadata[i*3], host_metadata[i*3+1], host_metadata[i*3+2]);
}
