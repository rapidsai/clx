#include <string.h>
#include <stdint.h>

struct TokenizerResult {
  uint32_t nrows_tensor;
  uint32_t *device_tensor_tokenIDS;
  uint32_t *device_attention_mask;
  uint32_t *device_tensor_metadata;

TokenizerResult(): nrows_tensor(0), device_tensor_tokenIDS(nullptr), device_attention_mask(nullptr), device_tensor_metadata(nullptr) {};
~TokenizerResult() {};
};

void cuda_tokenizer_file(std::string input_file_name, std::string hash_file, uint32_t max_sequence_length, uint32_t stride, bool do_lower, bool do_truncate,
                         uint32_t max_num_sentences_, uint32_t max_num_chars_, uint32_t max_rows_tensor_, TokenizerResult* result);
void cuda_tokenizer_cudf(char* device_sentences, uint32_t* offsets, uint32_t offset_size, std::string hash_file, uint32_t max_sequence_length, uint32_t stride, bool do_lower, bool do_truncate,
                         uint32_t max_num_sentences_, uint32_t max_num_chars_, uint32_t max_rows_tensor_, TokenizerResult* result);
