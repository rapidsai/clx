#include <benchmark/benchmark.h>
#include <for_cython.h>

#include <thrust/device_vector.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#define MAX_NUM_SENTENCES 101
#define MAX_NUM_CHARS 150000
#define MAX_ROWS_TENSOR 300

static void BM_cuda_tokenizer_file(benchmark::State& state) {
  std::string input_file_name = "cpp/benchmarks/tokenizer_benchmark.txt";
  std::string hash_file = "python/clx/analytics/resources/bert_hash_table.txt";
  uint32_t max_sequence_length = 64;
  uint32_t stride = 48;
  uint32_t do_truncate = 0;
  uint32_t do_lower = 1;
  TokenizerResult* result = new TokenizerResult();
  for (auto _ : state){
    cuda_tokenizer_file(input_file_name, hash_file, max_sequence_length, stride, do_lower, do_truncate,
      MAX_NUM_SENTENCES, MAX_NUM_CHARS, MAX_ROWS_TENSOR, result);
  }
}
BENCHMARK(BM_cuda_tokenizer_file);

void flatten_sentences(const std::vector<std::string>& sentences,
  char* flattened_sentences,
  uint32_t* sentence_offsets) {

      uint32_t start_copy = 0;
      for(uint32_t i = 0; i < sentences.size(); ++i){
        const uint32_t sentence_length = sentences[i].size();

        sentences[i].copy(flattened_sentences + start_copy, sentence_length);
        sentence_offsets[i] = start_copy;
        start_copy += sentence_length;
      }
      sentence_offsets[sentences.size()] = start_copy;
}

static void BM_cuda_tokenizer_cudf(benchmark::State& state) {
  rmm::device_vector<char> device_sentences{};
  device_sentences.resize(MAX_NUM_CHARS);

  std::string sentences = "This is a test";
  std::vector<char> char_sentences(sentences.length());
  std::copy(sentences.begin(), sentences.end(), char_sentences.begin());
  device_sentences = char_sentences;

  std::string hash_file = "python/clx/analytics/resources/bert_hash_table.txt";
  std::vector<uint32_t> offsets{14};
  uint32_t max_sequence_length = 64;
  uint32_t stride = 48;
  uint32_t do_truncate = 0;
  uint32_t do_lower = 1;
  TokenizerResult* result = new TokenizerResult();
  for (auto _ : state){
    cuda_tokenizer_cudf(thrust::raw_pointer_cast(device_sentences.data()), offsets.data(), offsets.size(), hash_file, max_sequence_length, stride, do_lower, do_truncate,
      MAX_NUM_SENTENCES, MAX_NUM_CHARS, MAX_ROWS_TENSOR, result);
  }
}
BENCHMARK(BM_cuda_tokenizer_cudf);

BENCHMARK_MAIN();