#include <benchmark/benchmark.h>
#include "../src/for_cython.h"

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

BENCHMARK_MAIN();