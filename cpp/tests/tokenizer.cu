#include "../src/for_cython.h"
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#define MAX_NUM_SENTENCES 101
#define MAX_NUM_CHARS 150000
#define MAX_ROWS_TENSOR 300


TEST(TokenizerTest, SmallDataSample) {
    std::string input_file_name = "cpp/tests/tokenizer_test.txt";
    std::string hash_file = "python/clx/analytics/resources/bert_hash_table.txt";

    //Write to test file
    std::fstream testfile;
    testfile.open(input_file_name, std::fstream::out);
    testfile << "This is a test.\n";
    testfile.close();

    uint32_t max_sequence_length = 64;
    uint32_t stride = 48;
    uint32_t do_truncate = 0;
    uint32_t do_lower = 1;
    TokenizerResult* result = new TokenizerResult();
    cuda_tokenizer_file(input_file_name, hash_file, max_sequence_length, stride, do_lower, do_truncate,
    101, 150000, 300, result);

    std::vector<uint32_t> host_final_tensor;
    std::vector<uint32_t> host_attn_mask;
    std::vector<uint32_t> host_metadata;
    host_final_tensor.resize(result->nrows_tensor*max_sequence_length);
    host_attn_mask.resize(result->nrows_tensor*max_sequence_length);
    host_metadata.resize(result->nrows_tensor*3);

    cudaMemcpy(host_final_tensor.data(), result->device_tensor_tokenIDS, result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_attn_mask.data(), result->device_attention_mask, result->nrows_tensor*max_sequence_length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_metadata.data(), result->device_tensor_metadata, result->nrows_tensor*3*sizeof(uint32_t), cudaMemcpyDeviceToHost);


    std::vector<uint32_t> expected_tensor;
    std::vector<uint32_t> expected_attn_mask;
    std::vector<uint32_t> expected_metadata;
    expected_tensor = {2023, 2003, 1037, 3231, 1012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(expected_tensor, host_final_tensor);
    expected_attn_mask = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(expected_attn_mask, host_attn_mask);
    expected_metadata = {0, 0, 4};
    EXPECT_EQ(expected_metadata, host_metadata);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}