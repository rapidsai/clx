#pragma once

#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

static void gpuCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
      std::cerr << cudaGetErrorString(err) << " in file " << file << " at line " 
                                           << line << "." << std::endl;
      exit(1);
  }
}

#define assertCudaSuccess(cu_err) {gpuCheck((cu_err), __FILE__, __LINE__);}

template<typename T>
void malloc_and_copy_vec_to_device(rmm::device_vector<T>& dest, std::vector<T> vec) {

  //assertCudaSuccess(cudaMalloc(dest_ptr, sizeof(T) * vec.size()));
  dest.resize(vec.size());
  //assertCudaSuccess(cudaMemcpy(*dest_ptr, vec.data(), vec.size() * sizeof(T),
  //                             cudaMemcpyHostToDevice));
  thrust::copy(vec.begin(), vec.end(), dest.begin());
}

