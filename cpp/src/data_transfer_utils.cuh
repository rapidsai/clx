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
void malloc_and_copy_vec_to_device(rmm::device_vector<T>& dest, std::vector<T> const& vec) {
  dest.resize(vec.size());
  thrust::copy(vec.begin(), vec.end(), dest.begin());
}

