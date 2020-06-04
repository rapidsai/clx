cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

project(cuda_tokenizer LANGUAGES CXX CUDA)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
 --expt-relaxed-constexpr \
--expt-extended-lambda \
--default-stream per-thread \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-O3 \
-lineinfo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-unused-function")

set(CU_TOK_LIBS
    cudart
    nvToolsExt
    clx
)

include_directories(
  ./cub
)

add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/src/cp_data.h
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/generate_gpu_metadata.py --out_file ${CMAKE_SOURCE_DIR}/src/cp_data.h
  DEPENDS python/generate_gpu_metadata.py
  COMMENT "Generating code point meta data"
)

ADD_CUSTOM_TARGET(gen_cuda_header ALL
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cp_data.h
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" ON)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{RMM_ROOT}/lib" "$ENV{RMM_ROOT}/build")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if(RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif(RMM_INCLUDE AND RMM_LIBRARY)

include_directories("${RMM_INCLUDE}")
link_directories("${RMM_LIBRARY}")

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)

  include(ConfigureGoogleBenchmark)

  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)

endif(BUILD_BENCHMARKS)

message(STATUS "BENCHMARK_LIST set to: ${BENCHMARK_LIST}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------
add_library(clx SHARED
  src/hash_utils.cu
  src/basicTokenizer.cu
  src/wordPieceTokenizer.cu
  src/fullTokenizer.cu
  src/main.cu
)
###################################################################################################
# - gtest -------------------------------------------------------------------------------
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(tokenizer_tests tests/tokenizer_test.cu)
target_link_libraries(tokenizer_tests ${GTEST_LIBRARIES} ${CU_TOK_LIBS} rmm)
set_target_properties(tokenizer_tests
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${RMM_INCLUDE}")
###################################################################################################
add_executable(tokenizer
  src/main.cu
)

add_executable(tokenizer_benchmark
  benchmarks/tokenizer_benchmark.cu
)

target_link_libraries(tokenizer ${CU_TOK_LIBS} rmm)

target_link_libraries(tokenizer_benchmark ${CU_TOK_LIBS} rmm benchmark::benchmark)

target_link_libraries(clx nvToolsExt rmm)

target_include_directories(tokenizer
    PRIVATE
    ./include)

set_target_properties(tokenizer
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

set_target_properties(tokenizer_benchmark
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

add_dependencies(clx gen_cuda_header)

install(TARGETS clx LIBRARY
    DESTINATION lib)