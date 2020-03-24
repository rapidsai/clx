cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cuda_tokenizer LANGUAGES CXX CUDA)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
--expt-relaxed-constexpr \
--expt-extended-lambda \
--default-stream per-thread \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-unused-function")

set(CU_TOK_LIBS
    cudart
    nvToolsExt
    tokenizers
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

add_library(tokenizers SHARED
  src/hash_utils.cu
  src/basicTokenizer.cu
  src/wordPieceTokenizer.cu
  src/fullTokenizer.cu
  src/main.cu
)

target_link_libraries(tokenizers nvToolsExt)

add_executable(tokenizer
  src/main.cu
)

set_target_properties(tokenizer
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_link_libraries(tokenizer ${CU_TOK_LIBS})

add_dependencies(tokenizers gen_cuda_header)

install(TARGETS tokenizers LIBRARY
    DESTINATION lib)
