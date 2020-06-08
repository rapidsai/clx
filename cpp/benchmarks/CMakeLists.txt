cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CLX_BENCHS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(BENCHMARK_LIST CACHE INTERNAL "BENCHMARK_LIST")

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   #"${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp"
                   #"${CMAKE_SOURCE_DIR}/tests/utilities/base_fixture.cpp")
    )
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark benchmark_main pthread clx )
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    set(BENCHMARK_LIST ${BENCHMARK_LIST} ${CMAKE_BENCH_NAME} CACHE INTERNAL "BENCHMARK_LIST")
endfunction(ConfigureBench)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------
message(STATUS "1. ${CUB_INCLUDE_DIR} \n 2. ${THRUST_INCLUDE_DIR} \n 3. ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} \n4. ${CMAKE_BINARY_DIR}/include \n5. ${CMAKE_SOURCE_DIR}/include \n6. ${CMAKE_SOURCE_DIR} \n7. ${CMAKE_SOURCE_DIR}/src \n8. ${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include \n9. ${CMAKE_SOURCE_DIR}/thirdparty/jitify \n10. ${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include \n11. ${GTEST_INCLUDE_DIR} \n111. ${GBENCH_INCLUDE_DIR} \n12. ${ARROW_INCLUDE_DIR} \n13. ${FLATBUFFERS_INCLUDE_DIR} \n14. ${RMM_INCLUDE} \n15. ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(#"${CUB_INCLUDE_DIR}"
                    #"${THRUST_INCLUDE_DIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    #"${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    #"${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    #"${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include"
                    #"${GTEST_INCLUDE_DIR}"
                    "${GBENCH_INCLUDE_DIR}"
                    #"${ARROW_INCLUDE_DIR}"
                    #"${FLATBUFFERS_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")



###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

###################################################################################################
# - tokenizer benchmarks -----------------------------------------------------------------------------

set(TOKENIZER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_benchmark.cu")

ConfigureBench(TOKENIZER_BENCH "${TOKENIZER_BENCH_SRC}")