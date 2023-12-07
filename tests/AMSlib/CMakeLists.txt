# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function (BUILD_TEST exe source)
  add_executable(${exe} ${source})
  target_include_directories(${exe} PRIVATE "${PROJECT_SOURCE_DIR}/src/AMSlib/" umpire ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_link_directories(${exe} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${exe} PRIVATE AMS ${AMS_APP_LIBRARIES})

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
  if (WITH_CUDA)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
    set_property(TARGET ${exe} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)

    target_compile_definitions(${exe} PRIVATE "-D__ENABLE_CUDA__ -DLIBAMS_VERBOSE")
  endif()
endfunction()

function (ADDTEST exe test_name)
  add_test(NAME "${test_name}::HOST" COMMAND ${exe} 0 ${ARGN})
  if (WITH_CUDA)
    add_test(NAME "${test_name}::DEVICE" COMMAND ${exe} 1 ${ARGN})
  endif()
endfunction()

BUILD_TEST(ams_allocator_test ams_allocate.cpp)
ADDTEST(ams_allocator_test AMSAllocate)
BUILD_TEST(ams_packing_test cpu_packing_test.cpp AMSPack)
ADDTEST(ams_packing_test AMSPack)

if (WITH_TORCH)
  BUILD_TEST(ams_inference_test torch_model.cpp)
  ADDTEST(ams_inference_test AMSInferDouble ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "double")
  ADDTEST(ams_inference_test AMSInferSingle ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "single")
  add_test(NAME AMSExampleInferSingle::HOST COMMAND  ams_example --precision single -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
  add_test(NAME AMSExampleInferDouble::HOST COMMAND  ams_example --precision double -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
endif()

add_test(NAME AMSInferSingleDeltaUQ::HOST COMMAND  ams_example --precision single --uqtype deltauq-mean -db ./db -S ${CMAKE_CURRENT_SOURCE_DIR}/tuple-single.torchscript -e 100)

add_test(NAME AMSExampleSingleRandomUQ::HOST COMMAND ams_example --precision single -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt --uqtype random -e 100)
add_test(NAME AMSExampleDoubleRandomUQ::HOST COMMAND ams_example --precision double -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt --uqtype random -e 100)

if(WITH_FAISS)
  BUILD_TEST(ams_hdcache_test test_hdcache.cpp)
  ADDTEST(ams_hdcache_test AMSHDCacheMeanPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 1 10 4.0 4 5)
  # The max case fails on DEVICE. We should be aware about this when adding support for CI for GPUs
  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 2 10 4.0 4 5)

  ADDTEST(ams_hdcache_test AMSHDCacheMeanPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 1 10 4.0 4 5)
  # The max case fails on DEVICE. We should be aware about this when adding support for CI for GPUs
  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 2 10 4.0 4 5)
  if (WITH_TORCH)
    add_test(NAME AMSExampleFaissInferSingle::HOST COMMAND  ams_example --precision single -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
    add_test(NAME AMSExampleFaissInferDouble::HOST COMMAND  ams_example --precision double -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
  endif()
endif()
