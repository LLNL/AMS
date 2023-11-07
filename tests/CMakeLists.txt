# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function (ADDTEST binary_name src_file test_name)
  add_executable(${binary_name} ${src_file})
  target_include_directories(${binary_name} PRIVATE "${PROJECT_SOURCE_DIR}/src" umpire ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_link_directories(${binary_name} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${binary_name} PRIVATE AMS umpire MPI::MPI_CXX)

  add_test(NAME "${test_name}::HOST" COMMAND ${binary_name} 0 ${ARGN})

  target_compile_definitions(${binary_name} PRIVATE ${AMS_APP_DEFINES})
  if (WITH_CUDA)
    set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
    set_property(TARGET ${binary_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE CUDA)

    target_compile_definitions(${binary_name} PRIVATE "-D__ENABLE_CUDA__ -DLIBAMS_VERBOSE")
    add_test(NAME "${test_name}_DEVICE" COMMAND ${binary_name} 1 ${ARGN})
  endif()
endfunction()

ADDTEST(ams_allocator ams_allocate.cpp AMSAllocate)
ADDTEST(ams_packing cpu_packing_test.cpp AMSPack)
if (WITH_TORCH)
ADDTEST(ams_inference_double torch_model.cpp AMSInferDouble ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "double")
ADDTEST(ams_inference_single torch_model.cpp AMSInferSingle ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "single")
add_test(NAME AMSExampleInferSingle::HOST COMMAND  ams_example --precision single -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
add_test(NAME AMSExampleInferDouble::HOST COMMAND  ams_example --precision double -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
endif()

add_test(NAME AMSInferSingleDeltaUQ::HOST COMMAND  ams_example --precision single --uqtype deltauq-mean -db ./db -S ${CMAKE_CURRENT_SOURCE_DIR}/tuple-single.torchscript -e 100)
if(WITH_FAISS)
ADDTEST(ams_hdcache_mean_double test_hdcache.cpp AMSHDCacheMeanPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 1 10 4.0 4 5)
# The max case fails on DEVICE. We should be aware abou this when adding support for CI for GPUs
ADDTEST(ams_hdcache_max_double  test_hdcache.cpp AMSHDCacheMaxPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 2 10 4.0 4 5)

ADDTEST(ams_hdcache_mean_single test_hdcache.cpp AMSHDCacheMeanPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 1 10 4.0 4 5)
# The max case fails on DEVICE. We should be aware abou this when adding support for CI for GPUs
ADDTEST(ams_hdcache_max_single  test_hdcache.cpp AMSHDCacheMaxPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 2 10 4.0 4 5)
if (WITH_TORCH)
  add_test(NAME AMSExampleFaissInferSingle::HOST COMMAND  ams_example --precision single -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
  add_test(NAME AMSExampleFaissInferDouble::HOST COMMAND  ams_example --precision double -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
endif()
endif()

