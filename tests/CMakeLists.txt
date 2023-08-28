# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function (ADDTEST binary_name src_file test_name)
  add_executable(${binary_name} ${src_file})
  target_include_directories(${binary_name} PRIVATE "${PROJECT_SOURCE_DIR}/src/AMSlib" umpire ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_link_directories(${binary_name} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${binary_name} PRIVATE AMS umpire MPI::MPI_CXX)

  add_test(NAME "${test_name}::HOST" COMMAND ${binary_name} 0 ${ARGN})
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
ADDTEST(ams_inference torch_model.cpp AMSInfer /usr/workspace/AMS/miniapp_resources/trained_models/debug_model.pt)
ADDTEST(ams_loadBalance lb.cpp AMSLoadBalance)
