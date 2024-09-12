# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(AMS_BENCHMARK_SRC ${MINIAPP_INCLUDES} ams_bench_db.cpp)

function(ADDExec binary_name definitions)
  target_include_directories(${binary_name} PRIVATE ${AMS_BENCHMARKS_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}/include "${PROJECT_SOURCE_DIR}/src/AMSlib/" umpire ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_compile_definitions(${binary_name} PRIVATE ${definitions})
  target_link_directories(${binary_name} PRIVATE ${AMS_BENCHMARKS_LIB_DIRS})
  target_link_libraries(${binary_name} PUBLIC AMS ${AMS_BENCHMARKS_LIBRARIES})

  if (WITH_CUDA)
      set_source_files_properties(ams_bench_db.cpp PROPERTIES LANGUAGE CUDA)
      set_source_files_properties(ams_bench_db.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
      set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
      set_property(TARGET ${binary_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()
  if (WITH_PERFFLOWASPECT)
    set_property(SOURCE ${AMS_BENCHMARK_SRC} APPEND_STRING PROPERTY COMPILE_FLAGS " -Xcompiler=-Xclang -Xcompiler=-load -Xcompiler=-Xclang -Xcompiler=${PERFFLOWASPECT_LIB_DIR}/libWeavePass.so")
  endif()
endfunction()

# Database benchmark (RMQ and/or HDF5 + MPI/CUDA . No ML model used)
add_executable(ams_benchmark_db ${AMS_BENCHMARK_SRC} ${MINIAPP_INCLUDES})
ADDExec(ams_benchmark_db "${AMS_BENCHMARKS_DEFINES}")
