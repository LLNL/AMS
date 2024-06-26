# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(binomial_options_src binomial_options.cpp kernel.cpp)

function(ADDExec binary_name definitions)
  target_include_directories(${binary_name} PUBLIC "${AMS_EXAMPLE_INCLUDES}")
  target_compile_definitions(${binary_name} PRIVATE ${definitions})
  target_compile_definitions(${binary_name} PRIVATE ${AMS_EXAMPLE_DEFINES})
  target_link_directories(${binary_name} PRIVATE ${AMS_EXAMPLE_LIB_DIRS})
  target_link_libraries(${binary_name} PUBLIC AMS ${AMS_EXAMPLE_LIBRARIES})

  if (WITH_CUDA)
      set_source_files_properties(kernel.cpp PROPERTIES LANGUAGE CUDA)
      set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
  endif()
endfunction()

add_executable(no_ams_bo ${binomial_options_src} ${MINIAPP_INCLUDES})
ADDExec(no_ams_bo minibude "${AMS_EXAMPLE_DEFINES}")

list(APPEND AMS_EXAMPLE_DEFINES "-DUSE_AMS")
add_executable(ams_bo ${binomial_options_src} ${MINIAPP_INCLUDES})
ADDExec(ams_bo "${AMS_EXAMPLE_DEFINES}")

