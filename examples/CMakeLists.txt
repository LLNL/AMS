# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB_RECURSE AMS_CURRENT_EXAMPLE_INCLUDES "*.hpp")

set(AMS_EXAMPLE_SRC ${MINIAPP_INCLUDES} main.cpp)
add_executable(ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})

if (WITH_RZ)
  list(APPEND AMS_EXAMPLE_SRC ${RZ_AMS_SOURCES} ${AMS_CURRENT_EXAMPLE_INCLUDES})
endif()

target_include_directories(ams_example PRIVATE ${AMS_EXAMPLE_INCLUDES}
  ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}/include)
target_compile_definitions(ams_example PRIVATE ${AMS_EXAMPLE_DEFINES})
target_link_directories(ams_example PRIVATE ${AMS_EXAMPLE_LIB_DIRS})
target_link_libraries(ams_example PUBLIC AMS ${AMS_EXAMPLE_LIBRARIES})
if (WITH_PERFFLOWASPECT)
    target_link_libraries(ams_example PUBLIC perfflowaspect_full)
endif()

if (WITH_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_source_files_properties(app/eos_idealgas.hpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_target_properties(ams_example PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")

    if (WITH_RZ)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
        set_property(TARGET ams_example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()

    if (WITH_PERFFLOWASPECT)
      set_property(SOURCE ${AMS_EXAMPLE_SRC} APPEND_STRING PROPERTY COMPILE_FLAGS " -ccbin clang++ -Xcompiler=-Xclang -Xcompiler=-load -Xcompiler=-Xclang -Xcompiler=${PERFFLOWASPECT_LIB_DIR}/libWeavePass.so")
    endif()
endif()
