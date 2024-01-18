# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB_RECURSE AMS_CURRENT_EXAMPLE_INCLUDES "*.hpp")

set(AMS_EXAMPLE_SRC ${MINIAPP_INCLUDES} main.cpp app/eos_ams.cpp)


function(ADDExec binary_name definitions)
  if (WITH_RZ)
    list(APPEND AMS_EXAMPLE_SRC ${RZ_AMS_SOURCES} ${AMS_CURRENT_EXAMPLE_INCLUDES})
  endif()

  target_include_directories(${binary_name} PRIVATE ${AMS_EXAMPLE_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}/include)
  target_compile_definitions(${binary_name} PRIVATE ${definitions})
  target_link_directories(${binary_name} PRIVATE ${AMS_EXAMPLE_LIB_DIRS})
  target_link_libraries(${binary_name} PUBLIC AMS ${AMS_EXAMPLE_LIBRARIES})

  if (WITH_CUDA)
      set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
      set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
      set_source_files_properties(app/eos_ams.cpp PROPERTIES LANGUAGE CUDA)
      set_source_files_properties(app/eos_ams.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
      set_target_properties(${binary_name} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")

      if (WITH_RZ)
          set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES LANGUAGE CUDA)
          set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
          set_property(TARGET ${binary_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      endif()

      if (WITH_PERFFLOWASPECT)
        set_property(SOURCE ${AMS_EXAMPLE_SRC} APPEND_STRING PROPERTY COMPILE_FLAGS " -ccbin clang++ -Xcompiler=-Xclang -Xcompiler=-load -Xcompiler=-Xclang -Xcompiler=${PERFFLOWASPECT_LIB_DIR}/libWeavePass.so")
      endif()
  endif()
endfunction()

add_executable(no_ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(no_ams_example "${AMS_EXAMPLE_DEFINES}")

list(APPEND AMS_EXAMPLE_DEFINES "-DUSE_AMS")
add_executable(ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})
ADDExec(ams_example "${AMS_EXAMPLE_DEFINES}")

if (WITH_WORKFLOW)
  set(TRAIN_DEVICE "cpu")
  if (WITH_CUDA)
    set(TRAIN_DEVICE "gpu")
  endif()
  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/prune.py" "${CMAKE_CURRENT_BINARY_DIR}/prune.py")
  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/train.py" "${CMAKE_CURRENT_BINARY_DIR}/train.py")
  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/sub_selection.py" "${CMAKE_CURRENT_BINARY_DIR}/sub_selection.py")
  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/example_run.sh" "${CMAKE_CURRENT_BINARY_DIR}/example_run.sh" @ONLY)
endif()
