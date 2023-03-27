# Copyright (c) Lawrence Livermore National Security, LLC and other AMS
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute

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

if (WITH_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

    set_target_properties(ams_example PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")

    if (WITH_RZ)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
        set_property(TARGET ams_example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()
