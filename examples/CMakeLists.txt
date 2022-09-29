file(GLOB_RECURSE AMS_CURRENT_EXAMPLE_INCLUDES "*.hpp")

set(AMS_EXAMPLE_SRC ${MINIAPP_INCLUDES} miniapp_lib.cpp miniapp_exe.cpp)
add_executable(ams_example ${AMS_EXAMPLE_SRC} ${MINIAPP_INCLUDES})

if (WITH_RZ)
  list(APPEND AMS_EXAMPLE_SRC ${RZ_AMS_SOURCES} ${AMS_CURRENT_EXAMPLE_INCLUDES})
endif()

target_compile_definitions(ams_example PRIVATE ${AMS_EXAMPLE_DEFINES})
target_include_directories(ams_example PRIVATE ${AMS_EXAMPLE_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(ams_example PRIVATE ${AMS_EXAMPLE_LIB_DIRS})
target_link_libraries(ams_example PUBLIC AMS ${AMS_EXAMPLE_LIBRARIES})

if (WITH_CUDA)
    set_source_files_properties(miniapp_exe.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(miniapp_lib.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(miniapp_lib.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_source_files_properties(miniapp_exe.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

    if (WITH_RZ)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(${RZ_AMS_SOURCES} PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
        set_property(TARGET ams_example PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()
