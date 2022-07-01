# ------------------------------------------------------------------------------
# handle sources and headers
file(GLOB_RECURSE MINIAPP_INCLUDES "*.hpp")

#set global library path to link with tests if necessary
set(LIBRARY_OUTPUT_PATH ${AMS_LIB_OUT_PATH})

set(AMS_LIB_SRC ${MINIAPP_INCLUDES} miniapp_lib.cpp wf/utilities.cpp)
set(AMS_EXE_SRC ${MINIAPP_INCLUDES} miniapp_exe.cpp)

# two targets: a shared lib and an exec
add_library(miniapp SHARED ${AMS_LIB_SRC} ${MINIAPP_INCLUDES})
add_executable(miniapp_bin ${AMS_EXE_SRC} ${MINIAPP_INCLUDES})


# ------------------------------------------------------------------------------
if (WITH_CUDA)
    set_source_files_properties(miniapp_lib.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(miniapp_exe.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(miniapp_lib.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
    set_source_files_properties(miniapp_exe.cpp PROPERTIES COMPILER_FLAGS "--expt-extended-lambda")
endif()

#link_directories(${CMAKE_CURRENT_BINARY_DIR})

#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#      # using regular Clang or AppleClang
#else()
#endif()


# ------------------------------------------------------------------------------
# setup the lib first
target_compile_definitions(miniapp PUBLIC ${AMS_APP_DEFINES})
target_include_directories(miniapp PRIVATE ${AMS_APP_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(miniapp PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(miniapp PUBLIC ${AMS_APP_LIBRARIES})


# setup the exec
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

target_compile_definitions(miniapp_bin PUBLIC ${AMS_APP_DEFINES})
target_include_directories(miniapp_bin PRIVATE ${AMS_APP_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(miniapp_bin PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(miniapp_bin PUBLIC ${AMS_APP_LIBRARIES} miniapp)

# rename the exec target to "miniapp"
set_target_properties(miniapp_bin PROPERTIES OUTPUT_NAME miniapp)

# ------------------------------------------------------------------------------
# installation paths
install(TARGETS miniapp DESTINATION lib)
install(TARGETS miniapp_bin DESTINATION bin)

# ------------------------------------------------------------------------------
