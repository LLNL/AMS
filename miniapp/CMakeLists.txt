# ------------------------------------------------------------------------------
# handle sources and headers
file(GLOB_RECURSE AMS_APP_INCLUDE "*.hpp")

set(AMS_LIB_SRC ${AMS_APP_INCLUDE} miniapp_lib.cpp)
set(AMS_EXE_SRC ${AMS_APP_INCLUDE} miniapp_exe.cpp)

# two targets: a shared lib and an exec
add_library(miniapp_lib SHARED ${AMS_LIB_SRC} ${AMS_INCLUDES})
add_executable(miniapp ${AMS_EXE_SRC} ${AMS_INCLUDES})

# ------------------------------------------------------------------------------
if (WITH_CUDA)
    set_source_files_properties(miniapp_lib.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(miniapp_exe.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(miniapp_lib PUBLIC "--expt-extended-lambda")
    target_compile_options(miniapp PUBLIC "--expt-extended-lambda")
endif()

link_directories(${CMAKE_CURRENT_BINARY_DIR})

#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#      # using regular Clang or AppleClang
#else()
#endif()


# ------------------------------------------------------------------------------
# setup the lib first
target_compile_definitions(miniapp_lib PUBLIC ${AMS_DEFINES})
target_include_directories(miniapp_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(miniapp_lib PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(miniapp_lib PUBLIC ${AMS_APP_LIBRARIES})


# setup the exec
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

target_compile_definitions(miniapp PUBLIC ${AMS_DEFINES})
target_include_directories(miniapp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(miniapp PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(miniapp PUBLIC ${AMS_APP_LIBRARIES} mmp)

# rename the exec target to "mmp"
set_target_properties(miniapp PROPERTIES OUTPUT_NAME miniapp)

# ------------------------------------------------------------------------------
# installation paths
install(TARGETS miniapp_lib DESTINATION lib)
install(TARGETS miniapp DESTINATION bin)

# ------------------------------------------------------------------------------
