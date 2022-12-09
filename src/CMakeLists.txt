# ------------------------------------------------------------------------------
# handle sources and headers
file(GLOB_RECURSE MINIAPP_INCLUDES "*.hpp")
#set global library path to link with tests if necessary
set(LIBRARY_OUTPUT_PATH ${AMS_LIB_OUT_PATH})
set(AMS_LIB_SRC ${MINIAPP_INCLUDES} AMS.cpp wf/resource_manager.cpp)
# two targets: a shared lib and an exec
add_library(AMS SHARED ${AMS_LIB_SRC} ${MINIAPP_INCLUDES})

# ------------------------------------------------------------------------------
if (WITH_CUDA)
    #TODO: replace all-major by native with cmake >= 3.24
    message("CUDA ARCH is '${AMS_CUDA_ARCH}'")
    set_target_properties(AMS PROPERTIES CUDA_ARCHITECTURES ${AMS_CUDA_ARCH})
    set_source_files_properties(AMS.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(AMS.cpp PROPERTIES CUDA_ARCHITECTURES ${AMS_CUDA_ARCH})
    set_source_files_properties(AMS.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
endif()

# ------------------------------------------------------------------------------
# setup the lib first
message(STATUS "ALL INCLUDES ARE ${AMS_APP_INCLUDES}")
target_compile_definitions(AMS PRIVATE ${AMS_APP_DEFINES})
target_include_directories(AMS PRIVATE ${AMS_APP_INCLUDES})
target_include_directories(AMS PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:include/>)
target_link_directories(AMS PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(AMS PRIVATE ${AMS_APP_LIBRARIES})

# setup the exec
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
# ------------------------------------------------------------------------------
# installation paths
install(TARGETS AMS
        EXPORT AMSTargets
        DESTINATION lib)

install(EXPORT AMSTargets
  FILE AMS.cmake
  DESTINATION lib/cmake/AMS)

install(FILES AMS.h DESTINATION include)
