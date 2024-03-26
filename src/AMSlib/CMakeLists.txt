# Copyright (c) Lawrence Livermore National Security, LLC and other AMS
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute

# ------------------------------------------------------------------------------
# handle sources and headers
file(GLOB_RECURSE MINIAPP_INCLUDES "*.hpp")
#set global library path to link with tests if necessary
set(LIBRARY_OUTPUT_PATH ${AMS_LIB_OUT_PATH})
set(AMS_LIB_SRC ${MINIAPP_INCLUDES} AMS.cpp wf/resource_manager.cpp wf/debug.cpp wf/basedb.cpp wf/logger.cpp)

if (WITH_CUDA)
  list(APPEND AMS_LIB_SRC wf/cuda/utilities.cpp)
  message(WARNING "FILES ARE ${AMS_LIB_SRC}")
endif()


# two targets: a shared lib and an exec
add_library(AMS ${AMS_LIB_SRC} ${MINIAPP_INCLUDES})

# ------------------------------------------------------------------------------
if (WITH_CUDA)

  set_target_properties(AMS PROPERTIES CUDA_ARCHITECTURES ${AMS_CUDA_ARCH})

  #    if (BUILD_SHARED_LIBS)
  #      set_target_properties(AMS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #    else()
  #      set_target_properties(AMS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  #      set_target_properties(AMS PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  #    endif()

  set_source_files_properties(wf/cuda/utilities.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(wf/cuda/utilities.cpp PROPERTIES CUDA_ARCHITECTURES ${AMS_CUDA_ARCH})
  set_source_files_properties(wf/cuda/utilities.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

  if (WITH_PERFFLOWASPECT)
      set_property(SOURCE AMS.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -Xcompiler=-Xclang -Xcompiler=-load -Xcompiler=-Xclang -Xcompiler=${PERFFLOWASPECT_LIB_DIR}/libWeavePass.so")
      set_source_files_properties(wf/resource_manager.cpp COMPILE_FLAGS "-Xclang -load -Xclang ${PERFFLOWASPECT_LIB_DIR}/libWeavePass.so")
    endif()
endif()

# ------------------------------------------------------------------------------
# setup the lib first
message(STATUS "ALL INCLUDES ARE ${AMS_APP_INCLUDES}")
target_compile_definitions(AMS PRIVATE ${AMS_APP_DEFINES})
target_include_directories(AMS PRIVATE ${AMS_APP_INCLUDES})
target_include_directories(AMS PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:include/>)
target_include_directories(AMS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(AMS PUBLIC ${AMS_APP_LIB_DIRS})
if (EXCLUDE_STATIC_LIBS)
   target_link_libraries(AMS PRIVATE ${NONSTATIC_AMS_APP_PRIVATE_LIBRARIES} ${AMS_TORCH_LIBRARY} stdc++fs stdc++ /usr/tce/packages/intel/intel-2022.1.0/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libintlc.so m)
   target_link_libraries(AMS INTERFACE ${NONSTATIC_AMS_APP_INTERFACE_LIBRARIES} stdc++fs stdc++ /usr/tce/packages/intel/intel-2022.1.0/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libintlc.so m)
   target_link_options(AMS PRIVATE "-Wl,--unresolved-symbols=ignore-all")
else()
   target_link_libraries(AMS PUBLIC ${AMS_APP_LIBRARIES} stdc++fs stdc++ /usr/tce/packages/intel/intel-2022.1.0/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libintlc.so m)
endif()

#-------------------------------------------------------------------------------
# create the configuration header file with the respective information
#-------------------------------------------------------------------------------
set(CALIPER_DEFINES "// #define __AMS_ENABLE_CALIPER__")
set(MPI_DEFINES "// #define __AMS_ENABLE_MPI__")
set(PERFF_DEFINES "// #define __AMS_ENABLE_PERFFLOWASPECT__")

if (${WITH_CALIPER})
  set(CALIPER_DEFINES "#define __AMS_ENABLE_CALIPER__")
endif()

if (${WITH_MPI})
  set(MPI_DEFINES "#define __AMS_ENABLE_MPI__")
endif()

if (${WITH_PERFFLOWASPECT})
  set(PERFF_DEFINES "#define __AMS_ENABLE_PERFFLOWASPECT__")
endif()

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMS-config.h.in" "${PROJECT_BINARY_DIR}/include/AMS-config.h")
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/AMS.h" "${PROJECT_BINARY_DIR}/include/AMS.h" COPYONLY)

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

install(FILES ${PROJECT_BINARY_DIR}/include/AMS.h DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/include/AMS-config.h DESTINATION include)
