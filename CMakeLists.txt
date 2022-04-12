cmake_minimum_required(VERSION 3.1)
project(AMSMiniApp CXX C) #CUDA)

set(CMAKE_CXX_STANDARD 11)

#Enable this to get compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")


# ------------------------------------------------------------------------------
option(WITH_DB "Option to enable the DB" OFF)
option(WITH_CUDA "Option to enable CUDA" OFF)
option(WITH_CALIPER "Option to enable the DB" OFF)

# ------------------------------------------------------------------------------
if (WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    set(CMAKE_CUDA_STANDARD 11)
    list(APPEND AMS_APP_LIBRARIES ${CUDA_LIBRARIES} cuda)
endif()


# ------------------------------------------------------------------------------
if (WITH_DB)
  list(APPEND AMS_DEFINES "-D__ENABLE_DB__")
endif()


# ------------------------------------------------------------------------------
if (WITH_CALIPER)
  find_package(caliper REQUIRED)
  list(APPEND AMS_APP_INCLUDE ${caliper_INCLUDE_DIR})
  list(APPEND AMS_APP_LIBRARIES caliper)
  list(APPEND AMS_DEFINES "-D__ENABLE_CALIPER__")
endif()


# ------------------------------------------------------------------------------
find_package(MFEM REQUIRED HINTS ${MFEM_DIR})
add_subdirectory(src)

# ------------------------------------------------------------------------------
