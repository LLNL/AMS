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
option(WITH_MPI "Option to enable MPI" OFF)
option(WITH_TORCH "Use C/C++ Torch interface for model inference" OFF)
option(WITH_TORCH_DEBUG "Compute RMSE of Surrogate Model and Physics Module" OFF)
# ------------------------------------------------------------------------------
if (WITH_CUDA)
  #we need to enable nvcc
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  set(CMAKE_CUDA_STANDARD 11)
  #Turn off message of mfem
  set(ENABLE_CUDA True)
  list(APPEND AMS_APP_LIBRARIES ${CUDA_LIBRARIES} cuda)
  list(APPEND AMS_DEFINES "-D__ENABLE_CUDA__")
endif()

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

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
if (NOT MFEM_DIR)
  find_package(MFEM REQUIRED HINTS ${MFEM_DIR} ${MFEM_DIR}/share/mfem/)
else()
  include(cmake/FindMFEM.cmake)
endif()

if (NOT MFEM_FOUND)
  message(WARNING "WARNING CANNOT FIND MFEM")
endif()

if ( WITH_TORCH )
  find_package(Torch REQUIRED)
  list(APPEND AMS_APP_INCLUDE "${TORCH_INCLUDE_DIRS}")
  list(APPEND AMS_APP_LIBRARIES "${TORCH_LIBRARIES}")
  message(WARN "Torch Include dirs are: ${TORCH_INCLUDE_DIRS}")
  message(WARN "Torch Include dirs are: ${TORCH_LIBRARIES}")
  list(APPEND AMS_DEFINES "-D__ENABLE_TORCH__")
  if ( WITH_TORCH_DEBUG )
    list(APPEND AMS_DEFINES "-D__SURROGATE_DEBUG__")
  endif()
endif()

add_subdirectory(src)

# ------------------------------------------------------------------------------
