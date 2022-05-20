# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)
project(AMSMiniApp CXX C)

set(CMAKE_CXX_STANDARD 14)

# Enable this to get compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")


# ------------------------------------------------------------------------------
option(WITH_CUDA        "Option to enable CUDA" OFF)
option(WITH_MPI         "Option to enable MPI" OFF)
option(WITH_DB          "Use Database for Caching Simulation Results" OFF)
option(WITH_CALIPER     "Use Caliper for Profiling" OFF)
option(WITH_FAISS       "Use C/C++ FAISS interface for HD Cache" OFF)
option(WITH_TORCH       "Use C/C++ Torch interface for Surrogate Model Inference" OFF)
option(WITH_TORCH_DEBUG "Compute RMSE of Surrogate Model and Physics Module" OFF)


# ------------------------------------------------------------------------------
if (WITH_CUDA)
  # we need to enable nvcc
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()

  # Turn off message of mfem
  set(ENABLE_CUDA True)
  list(APPEND AMS_APP_LIBRARIES ${CUDA_LIBRARIES} cuda)
  list(APPEND AMS_DEFINES "-D__ENABLE_CUDA__")
endif()


# ------------------------------------------------------------------------------
if (WITH_CALIPER)
  find_package(caliper REQUIRED)

  list(APPEND AMS_APP_INCLUDE ${caliper_INCLUDE_DIR})
  list(APPEND AMS_APP_LIBRARIES caliper)
  list(APPEND AMS_DEFINES "-D__ENABLE_CALIPER__")
endif()


# ------------------------------------------------------------------------------
if (WITH_DB)
  list(APPEND AMS_DEFINES "-D__ENABLE_DB__")
endif()


# ------------------------------------------------------------------------------
if (MFEM_DIR)
  include(${PROJECT_SOURCE_DIR}/cmake/FindMFEM.cmake)
else()
  find_package(MFEM REQUIRED)  # HINTS ${MFEM_DIR} ${MFEM_DIR}/share/mfem/)
endif()

list(APPEND AMS_APP_INCLUDE "${MFEM_INCLUDE_DIRS}")
list(APPEND AMS_APP_LIBRARIES "${MFEM_LIBRARIES}")
list(APPEND AMS_APP_LIB_DIRS "${MFEM_LIB_DIR}")


# ------------------------------------------------------------------------------
if (WITH_TORCH)
  find_package(Torch REQUIRED)

  message("Torch Includes: ${TORCH_INCLUDE_DIRS}")
  message("Torch Libraries: ${TORCH_LIBRARIES}")

  list(APPEND AMS_APP_INCLUDE "${TORCH_INCLUDE_DIRS}")
  list(APPEND AMS_APP_LIBRARIES "${TORCH_LIBRARIES}")

  list(APPEND AMS_DEFINES "-D__ENABLE_TORCH__")
  if (WITH_TORCH_DEBUG)
    list(APPEND AMS_DEFINES "-D__SURROGATE_DEBUG__")
  endif()
endif()


# ------------------------------------------------------------------------------
if (WITH_FAISS)
  ## TODO: still need to create FindFaiss.cmake
  #find_package(FAISS REQUIRED HINTS ${FAISS_DIR})
  #include(${PROJECT_SOURCE_DIR}/cmake/FindFAISS.cmake)

  if (FAISS_DIR)
    message(STATUS "FAISS_DIR = ${FAISS_DIR}")
    set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/include")
    set(FAISS_LIB_DIR "${FAISS_DIR}/lib64")
    set(FAISS_LIBRARIES "faiss")
  else()
    message(FATAL_ERROR "Cannot find FAISS. FAISS_DIR is not defined.")
  endif()

  list(APPEND AMS_APP_INCLUDE "${FAISS_INCLUDE_DIRS}")
  list(APPEND AMS_APP_LIB_DIRS "${FAISS_LIB_DIR}")
  list(APPEND AMS_APP_LIBRARIES "${FAISS_LIBRARIES}")
  list(APPEND AMS_DEFINES "-D__ENABLE_FAISS__")
endif()


#message("includes: ${AMS_APP_INCLUDE}")
#message("libdirs: ${AMS_APP_LIBDIRS}")
#message("libs: ${AMS_APP_LIBRARIES}")
#message("defines: ${AMS_DEFINES}")


# ------------------------------------------------------------------------------
add_subdirectory(src)

# ------------------------------------------------------------------------------
