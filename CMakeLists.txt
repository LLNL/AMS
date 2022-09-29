# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)
project(AMS VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)

# Enable this to get compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(AMS_APP_LIBRARIES "")
set(AMS_APP_DEFINES "")
set(AMS_APP_INCLUDES "")

set(AMS_EXAMPLE_LIBRARIES "")
set(AMS_EXAMPLE_DEFINES "")
set(AMS_EXAMPLE_INCLUDES "")

# ------------------------------------------------------------------------------
option(WITH_CUDA        "Option to enable CUDA" OFF)
option(WITH_EXAMPLES     "Build examples" OFF)
option(WITH_MPI         "Option to enable MPI" OFF)
option(WITH_DB          "Use Database for Caching Simulation Results" OFF)
option(WITH_CALIPER     "Use Caliper for Profiling" OFF)
option(WITH_FAISS       "Use C/C++ FAISS interface for HD Cache" OFF)
option(WITH_TORCH       "Use C/C++ Torch interface for Surrogate Model Inference" OFF)
option(WITH_TORCH_DEBUG "Compute RMSE of Surrogate Model and Physics Module" OFF)
option(WITH_TESTS "Compile tests" OFF)

# ------------------------------------------------------------------------------
if (WITH_CUDA)
  # we need to enable nvcc
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "Please Define CMAKE_CUDA_ARCHITECTURES")
  endif()

  # Turn off message of mfem
  set(ENABLE_CUDA True)
  list(APPEND AMS_APP_LIBRARIES ${CUDA_LIBRARIES} cuda)
  list(APPEND AMS_APP_DEFINES "-D__ENABLE_CUDA__")
endif()

# ------------------------------------------------------------------------------
if (WITH_CALIPER)
  find_package(caliper REQUIRED)
  list(APPEND AMS_APP_INCLUDES ${caliper_INCLUDE_DIR})
  list(APPEND AMS_APP_LIBRARIES caliper)
  list(APPEND AMS_APP_DEFINES "-D__ENABLE_CALIPER__")
endif()


# ------------------------------------------------------------------------------
if (WITH_DB)
  list(APPEND AMS_APP_DEFINES "-D__ENABLE_DB__")
endif()

# ------------------------------------------------------------------------------
if(NOT DEFINED UMPIRE_DIR)
  message(FATAL_ERROR "Missing required 'UMPIRE_DIR' variable pointing to an installed Umpire ${UMPIRE_DIR}")
endif()

find_package(UMPIRE REQUIRED
             NO_DEFAULT_PATH
             PATHS ${UMPIRE_DIR}/lib/cmake/umpire ${UMPIRE_DIR}/share/umpire/cmake)
list(APPEND AMS_APP_LIBRARIES umpire)
list(APPEND AMS_APP_INCLUDES ${UMPIRE_INCLUDE_DIR})

# ------------------------------------------------------------------------------
find_package(Threads REQUIRED)

# ------------------------------------------------------------------------------
if (WITH_TORCH)
  find_package(Torch REQUIRED)

  list(APPEND AMS_APP_INCLUDES "${TORCH_INCLUDE_DIRS}")
  list(APPEND AMS_APP_LIBRARIES "${TORCH_LIBRARIES}")

  list(APPEND AMS_APP_DEFINES "-D__ENABLE_TORCH__")
endif()

# ------------------------------------------------------------------------------
if (WITH_FAISS)
  ## TODO: still need to create FindFaiss.cmake
  #find_package(FAISS REQUIRED HINTS ${FAISS_DIR})
  #include(${PROJECT_SOURCE_DIR}/cmake/FindFAISS.cmake)

  if (FAISS_DIR)
    message(STATUS "FAISS_DIR = ${FAISS_DIR}")
    set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/include")
    set(FAISS_LIB_DIR "${FAISS_DIR}/lib")
    set(FAISS_LIB64_DIR "${FAISS_DIR}/lib64")
    set(FAISS_LIBRARIES "faiss")
  else()
    message(FATAL_ERROR "Cannot find FAISS. FAISS_DIR is not defined.")
  endif()

  ## TODO: we need to make a FindFaiss.cmake
  list(APPEND AMS_APP_INCLUDES "${FAISS_INCLUDE_DIRS}")
  list(APPEND AMS_APP_LIB_DIRS "${FAISS_LIB_DIR}" "${FAISS_LIB64_DIR}")
  list(APPEND AMS_APP_LIBRARIES "${FAISS_LIBRARIES}")
  list(APPEND AMS_APP_DEFINES "-D__ENABLE_FAISS__")


  find_package(OpenMP)
  if (OPENMP_FOUND)
     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()

if (WITH_RZ)
  find_package(MPI REQUIRED)
  add_subdirectory(rz)
  list(APPEND AMS_APP_INCLUDES "${RZ_AMS_INCLUDES}" "${MPI_INCLUDE_PATH}")
  list(APPEND AMS_APP_LIB_DIRS "${RZ_AMS_LIBDIRS}")
  list(APPEND AMS_APP_LIBRARIES "${RZ_AMS_LIBRARIES}" "${MPI_C_LIBRARIES}")
  list(APPEND AMS_APP_DEFINES "${RZ_AMS_DEFINES}")
endif()

# TODO: currently we are focusing only on BLASS. Try out other libraries.
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
list(APPEND AMS_APP_LIBRARIES "${BLAS_LIBRARIES}")

add_subdirectory(src)

# ------------------------------------------------------------------------------
if (WITH_EXAMPLES)
  #The AMS Runtime does not depend on MFEM. The
  #examples we use do.

  list(APPEND AMS_EXAMPLE_LIBRARIES umpire)
  list(APPEND AMS_EXAMPLE_INCLUDES ${UMPIRE_INCLUDE_DIR})

  if (MFEM_DIR)
    include(${PROJECT_SOURCE_DIR}/cmake/FindMFEM.cmake)
  else()
    find_package(MFEM REQUIRED)
  endif()
  list(APPEND AMS_EXAMPLE_INCLUDES "${MFEM_INCLUDE_DIRS}")
  list(APPEND AMS_EXAMPLE_LIBRARIES "${MFEM_LIBRARIES}")
  list(APPEND AMS_EXAMPLE_LIB_DIRS "${MFEM_LIB_DIR}")

  if (WITH_CUDA)
    list(APPEND AMS_EXAMPLE_LIBRARIES ${CUDA_LIBRARIES} cuda)
    list(APPEND AMS_EXAMPLE_DEFINES "-D__ENABLE_CUDA__")
  endif()

  if (WITH_CALIPER)
    list(APPEND AMS_EXAMPLE_INCLUDES ${caliper_INCLUDE_DIR})
    list(APPEND AMS_EXAMPLE_LIBRARIES caliper)
    list(APPEND AMS_EXAMPLE_DEFINES "-D__ENABLE_CALIPER__")
  endif()

  add_subdirectory(examples)
endif()

if (WITH_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------

