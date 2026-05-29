# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchAndAddCatch2)
set(_ams_catch2_args)
if (AMS_CATCH2_DIR)
  list(APPEND _ams_catch2_args PATHS "${AMS_CATCH2_DIR}" NO_DEFAULT_PATH)
  set(Catch2_DIR ${AMS_CATCH2_DIR} CACHE PATH "")
endif()
find_package(Catch2 CONFIG QUIET ${_ams_catch2_args})
if (NOT TARGET Catch2::Catch2)
  message(STATUS "Catch2 not found via CMake package discovery; falling back to FetchContent")
  FetchAndAddCatch2(ON)
else()
  message(STATUS "Catch2 version found: ${Catch2_VERSION}")
endif()
if (NOT TARGET Catch2::Catch2)
  message(FATAL_ERROR
          "Unable to find Catch2. Provide a Catch2 package or allow "
          "FetchContent to populate it.")
endif()

# Optional: bring in Catch2's CTest adapter (catch_discover_tests)
# (The module lives under Catch2/extras in source builds and alongside
# Catch2Config.cmake in installed packages.)
foreach (_ams_catch2_source_dir IN ITEMS "${Catch2_SOURCE_DIR}" "${catch2_SOURCE_DIR}")
  if (_ams_catch2_source_dir AND EXISTS "${_ams_catch2_source_dir}/extras/Catch.cmake")
    list(APPEND CMAKE_MODULE_PATH "${_ams_catch2_source_dir}/extras")
  endif()
endforeach()
if (DEFINED Catch2_DIR AND EXISTS "${Catch2_DIR}/Catch.cmake")
  list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
endif()
include(Catch OPTIONAL RESULT_VARIABLE _ams_catch2_ctest_adapter)
if (NOT _ams_catch2_ctest_adapter)
  message(STATUS "Catch2 CTest adapter not found; using explicit add_test registrations")
endif()

if (ENABLE_TORCH)
  add_subdirectory(models)
  add_subdirectory(torch)
endif()
add_subdirectory(core)
add_subdirectory(db)
add_subdirectory(wf)
add_subdirectory(ams_interface)
add_subdirectory(downstream)
#add_subdirectory(perf_regression)
