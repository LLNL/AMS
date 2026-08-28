function(ADD_CORE_UNIT_TEST name exec)
  string(JOIN " " args ${ARGN})
  add_test(NAME ${name} COMMAND ${exec} -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS CORE_UNIT_TEST)
endfunction()

function(BUILD_UNIT_TEST exe source)
  add_executable(${exe} ${source})

  target_compile_features(${exe} PRIVATE cxx_std_17)
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

  set(catch2_target Catch2::Catch2WithMain)
  if (${ARGC} GREATER 2)
    set(catch2_target ${ARGV2})
  endif()
  if (${ARGC} GREATER 3)
    target_sources(${exe} PRIVATE ${ARGV3})
  endif()

  target_link_libraries(${exe} PRIVATE stdc++fs AMS ${catch2_target})
  target_link_libraries(${exe} PRIVATE fmt::fmt)

  if (ENABLE_TORCH)
    target_link_libraries(${exe} PRIVATE torch)
  endif()
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)

  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_TARGET})

  target_link_libraries(${exe} PRIVATE Threads::Threads)

  if (ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif (ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  endif()

  if (ENABLE_CALIPER)
    target_link_libraries(${exe} PRIVATE caliper)
  endif()

  if (ENABLE_RMQ)
    target_link_libraries(${exe} PRIVATE amqpcpp)
    if (OPENSSL_FOUND)
      target_link_libraries(${exe} PRIVATE OpenSSL::SSL OpenSSL::Crypto) 
    endif()
    # NOTE: We set here the event/event pthreads as public. As there is no easy way
    # to do a find package(libevent) and RMQ is not exposing that properly.
    target_link_libraries(${exe} PRIVATE ${LIBEVENT_LIBRARY} ${LIBEVENT_THREAD})
  endif()

  if (ENABLE_MPI)
    target_link_libraries(${exe} PRIVATE MPI::MPI_CXX)
  endif()
endfunction()

# Tests that do NOT require torch
BUILD_UNIT_TEST(amstensor_int amstensor_int.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_CORE_UNIT_TEST(CORE::TENSOR_INT amstensor_int)
BUILD_UNIT_TEST(amstensor_float amstensor_float.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_CORE_UNIT_TEST(CORE::TENSOR_FLOAT amstensor_float)
BUILD_UNIT_TEST(amstensor_mixed amstensor_mixed.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_CORE_UNIT_TEST(CORE::TENSOR_MIXED amstensor_mixed)

# Tests that require torch
# TODO: rewrite some of these tests with AMSTensor
if (ENABLE_TORCH)
  BUILD_UNIT_TEST(tensor_bundle tensor_bundle.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  ADD_CORE_UNIT_TEST(CORE::TENSOR_BUNDLE tensor_bundle)
  BUILD_UNIT_TEST(amstorch_interop amstorch_interop.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  ADD_CORE_UNIT_TEST(CORE::TORCH_INTEROP amstorch_interop)

  if (ENABLE_BENCHMARKS)
    set(_ams_tensor_benchmark_commit "unknown")
    set(_ams_tensor_benchmark_source_state "unknown")
    if (GIT_FOUND)
      execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _ams_tensor_benchmark_git_result
        OUTPUT_VARIABLE _ams_tensor_benchmark_git_output
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if (_ams_tensor_benchmark_git_result EQUAL 0)
        set(_ams_tensor_benchmark_commit
            "${_ams_tensor_benchmark_git_output}")
        execute_process(
          COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          RESULT_VARIABLE _ams_tensor_benchmark_status_result
          OUTPUT_VARIABLE _ams_tensor_benchmark_git_status
          OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (_ams_tensor_benchmark_status_result EQUAL 0)
          if (_ams_tensor_benchmark_git_status)
            set(_ams_tensor_benchmark_source_state "dirty")
          else()
            set(_ams_tensor_benchmark_source_state "clean")
          endif()
        endif()
      endif()
    endif()

    set(_ams_tensor_benchmark_csv
        "${CMAKE_BINARY_DIR}/benchmark-results/amstensor_torch.csv")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark-results")

    add_executable(amstensor_torch_benchmark amstensor_torch_benchmark.cpp)
    target_compile_features(amstensor_torch_benchmark PRIVATE cxx_std_17)
    target_compile_definitions(
      amstensor_torch_benchmark
      PRIVATE CATCH_CONFIG_PREFIX_ALL
              AMS_BENCHMARK_GIT_COMMIT="${_ams_tensor_benchmark_commit}"
              AMS_BENCHMARK_SOURCE_STATE="${_ams_tensor_benchmark_source_state}"
              AMS_TENSOR_BENCHMARK_CSV_DEFAULT="${_ams_tensor_benchmark_csv}")
    target_include_directories(amstensor_torch_benchmark
                               PRIVATE ${CMAKE_BINARY_DIR}/include/
                                       ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(amstensor_torch_benchmark
                          PRIVATE AMS torch Catch2::Catch2)

    add_test(
      NAME PERFORMANCE::AMSTENSOR_TORCH
      COMMAND amstensor_torch_benchmark
              --reporter console
              --benchmark-samples 50
              --benchmark-resamples 1000
              --benchmark-warmup-time 100)
    set_tests_properties(
      PERFORMANCE::AMSTENSOR_TORCH
      PROPERTIES LABELS PERFORMANCE_REGRESSION
                 ENVIRONMENT
                 "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endif()
endif()
