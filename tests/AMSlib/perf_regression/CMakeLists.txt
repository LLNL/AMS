function(BUILD_TEST exe source)
  blt_add_executable(NAME ${exe} SOURCES ${source})
  target_include_directories(${exe} PRIVATE "${PROJECT_SOURCE_DIR}/src/AMSlib/" "${PROJECT_SOURCE_DIR}/src/AMSlib/include" ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
  message("On test defines are ${AMS_APP_DEFINES}")
  target_link_libraries(${exe} PRIVATE AMS)
  if (ENABLE_TORCH)
    target_link_libraries(${exe} PRIVATE torch)
  endif()
  target_link_libraries(${exe} PRIVATE Threads::Threads)
  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})
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


# AMS Database benchmark (RMQ and/or HDF5 + MPI / No ML models used)
BUILD_TEST(ams_benchmark_db ams_bench_db.cpp)
# The AMS DB Benchmark requires mfem
# TODO: Remove mfem requirement from the benchmark
target_link_libraries(ams_benchmark_db PRIVATE AMS ${AMS_EXAMPLE_LIBRARIES})
