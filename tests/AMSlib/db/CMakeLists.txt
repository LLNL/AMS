function(BUILD_UNIT_TEST exe source)
  blt_add_executable(NAME ${exe} SOURCES ${source})

  target_include_directories(${exe} PRIVATE ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  # DB tests use the AMS-owned main to avoid HIP shutdown finalizer crashes.
  target_sources(${exe} PRIVATE ../ams_catch_main.cpp)
  target_link_libraries(${exe} PRIVATE stdc++fs AMS Catch2::Catch2)
  if (ENABLE_TORCH)
    target_link_libraries(${exe} PRIVATE torch)
  endif()

  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_link_libraries(${exe} PRIVATE Threads::Threads)
  target_link_libraries(${exe} PRIVATE fmt::fmt)
  target_compile_features(${exe} PRIVATE cxx_std_17)

  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})

  if (ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif (ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
  endif()

  if (ENABLE_CALIPER)
    message(STATUS "Building with caliper ${exe}")
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

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})
endfunction()

function(ADD_DB_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS HDF5_UNIT_TEST)
endfunction()

# AMSTensor-only tests (always built)
BUILD_UNIT_TEST(db_hdf5_ams db_hdf5_ams.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_DB_UNIT_TEST(DB::HDF5_AMSTENSOR db_hdf5_ams)

# db_hdf5 test currently uses torch::Tensor for test data generation/validation.

if (ENABLE_TORCH)
  BUILD_UNIT_TEST(db_hdf5_torch db_hdf5_torch.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  ADD_DB_UNIT_TEST(DB::HDF5_TORCH db_hdf5_torch)
endif()
