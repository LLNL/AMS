function(ADD_WORKFLOW_UNIT_TEST name exec)
  string(JOIN " " args ${ARGN})
  add_test(NAME ${name} COMMAND ${exec} -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS WORKFLOW_UNIT_TEST)
endfunction()

function(BUILD_UNIT_TEST exe source)
  blt_add_executable(NAME ${exe} SOURCES ${source})

  target_compile_features(${exe} PRIVATE cxx_std_17)
  target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src/AMSlib/)
  target_include_directories(${exe} PRIVATE ${CMAKE_BINARY_DIR}/include/)
  target_include_directories(${exe} PRIVATE ${AMS_TEST_ROOT})
  target_include_directories(${exe} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set(catch2_target Catch2::Catch2WithMain)
  # Optional third argument lets tests provide their own main.
  if (${ARGC} GREATER 2)
    set(catch2_target ${ARGV2})
  endif()
  # Optional fourth argument adds that custom main source when needed.
  if (${ARGC} GREATER 3)
    target_sources(${exe} PRIVATE ${ARGV3})
  endif()
  target_link_libraries(${exe} PRIVATE stdc++fs AMS ${catch2_target})
  if (ENABLE_TORCH)
    target_link_libraries(${exe} PRIVATE torch)
  endif()
  target_link_libraries(${exe} PRIVATE tl::expected)
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)

  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})


  if (ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif (ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
    target_link_libraries(${exe} PRIVATE Threads::Threads)
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
BUILD_UNIT_TEST(action action.cpp Catch2::Catch2 ../ams_catch_main.cpp)
target_link_libraries(action PRIVATE fmt::fmt)
ADD_WORKFLOW_UNIT_TEST(WORKFLOW::ACTION action)


if (ENABLE_TORCH)
  BUILD_UNIT_TEST(operations operations.cpp Catch2::Catch2)
  target_link_libraries(operations PRIVATE fmt::fmt)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::OPERATIONS operations)

  BUILD_UNIT_TEST(evaluate_in_and_outs evaluate_in_and_outs.cpp Catch2::Catch2)
  target_link_libraries(evaluate_in_and_outs PRIVATE fmt::fmt)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::EVALUATE_IN_OUTS evaluate_in_and_outs)

  BUILD_UNIT_TEST(tensor_bundle tensor_bundle.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::TENSOR_BUNDLE tensor_bundle)

  BUILD_UNIT_TEST(eval_context eval_context.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::EVAL_CONTEXT eval_context)

  BUILD_UNIT_TEST(pipeline pipeline.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  target_link_libraries(pipeline PRIVATE fmt::fmt)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::PIPELINE pipeline)

  BUILD_UNIT_TEST(pointwise pointwise_layout_transform.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  target_link_libraries(pointwise PRIVATE fmt::fmt)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::POINTWISE pointwise)

  BUILD_UNIT_TEST(policy policy.cpp Catch2::Catch2 ../ams_catch_main.cpp)
  target_link_libraries(policy PRIVATE fmt::fmt)
  ADD_WORKFLOW_UNIT_TEST(WORKFLOW::POLICY policy)
endif()
