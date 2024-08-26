# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(JSON_TESTS db_type)
  set(FS_PATH "${CMAKE_CURRENT_BINARY_DIR}")
  set(AMS_DB_TEST_TYPE ${db_type})
  set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/${db_type}.json")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/json_configs/env_2_models_fs_rand_uq.json.in" "${JSON_FP}" @ONLY)
  
  # Tests Random models with different percentages both models store to file
  add_test(NAME AMSEndToEndFromJSON::Random10::Random50::Double::DB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_random_10 app_random_50;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_random_10 app_random_50")


  # Tests delta-uq models with different aggregation both models store to file
  add_test(NAME AMSEndToEndFromJSON::DuqMean::DuqMax::Double::DB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_uq_mean app_uq_max;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_uq_mean app_uq_max")


  # Tests detla uq model with a random uq model both models store to files
  add_test(NAME AMSEndToEndFromJSON::Random::DuqMax::Double::DB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_random_10 app_uq_max;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_random_10 app_uq_max")

  # Tests detla uq model with no model. uq model both store to files
  add_test(NAME AMSEndToEndFromJSON::Random::NoModel::Double::DB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_random_10 app_no_model;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_random_10 app_no_model")

  # Tests 2 delta uq models with no deb . uq model both store to files
  add_test(NAME AMSEndToEndFromJSON::DuqMean::DuqMax::Double::NODB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_uq_mean_ndb app_uq_max_ndb;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_uq_mean_ndb app_uq_max_ndb")

  # Tests null models null dbs 
  add_test(NAME AMSEndToEndFromJSON::None::None::Double::NODB::${db_type}::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_no_model_no_db  app_no_model_no_db ;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_no_model_no_db app_no_model_no_db")

  unset(AMS_DB_TEST_TYPE)
  unset(JSON_FP)
endfunction()

function(CHECK_RMQ_CONFIG file)
  # Read the JSON file.
  file(READ ${file} MY_JSON_STRING)
  message(STATUS "RabbitMQ config ${file}")

  string(JSON DB_CONF GET ${MY_JSON_STRING} db)
  string(JSON DB_CONF GET ${DB_CONF} rmq_config)
  string(JSON RMQ_HOST GET ${DB_CONF} "service-host")
  string(JSON RMQ_PORT GET ${DB_CONF} "service-port")

  if(NOT "${RMQ_HOST}" STREQUAL "" AND NOT "${RMQ_PORT}" STREQUAL "0")
  message(STATUS "RabbitMQ config ${file}: ${RMQ_HOST}:${RMQ_PORT}")
  else()
    message(WARNING "RabbitMQ config file ${file} looks empty! Make sure to fill these fields before running the tests")
  endif()
endfunction()

function(INTEGRATION_TEST_ENV)
  JSON_TESTS("csv")
  if (WITH_HDF5)
  JSON_TESTS("hdf5")
  set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/hdf5.json")
  # Tests delta-uq models with different aggregation both models store to file with debug option set to on.
  add_test(NAME AMSEndToEndFromJSON::DuqMean::DuqMax::Double::DB::hdf5-debug::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end_env  0 8 9 \"double\" 1 1024 app_uq_mean_debug app_uq_max_debug;AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 \"double\" 1024 app_uq_mean_debug app_uq_max_debug")
  unset(JSON_FP)
  endif()
endfunction()

function(INTEGRATION_TEST_RMQ)
  if (WITH_RMQ)
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
      # If file exists we do not overwrite it
      message(STATUS "Ctest will use ${CMAKE_CURRENT_BINARY_DIR}/rmq.json as RabbitMQ configuration for testing. Make sure RabbitMQ parameters are valid.")
    else()
      message(STATUS "Copying empty configuration to ${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/json_configs/rmq.json.in" "rmq.json" @ONLY)
    endif()
    set(JSON_FP "${CMAKE_CURRENT_BINARY_DIR}/rmq.json")
    CHECK_RMQ_CONFIG(${JSON_FP})
    add_test(NAME AMSEndToEndFromJSON::NoModel::Double::DB::rmq::HOST COMMAND bash -c "AMS_OBJECTS=${JSON_FP} ${CMAKE_CURRENT_BINARY_DIR}/ams_rmq 0 8 9 \"double\" 2 1024; AMS_OBJECTS=${JSON_FP} python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_rmq.py 0 8 9 \"double\" 2 1024")
  endif()
endfunction()

function (INTEGRATION_TEST)
  #######################################################
  # TEST: output format 
  # UQ: Random
  #######################################################

  add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyPhysics::None::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"")
  add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyModel::None::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"none\" \"./\"")


  #######################################################
  # TEST: CSV output format 
  # UQ: Random
  # 3 Thresholds: 0, 0.5, 1.0
  #######################################################

  add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyPhysics::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"csv\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"csv\" \"./\"")
  add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyModel::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"csv\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"csv\" \"./\"")
  add_test(NAME AMSEndToEnd::Random::Double::DB::HALF::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"csv\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"csv\" \"./\"")


  if (WITH_HDF5)
  #######################################################
  # TEST: hdf5 output format 
  # UQ: Random
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################

    add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Random::Double::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Random::Double::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: Random
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################


    add_test(NAME AMSEndToEnd::Random::Single::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Random::Single::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Random::Single::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"")
  endif()


  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-mean
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################
  add_test(NAME AMSEndToEnd::DeltaUQMean::Double::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 0.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMean::Double::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 1.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMean::Double::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_double_cpu.pt \"double\" \"deltaUQ (mean)\" 0.5 1 1024 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-mean
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################
  add_test(NAME AMSEndToEnd::DeltaUQMean::Single::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 0.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMean::Single::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 1.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMean::Single::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_mean_single_cpu.pt \"float\" \"deltaUQ (mean)\" 0.5 1 1024 \"hdf5\" \"./\"")

  #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-max
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: double
  #######################################################
  add_test(NAME AMSEndToEnd::DeltaUQMax::Double::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 0.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMax::Double::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 1.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMax::Double::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_double_cpu.pt \"double\" \"deltaUQ (max)\" 0.5 1 1024 \"hdf5\" \"./\"")

 #######################################################
  # TEST: hdf5 output format 
  # UQ: deltauq-max
  # 3 Thresholds: 0, 0.5, 1.0
  # precision: single
  #######################################################
  add_test(NAME AMSEndToEnd::DeltaUQMax::Single::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 0.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 0.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMax::Single::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 1.0 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 1.0 1 1024 \"hdf5\" \"./\"")
  add_test(NAME AMSEndToEnd::DeltaUQMax::Single::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 0.5 1 1024 \"hdf5\" \"./\"; python3 ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/uq_max_single_cpu.pt \"float\" \"deltaUQ (max)\" 0.5 1 1024 \"hdf5\" \"./\"")
endfunction()



function(BUILD_TEST exe source)
  add_executable(${exe} ${source})
  target_include_directories(${exe} PRIVATE "${PROJECT_SOURCE_DIR}/src/AMSlib/" umpire ${caliper_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
  target_link_directories(${exe} PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(${exe} PRIVATE AMS ${AMS_APP_LIBRARIES})

  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES})

  if(WITH_CUDA)
    set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${AMS_CUDA_ARCH}")
    set_property(TARGET ${exe} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)

    target_compile_definitions(${exe} PRIVATE "-D__ENABLE_CUDA__ -DLIBAMS_VERBOSE")
  endif()
endfunction()

function(ADDTEST exe test_name)
  add_test(NAME "${test_name}::HOST" COMMAND ${exe} 0 ${ARGN})

  if(WITH_CUDA)
    add_test(NAME "${test_name}::DEVICE" COMMAND ${exe} 1 ${ARGN})
  endif()
endfunction()

# This test requires Allocate
# TODO: Include tests once we re-instate a pool
#BUILD_TEST(ams_allocator_test ams_allocate.cpp)
#ADDTEST(ams_allocator_test AMSAllocate)
BUILD_TEST(ams_packing_test cpu_packing_test.cpp AMSPack)
ADDTEST(ams_packing_test AMSPack)

if(WITH_TORCH)
  BUILD_TEST(ams_inference_test torch_model.cpp)
  ADDTEST(ams_inference_test AMSInferDouble ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "double")
  ADDTEST(ams_inference_test AMSInferSingle ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "single")
  if (WITH_EXAMPLES)
  add_test(NAME AMSExampleSingleDeltaUQ::HOST COMMAND ams_example --precision single --uqtype deltauq-mean -db "./" -S ${CMAKE_CURRENT_SOURCE_DIR}/tuple-single.torchscript -e 100)
  add_test(NAME AMSExampleSingleRandomUQ::HOST COMMAND ams_example --precision single --uqtype random -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
  add_test(NAME AMSExampleDoubleRandomUQ::HOST COMMAND ams_example --precision double --uqtype random -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
endif()

# These are integration tests. Try to use an end to end AMS run and realize issues between multiple pieces
  BUILD_TEST(ams_end_to_end ams_ete.cpp)
  INTEGRATION_TEST()
  BUILD_TEST(ams_end_to_end_env ams_ete_env.cpp)
  INTEGRATION_TEST_ENV()
  BUILD_TEST(ams_rmq ams_rmq_env.cpp)
  INTEGRATION_TEST_RMQ()


  # UQ Tests
  BUILD_TEST(ams_delta_uq_test ams_uq_model.cpp)

  if(WITH_TORCH)
    add_test(NAME AMSDeltaUQDoubleMean::HOST COMMAND ams_delta_uq_test 0 ${CMAKE_CURRENT_SOURCE_DIR}/torch.duq.cuda "double" 8 9 3 0.0)
    add_test(NAME AMSDeltaUQDoubleMax::HOST COMMAND ams_delta_uq_test 0 ${CMAKE_CURRENT_SOURCE_DIR}/torch.duq.cuda "double" 8 9 4 0.0)
    add_test(NAME AMSDeltaUQDoubleMean_2::HOST COMMAND ams_delta_uq_test 0 ${CMAKE_CURRENT_SOURCE_DIR}/tuple.duq "double" 2 4 3 0.5)
    add_test(NAME AMSDeltaUQDoubleMax_2::HOST COMMAND ams_delta_uq_test 0 ${CMAKE_CURRENT_SOURCE_DIR}/tuple.duq "double" 2 4 4 0.1)

    if(WITH_CUDA)
      add_test(NAME AMSDeltaUQDoubleMean::DEVICE COMMAND ams_delta_uq_test 1 ${CMAKE_CURRENT_SOURCE_DIR}/torch.duq.cuda "double" 8 9 3 0.0)
      add_test(NAME AMSDeltaUQDoubleMax::DEVICE COMMAND ams_delta_uq_test 1 ${CMAKE_CURRENT_SOURCE_DIR}/torch.duq.cuda "double" 8 9 4 0.0)
      add_test(NAME AMSDeltaUQDoubleMean_2::DEVICE COMMAND ams_delta_uq_test 1 ${CMAKE_CURRENT_SOURCE_DIR}/tuple.duq.cuda "double" 2 4 3 0.5)
      add_test(NAME AMSDeltaUQDoubleMax_2::DEVICE COMMAND ams_delta_uq_test 1 ${CMAKE_CURRENT_SOURCE_DIR}/tuple.duq.cuda "double" 2 4 4 0.1)
    endif()
  endif()

  # TODO Add tests with cpu model
  BUILD_TEST(ams_update_model ams_update_model.cpp)
  ADDTEST(ams_update_model AMSUpdateModelDouble "double" ${CMAKE_CURRENT_SOURCE_DIR}/ConstantZeroModel_cpu.pt ${CMAKE_CURRENT_SOURCE_DIR}/ConstantOneModel_cpu.pt)
endif()

if(WITH_FAISS)
  BUILD_TEST(ams_hdcache_test test_hdcache.cpp)
  ADDTEST(ams_hdcache_test AMSHDCacheMeanPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 1 10 4.0 4 5)

  ADDTEST(ams_hdcache_test AMSHDCacheMeanPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 1 10 4.0 4 5)

  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 2 10 4.0 4 5)
  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 2 10 4.0 4 5)
  # The max case fails on DEVICE. We should be aware about this when adding support for CI for GPUs
  if (WITH_CUDA)
    set_tests_properties(AMSHDCacheMaxPolicySingle::DEVICE AMSHDCacheMaxPolicyDouble::DEVICE PROPERTIES DISABLED TRUE)
  endif()

  if(WITH_TORCH)
    if (WITH_EXAMPLES)
    add_test(NAME AMSExampleFaissInferSingle::HOST COMMAND ams_example --precision single --uqtype faiss-mean -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
    add_test(NAME AMSExampleFaissInferDouble::HOST COMMAND ams_example --precision double --uqtype faiss-mean -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
    endif()
  endif()
endif()
