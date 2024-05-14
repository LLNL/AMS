# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

BUILD_TEST(ams_allocator_test ams_allocate.cpp)
ADDTEST(ams_allocator_test AMSAllocate)
BUILD_TEST(ams_packing_test cpu_packing_test.cpp AMSPack)
ADDTEST(ams_packing_test AMSPack)

if(WITH_TORCH)
  BUILD_TEST(ams_inference_test torch_model.cpp)
  ADDTEST(ams_inference_test AMSInferDouble ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "double")
  ADDTEST(ams_inference_test AMSInferSingle ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt "single")
  if (WITH_EXAMPLES)
  add_test(NAME AMSExampleSingleDeltaUQ::HOST COMMAND ams_example --precision single --uqtype deltauq-mean -db ./db -S ${CMAKE_CURRENT_SOURCE_DIR}/tuple-single.torchscript -e 100)
  add_test(NAME AMSExampleSingleRandomUQ::HOST COMMAND ams_example --precision single --uqtype random -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
  add_test(NAME AMSExampleDoubleRandomUQ::HOST COMMAND ams_example --precision double --uqtype random -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -e 100)
endif()

  #
  BUILD_TEST(ams_end_to_end ams_ete.cpp)
  # Test EndToEnd Double Precision, Random, All Model
  add_test(NAME AMSEndToEnd::Double::DB::OnlyPhysics::None::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"")
  add_test(NAME AMSEndToEnd::Double::DB::OnlyModel::None::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"none\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"none\" \"./\"")

  # Test EndToEnd Double Precision, Random, All Model CSV
  add_test(NAME AMSEndToEnd::Double::DB::OnlyPhysics::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"csv\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"csv\" \"./\"")
  add_test(NAME AMSEndToEnd::Double::DB::OnlyModel::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"csv\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"csv\" \"./\"")
  add_test(NAME AMSEndToEnd::Double::DB::HALF::CSV::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"csv\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"csv\" \"./\"")


  if (WITH_HDF5)
    add_test(NAME AMSEndToEnd::Double::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Double::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Double::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"double\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"")

    add_test(NAME AMSEndToEnd::Single::DB::OnlyPhysics::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Single::DB::OnlyModel::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 1.0 1 1024 \"hdf5\" \"./\"")
    add_test(NAME AMSEndToEnd::Single::DB::HALF::HDF5::HOST COMMAND bash -c "${CMAKE_CURRENT_BINARY_DIR}/ams_end_to_end 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"; python ${CMAKE_CURRENT_SOURCE_DIR}/verify_ete.py 0 8 9 ${CMAKE_CURRENT_SOURCE_DIR}/linear_scripted_cpu.pt \"float\" \"random\" 0.5 1 1024 \"hdf5\" \"./\"")


  endif()

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

  # The max case fails on DEVICE. We should be aware about this when adding support for CI for GPUs
  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicyDouble ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "double" 2 10 4.0 4 5)

  ADDTEST(ams_hdcache_test AMSHDCacheMeanPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 1 10 4.0 4 5)

  # The max case fails on DEVICE. We should be aware about this when adding support for CI for GPUs
  ADDTEST(ams_hdcache_test AMSHDCacheMaxPolicySingle ${CMAKE_CURRENT_SOURCE_DIR}/faiss_debug.pt "single" 2 10 4.0 4 5)

  if(WITH_TORCH)
    if (WITH_EXAMPLES)
    add_test(NAME AMSExampleFaissInferSingle::HOST COMMAND ams_example --precision single --uqtype faiss-mean -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
    add_test(NAME AMSExampleFaissInferDouble::HOST COMMAND ams_example --precision double --uqtype faiss-mean -S ${CMAKE_CURRENT_SOURCE_DIR}/debug_model.pt -H ${CMAKE_CURRENT_SOURCE_DIR}/example_faiss.idx -e 100)
    endif()
  endif()
endif()
