
function(ADD_TORCH_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS TORCH_UNIT_TEST)
endfunction()


function(BUILD_UNIT_TEST exe source)
  blt_add_executable(NAME ${exe} SOURCES ${source})
  target_compile_features(${exe} PRIVATE cxx_std_17)
  if (ENABLE_CUDA)
    target_link_libraries(${exe} PRIVATE CUDA::cudart)
  elseif (ENABLE_HIP)
    target_link_libraries(${exe} PRIVATE hip::host)
    target_link_libraries(${exe} PRIVATE Threads::Threads)
  endif()

  target_link_libraries(${exe} PRIVATE fmt::fmt)
  target_link_libraries(${exe} PRIVATE nlohmann_json::nlohmann_json)
  
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
  target_link_libraries(${exe} PRIVATE stdc++fs AMS torch ${catch2_target})
  target_link_libraries(${exe} PRIVATE tl::expected)
  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)
endfunction()


BUILD_UNIT_TEST(ams_surrogate_tests evaluate_model.cpp Catch2::Catch2)
ADD_TORCH_UNIT_TEST(Surrogate ams_surrogate_tests)

BUILD_UNIT_TEST(ams_test_base_model test_model.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_TORCH_UNIT_TEST(BaseModel ams_test_base_model)

BUILD_UNIT_TEST(ams_test_inference_model test_inference_model.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_TORCH_UNIT_TEST(InferenceModel ams_test_inference_model)
