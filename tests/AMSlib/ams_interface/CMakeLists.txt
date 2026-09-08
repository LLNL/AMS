function(ADD_AMS_UNIT_TEST name exec)
  add_test(NAME ${name} COMMAND ${exec}  -s --reporter console)
  set_tests_properties(${name} PROPERTIES LABELS AMS_INTERFACE)
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

  if (ENABLE_MPI)
    target_link_libraries(${exe} PRIVATE MPI::MPI_CXX)
  endif()

  if (ENABLE_CALIPER)
    target_link_libraries(${exe} PRIVATE caliper)
    target_include_directories(${exe} PRIVATE ${caliper_INCLUDE_DIR})
  endif()

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
  target_link_libraries(${exe} PRIVATE ${AMS_HDF5_LINK_TARGETS})
  target_compile_definitions(${exe} PRIVATE ${AMS_APP_DEFINES} CATCH_CONFIG_PREFIX_ALL)
endfunction()

BUILD_UNIT_TEST(ams_explicit_end_to_end ams_ete.cpp Catch2::Catch2)
ADD_AMS_UNIT_TEST(AMS_EXPLICIT ams_explicit_end_to_end)

BUILD_UNIT_TEST(int_interface int_interface.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_INT_INTERFACE int_interface)

BUILD_UNIT_TEST(ams_graph_fallback test_graph_fallback.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_GRAPH_FALLBACK ams_graph_fallback)

BUILD_UNIT_TEST(ams_graph_surrogate test_graph_surrogate.cpp Catch2::Catch2 ../ams_catch_main.cpp)
ADD_AMS_UNIT_TEST(AMS_GRAPH_SURROGATE ams_graph_surrogate)

BUILD_UNIT_TEST(ams_graph_workflow_storage test_graph_workflow_storage.cpp Catch2::Catch2 ../ams_catch_main.cpp)
target_link_libraries(ams_graph_workflow_storage PRIVATE nlohmann_json::nlohmann_json)
ADD_AMS_UNIT_TEST(AMS_GRAPH_WORKFLOW_STORAGE ams_graph_workflow_storage)

BUILD_UNIT_TEST(ams_graph_mgn_surrogate test_graph_mgn_surrogate.cpp Catch2::Catch2 ../ams_catch_main.cpp)
target_link_libraries(ams_graph_mgn_surrogate PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(ams_graph_mgn_surrogate
  PRIVATE
    AMS_MGN_DIFFUSION_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models/mgn_graph_diffusion"
)

# Normal parity testing consumes the checked-in TorchScript model and
# self-contained JSON fixtures; training/export is an independent opt-in path.
add_test(
  NAME MGN_DIFFUSION_AMS_PARITY
  COMMAND $<TARGET_FILE:ams_graph_mgn_surrogate> -s --reporter console
)
set_tests_properties(MGN_DIFFUSION_AMS_PARITY PROPERTIES
  LABELS "AMS_INTERFACE;MGN_DIFFUSION"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(NOT Torch_FOUND)
  set_tests_properties(MGN_DIFFUSION_AMS_PARITY PROPERTIES DISABLED TRUE)
endif()
