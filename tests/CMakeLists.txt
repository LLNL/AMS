
link_directories(${AMS_LIB_OUT_PATH})

add_executable(ams_allocator ams_allocate.cpp)
target_include_directories(ams_allocator PRIVATE "${PROJECT_SOURCE_DIR}/miniapp" ${AMS_APP_INCLUDES})
target_link_libraries(ams_allocator PUBLIC ${AMS_APP_LIBRARIES} miniapp)
add_test(NAME AMSHostAllocate COMMAND ams_allocator host) 

add_executable(cpu_packing cpu_packing_test.cpp)
target_include_directories(cpu_packing PRIVATE "${PROJECT_SOURCE_DIR}/miniapp" ${AMS_APP_INCLUDES})
target_include_directories(cpu_packing PRIVATE ${AMS_APP_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(cpu_packing PRIVATE ${AMS_APP_LIB_DIRS})
target_link_libraries(cpu_packing PUBLIC ${AMS_APP_LIBRARIES} miniapp)
add_test(NAME CPUPacking COMMAND cpu_packing 0) 
add_test(NAME CPUPackingIndexing COMMAND cpu_packing 1) 

if (WITH_CUDA)
  set_source_files_properties(cpu_packing_test.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(cpu_packing_test.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

  add_test(NAME AMSDeviceAllocate COMMAND ams_allocator device) 
  add_executable(gpu_packing gpu_packing_test.cpp)
  
  set_source_files_properties(gpu_packing_test.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(gpu_packing_test.cpp PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")
  target_include_directories(gpu_packing PRIVATE "${PROJECT_SOURCE_DIR}/miniapp" ${AMS_APP_INCLUDES})
  target_include_directories(gpu_packing PRIVATE ${AMS_APP_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_directories(gpu_packing PRIVATE ${AMS_APP_LIB_DIRS})
  target_link_libraries(gpu_packing PUBLIC ${AMS_APP_LIBRARIES} miniapp)
  add_test(NAME GPUPacking COMMAND gpu_packing 0) 
  add_test(NAME GPUPackingIndexing COMMAND gpu_packing 1) 

endif()

