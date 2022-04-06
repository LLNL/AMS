
add_executable(miniAPP main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(miniAPP PUBLIC "--expt-extended-lambda")

if(WITH_DB)
  target_compile_definitions(miniAPP PUBLIC -D__ENABLE_DB__)
endif()

target_include_directories(miniAPP PRIVATE ${MFEM_INCLUDE_DIRS} ${AMS_APP_INCLUDE})
target_link_libraries(miniAPP PUBLIC ${CUDA_LIBRARIES} ${MFEM_LIBRARIES} ${AMS_APP_LIBRARIES} cuda)
