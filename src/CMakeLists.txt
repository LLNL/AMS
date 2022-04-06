
add_executable(miniAPP main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(miniAPP PUBLIC "--expt-extended-lambda")

target_compile_definitions(miniAPP PUBLIC ${AMS_DEFINES})

target_include_directories(miniAPP PRIVATE ${MFEM_INCLUDE_DIRS} ${AMS_APP_INCLUDE})
target_link_libraries(miniAPP PUBLIC ${CUDA_LIBRARIES} ${MFEM_LIBRARIES} ${AMS_APP_LIBRARIES} cuda)
