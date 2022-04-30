
# handle sources and headers
file(GLOB AMS_INCLUDES "*.hpp")
set(AMS_LIB_SRC mmp.cpp)
set(AMS_EXE_SRC main.cpp)

# two targets: a shared lib and an exec
add_library(mmp SHARED ${AMS_LIB_SRC} ${AMS_INCLUDES})
add_executable(mmpbin ${AMS_EXE_SRC} ${AMS_INCLUDES})

if (WITH_CUDA)
    set_source_files_properties(mmp.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(mmp PUBLIC "--expt-extended-lambda")
    target_compile_options(mmpbin PUBLIC "--expt-extended-lambda")
endif()

link_directories(${CMAKE_CURRENT_BINARY_DIR})

#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#      # using regular Clang or AppleClang
#else()
#endif()


# setup the lib first
target_compile_definitions(mmp PUBLIC ${AMS_DEFINES})
target_include_directories(mmp PRIVATE ${MFEM_INCLUDE_DIRS} ${FAISS_INCLUDE_DIRS} ${AMS_APP_INCLUDE})
target_link_directories(mmp PRIVATE ${FAISS_LIB_DIR})
target_link_libraries(mmp PUBLIC ${MFEM_LIBRARIES} ${FAISS_LIBRARIES} ${AMS_APP_LIBRARIES})


# setup the exec
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

target_compile_definitions(mmpbin PUBLIC ${AMS_DEFINES})
target_include_directories(mmpbin PRIVATE ${MFEM_INCLUDE_DIRS} ${FAISS_INCLUDE_DIRS} ${AMS_APP_INCLUDE})
target_link_directories(mmpbin PRIVATE ${FAISS_LIB_DIR} ${MFEM_LIB_DIR})
target_link_libraries(mmpbin PUBLIC ${MFEM_LIBRARIES} ${FAISS_LIBRARIES} ${AMS_APP_LIBRARIES} mmp)

# rename the exec target to "mmp"
set_target_properties(mmpbin PROPERTIES OUTPUT_NAME mmp)


# installation paths
install(TARGETS mmp DESTINATION lib)
install(TARGETS mmpbin DESTINATION bin)
