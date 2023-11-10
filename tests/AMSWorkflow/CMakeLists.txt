# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Python COMPONENTS Interpreter REQUIRED)

file(GLOB PY_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.py)
foreach(X IN LISTS PY_TEST_FILES)
    string(REPLACE  "${CMAKE_CURRENT_SOURCE_DIR}/test_" "" TEST_NAME ${X})
    string(REPLACE  ".py" "" TEST_NAME ${TEST_NAME})
    add_test (NAME ${TEST_NAME} COMMAND ${Python_EXECUTABLE} ${X})
endforeach()

#add_test (NAME python-tests
#  COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
#  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/mypyspace
#)
