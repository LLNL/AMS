# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(GLOB_RECURSE pyfiles *.py)
foreach (filename ${pyfiles})
    get_filename_component(target "${filename}" NAME)
    message(STATUS "Copying ${filename} to ${target}")
    configure_file("${filename}" "${CMAKE_CURRENT_BINARY_DIR}/${target}" COPYONLY)
endforeach (filename)

