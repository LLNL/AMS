# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(ideal_gas)
if (WITH_CUDA)
add_subdirectory(bnm_opt)
endif()

