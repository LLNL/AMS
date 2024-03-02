# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (WITH_AMS_LIB)
  add_subdirectory(AMSlib)
endif()

if (WITH_WORKFLOW)
  add_subdirectory(AMSWorkflow)
endif()
