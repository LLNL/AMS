/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/utils.hpp"

void random_uq_host(bool *uq_flags, int ndata, double acceptable_error)
{

  for (int i = 0; i < ndata; i++) {
    uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}
