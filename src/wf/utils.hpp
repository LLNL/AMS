// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_UTILS_HPP__
#define __AMS_UTILS_HPP__

#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <iostream>


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#if __cplusplus < 201402L
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
#else
#endif


const int partitionSize = 1 << 24;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void random_uq_host(bool *uq_flags, int ndata, double acceptable_error) {

  for(int i = 0; i < ndata; i++) {
      uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}


// -----------------------------------------------------------------------------
#endif
