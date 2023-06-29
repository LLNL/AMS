// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_UTILS_HPP__
#define __AMS_UTILS_HPP__

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#if __cplusplus < 201402L
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
#else
#endif

template <typename T>
class isDouble
{
public:
  static constexpr bool default_value() { return false; }
};

template <>
class isDouble<double>
{
public:
  static constexpr bool default_value() { return true; }
};

template <>
class isDouble<float>
{
public:
  static constexpr bool default_value() { return false; }
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void random_uq_host(bool *uq_flags, int ndata, double acceptable_error)
{

  for (int i = 0; i < ndata; i++) {
    uq_flags[i] = ((double)rand() / RAND_MAX) <= acceptable_error;
  }
}


// -----------------------------------------------------------------------------
#endif
