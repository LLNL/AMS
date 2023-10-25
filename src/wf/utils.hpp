/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

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

template<typename T>
inline bool is_real_equal(T l, T r)
{
  return r == std::nextafter(l, r);
}

// -----------------------------------------------------------------------------
#endif
