/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_UTILS_HPP__
#define __AMS_UTILS_HPP__

#ifdef __AMS_ENABLE_TORCH__
#include <ATen/core/TensorBody.h>
#endif

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>
#include <sstream>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "SmallVector.hpp"

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

template <typename T>
inline bool is_real_equal(T l, T r)
{
  return r == std::nextafter(l, r);
}


static inline size_t dtype_to_size(ams::AMSDType dType)
{
  switch (dType) {
    case ams::AMSDType::AMS_DOUBLE:
      return sizeof(double);
    case ams::AMSDType::AMS_SINGLE:
      return sizeof(float);
    case ams::AMSDType::AMS_INT64:
      return sizeof(int64_t);
    case ams::AMSDType::AMS_INT32:
      return sizeof(int32_t);
    default:
      throw std::runtime_error("Requesting the size of unknown object");
  }
}

static inline std::string shapeToString(const ams::AMSTensor& tensor)
{
  std::ostringstream oss;
  oss << "[";
  auto shape = tensor.sizes();
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) oss << ", ";
  }
  oss << "]";
  return oss.str();
}

#ifdef __AMS_ENABLE_TORCH__
static inline std::string shapeToString(const at::Tensor& tensor)
{
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}
#endif  // __AMS_ENABLE_TORCH__

namespace ams
{
namespace tensor
{
SmallVector<at::Tensor> maskTensor(at::Tensor& Src, at::Tensor& Mask);
}  // namespace tensor

}  // namespace ams

template <>
struct fmt::formatter<ams::AMSResourceType> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const ams::AMSResourceType& t, FormatContext& ctx) const
  {
    std::string_view name;

    switch (t) {
      case ams::AMSResourceType::AMS_UNKNOWN:
        name = "MemResource::unknown";
        break;
      case ams::AMSResourceType::AMS_DEVICE:
        name = "MemResource::Device";
        break;
      case ams::AMSResourceType::AMS_PINNED:
        name = "MemResource::Pinned";
        break;
      case ams::AMSResourceType::AMS_HOST:
        name = "MemResource::Host";
        break;
      default:
        name = "UNKNOWN";
        break;
    }

    return formatter<std::string_view>::format(name, ctx);
  }
};


#endif
