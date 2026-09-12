#pragma once

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <vector>

#include "AMSTensor.hpp"

namespace ams::test
{
using Dim = AMSTensor::IntDimType;

inline std::vector<Dim> contiguousStrides(const std::vector<Dim>& shape)
{
  std::vector<Dim> strides(shape.size(), 1);
  Dim stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename T>
AMSTensor makeTensor(const std::vector<Dim>& shape)
{
  return AMSTensor::create<T>(shape, contiguousStrides(shape), AMS_HOST);
}

template <typename T>
AMSTensor makeTensor(const std::vector<Dim>& shape,
                     const std::vector<T>& values)
{
  AMSTensor tensor = makeTensor<T>(shape);
  CATCH_REQUIRE(values.size() == static_cast<std::size_t>(tensor.elements()));
  std::copy(values.begin(), values.end(), tensor.template data<T>());
  return tensor;
}
}  // namespace ams::test
