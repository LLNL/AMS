/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

// =========================================================================
// float — create
// =========================================================================

CATCH_TEST_CASE("float: create 1D tensor", "[ams][tensor][float][create]")
{
  AMSInit();
  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);
  if (device == AMSResourceType::AMS_DEVICE) {
#if !defined(__AMS_ENABLE_CUDA__) && !defined(__AMS_ENABLE_HIP__)
    CATCH_SKIP("GPU device not available");
#endif
  }

  std::vector<AMSTensor::IntDimType> shape = {8};
  std::vector<AMSTensor::IntDimType> strides = {1};
  
  auto tensor = AMSTensor::create<float>(shape, strides, device);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(tensor.elements() == 8);
  CATCH_REQUIRE(tensor.element_size() == sizeof(float));
  CATCH_REQUIRE(tensor.dim() == 1);
  CATCH_REQUIRE(tensor.nbytes() == 8 * sizeof(float));
  CATCH_REQUIRE(tensor.location() == device);
  CATCH_REQUIRE(tensor.shape()[0] == 8);
}


CATCH_TEST_CASE("float: create 2D tensor", "[ams][tensor][float][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {4, 6};
  std::vector<AMSTensor::IntDimType> strides = {6, 1};

  auto tensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(tensor.elements() == 24);
  CATCH_REQUIRE(tensor.dim() == 2);
  CATCH_REQUIRE(tensor.shape()[0] == 4);
  CATCH_REQUIRE(tensor.shape()[1] == 6);
  CATCH_REQUIRE(tensor.nbytes() == 24 * sizeof(float));
}


CATCH_TEST_CASE("float: create 3D tensor", "[ams][tensor][float][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {2, 3, 5};
  std::vector<AMSTensor::IntDimType> strides = {15, 5, 1};

  auto tensor = AMSTensor::create<float>(
      shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(tensor.elements() == 30);
  CATCH_REQUIRE(tensor.dim() == 3);
}


// =========================================================================
// float — view
// =========================================================================

CATCH_TEST_CASE("float: view 1D from existing buffer",
                "[ams][tensor][float][view]")
{
  AMSInit();
  std::vector<float> data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
  std::vector<AMSTensor::IntDimType> shape = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto v = AMSTensor::view<float>(
      data.data(), shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(v.elements() == 5);
  CATCH_REQUIRE(v.data<float>() == data.data());
  CATCH_REQUIRE(v.data<float>()[0] == 1.1f);
  CATCH_REQUIRE(v.data<float>()[4] == 5.5f);
}


CATCH_TEST_CASE("float: view 2D from existing buffer",
                "[ams][tensor][float][view]")
{
  AMSInit();
  std::vector<float> data(12, 3.14f);
  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto v = AMSTensor::view<float>(
      data.data(), shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.elements() == 12);
  CATCH_REQUIRE(v.shape()[0] == 3);
  CATCH_REQUIRE(v.shape()[1] == 4);
  CATCH_REQUIRE(v.data<float>()[11] == 3.14f);
}


CATCH_TEST_CASE("float: view from AMSTensor alias",
                "[ams][tensor][float][view]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {4};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = tensor.data<float>();
  for (int i = 0; i < 4; ++i) ptr[i] = static_cast<float>(i + 1);

  auto alias = AMSTensor::view(tensor);

  CATCH_REQUIRE(alias.data<float>() == ptr);
  CATCH_REQUIRE(alias.elements() == 4);
  CATCH_REQUIRE(alias.data<float>()[3] == 4.0f);
}


// =========================================================================
// float — data access
// =========================================================================

CATCH_TEST_CASE("float: write and read 1D data",
                "[ams][tensor][float][data]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {4};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<float>();

  data[0] = 0.1f;
  data[1] = 0.2f;
  data[2] = 0.3f;
  data[3] = 0.4f;

  CATCH_REQUIRE(data[0] == 0.1f);
  CATCH_REQUIRE(data[3] == 0.4f);
}


CATCH_TEST_CASE("float: write and read 2D data",
                "[ams][tensor][float][data]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto tensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<float>();

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      data[i * 4 + j] = static_cast<float>(i) + static_cast<float>(j) * 0.1f;

  CATCH_REQUIRE(data[0] == 0.0f);          // [0,0]
  CATCH_REQUIRE(data[3] == 0.3f);          // [0,3]
  CATCH_REQUIRE(std::fabs(data[5] - 1.1f) < 1e-6f);  // [1,1]
}


// =========================================================================
// float — transpose
// =========================================================================

CATCH_TEST_CASE("float: transpose 2D tensor",
                "[ams][tensor][float][transpose]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {3, 5};
  std::vector<AMSTensor::IntDimType> strides = {5, 1};

  auto tensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = tensor.transpose(0, 1);

  CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(transposed.shape()[0] == 5);
  CATCH_REQUIRE(transposed.shape()[1] == 3);
  CATCH_REQUIRE(transposed.elements() == 15);
  CATCH_REQUIRE(!transposed.contiguous());
}


// =========================================================================
// float — move
// =========================================================================

CATCH_TEST_CASE("float: move constructor", "[ams][tensor][float][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {12};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto t1 = AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = t1.data<float>();

  auto t2 = std::move(t1);

  CATCH_REQUIRE(t2.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(t2.elements() == 12);
  CATCH_REQUIRE(t2.dim() == 1);
  CATCH_REQUIRE(t2.nbytes() == 12 * sizeof(float));
  CATCH_REQUIRE(t2.data<float>() == ptr);
}


CATCH_TEST_CASE("float: move assignment", "[ams][tensor][float][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape1 = {10};
  std::vector<AMSTensor::IntDimType> strides = {1};
  std::vector<AMSTensor::IntDimType> shape2 = {3};

  auto t1 = AMSTensor::create<float>(shape1, strides, AMSResourceType::AMS_HOST);
  auto t2 = AMSTensor::create<float>(shape2, strides, AMSResourceType::AMS_HOST);
  auto* ptr1 = t1.data<float>();

  t2 = std::move(t1);

  CATCH_REQUIRE(t2.elements() == 10);
  CATCH_REQUIRE(t2.data<float>() == ptr1);
}


// =========================================================================
// float — clone
// =========================================================================

CATCH_TEST_CASE("float: clone contiguous 1D tensor",
                "[ams][tensor][float][clone]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  auto view = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(cloned.elements() == 5);
  CATCH_REQUIRE(cloned.dim() == 1);
  CATCH_REQUIRE(cloned.strides()[0] == 1);
  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.nbytes() == 5 * sizeof(float));

  auto* p = cloned.data<float>();
  CATCH_REQUIRE(p != src.data());
  for (int i = 0; i < 5; ++i) CATCH_REQUIRE(p[i] == src[i]);
}


CATCH_TEST_CASE("float: clone contiguous 2D tensor",
                "[ams][tensor][float][clone]")
{
  AMSInit();
  std::vector<float> src(12);
  for (int i = 0; i < 12; ++i) src[i] = static_cast<float>(i) * 0.5f;

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto view = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dim() == 2);
  CATCH_REQUIRE(cloned.shape()[0] == 3);
  CATCH_REQUIRE(cloned.shape()[1] == 4);
  CATCH_REQUIRE(cloned.contiguous());

  auto* p = cloned.data<float>();
  CATCH_REQUIRE(p != src.data());
  for (int i = 0; i < 12; ++i) CATCH_REQUIRE(p[i] == src[i]);
}


CATCH_TEST_CASE("float: clone non-contiguous (transposed) tensor",
                "[ams][tensor][float][clone][transpose]")
{
  AMSInit();
  // 3x4 row-major → transpose to 4x3
  std::vector<float> src(12);
  for (int i = 0; i < 12; ++i) src[i] = static_cast<float>(i);

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto original = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = original.transpose(0, 1);
  CATCH_REQUIRE(!transposed.contiguous());

  auto cloned = transposed.clone();

  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.shape()[0] == 4);
  CATCH_REQUIRE(cloned.shape()[1] == 3);
  CATCH_REQUIRE(cloned.strides()[0] == 3);
  CATCH_REQUIRE(cloned.strides()[1] == 1);

  // Logical [i,j] of transposed = src[j*4 + i]
  auto* p = cloned.data<float>();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j) {
      float expected = static_cast<float>(j * 4 + i);
      CATCH_INFO("clone[" << i << "," << j << "]");
      CATCH_REQUIRE(p[i * 3 + j] == expected);
    }
}


CATCH_TEST_CASE("float: clone is independent of source",
                "[ams][tensor][float][clone]")
{
  AMSInit();
  std::vector<float> src = {1.0f, 2.0f, 3.0f};

  std::vector<AMSTensor::IntDimType> shape = {3};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto view = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  src[0] = 999.0f;
  CATCH_REQUIRE(cloned.data<float>()[0] == 1.0f);
}


// =========================================================================
// float — concat
// =========================================================================

CATCH_TEST_CASE("float: concat single tensor",
                "[ams][tensor][float][concat]")
{
  AMSInit();
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto tA = AMSTensor::view<float>(
      a.data(), shape, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 3);
  auto* p = result.data<float>();
  for (int i = 0; i < 6; ++i) CATCH_REQUIRE(p[i] == a[i]);
}


CATCH_TEST_CASE("float: concat two 2D tensors",
                "[ams][tensor][float][concat]")
{
  AMSInit();
  // A:[3,2]       B:[3,3]
  // [1, 2]        [7, 8, 9]
  // [3, 4]        [10, 11, 12]
  // [5, 6]        [13, 14, 15]
  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  std::vector<float> b = {7, 8, 9, 10, 11, 12, 13, 14, 15};

  std::vector<AMSTensor::IntDimType> shapeA = {3, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};
  std::vector<AMSTensor::IntDimType> shapeB = {3, 3};
  std::vector<AMSTensor::IntDimType> stridesB = {3, 1};

  auto tA = AMSTensor::view<float>(
      a.data(), shapeA, stridesA, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<float>(
      b.data(), shapeB, stridesB, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(result.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(result.shape()[0] == 3);
  CATCH_REQUIRE(result.shape()[1] == 5);
  CATCH_REQUIRE(result.contiguous());

  auto* p = result.data<float>();
  std::vector<float> expected = {1, 2, 7, 8, 9, 3, 4, 10, 11, 12, 5, 6, 13, 14, 15};
  for (int i = 0; i < 15; ++i) {
    CATCH_INFO("index " << i);
    CATCH_REQUIRE(p[i] == expected[i]);
  }
}


CATCH_TEST_CASE("float: concat three tensors",
                "[ams][tensor][float][concat]")
{
  AMSInit();
  // A:[2,2]  B:[2,3]  C:[2,1]  →  [2,6]
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {10, 20, 30, 40, 50, 60};
  std::vector<float> c = {100, 200};

  std::vector<AMSTensor::IntDimType> shapeA =  {2, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};

  std::vector<AMSTensor::IntDimType> shapeB = {2, 3};
  std::vector<AMSTensor::IntDimType> stridesB = {3, 1};

  std::vector<AMSTensor::IntDimType> shapeC = {2, 1};
  std::vector<AMSTensor::IntDimType> stridesC = {1, 1};

  auto tA = AMSTensor::view<float>(
      a.data(), shapeA, stridesA, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<float>(
      b.data(), shapeB, stridesB, AMSResourceType::AMS_HOST);
  auto tC = AMSTensor::view<float>(
      c.data(), shapeC, stridesC, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));
  tensors.push_back(AMSTensor::view(tC));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 6);

  auto* p = result.data<float>();
  // Row 0: [1, 2, 10, 20, 30, 100]
  CATCH_REQUIRE(p[0] == 1.0f);
  CATCH_REQUIRE(p[1] == 2.0f);
  CATCH_REQUIRE(p[2] == 10.0f);
  CATCH_REQUIRE(p[3] == 20.0f);
  CATCH_REQUIRE(p[4] == 30.0f);
  CATCH_REQUIRE(p[5] == 100.0f);
  // Row 1: [3, 4, 40, 50, 60, 200]
  CATCH_REQUIRE(p[6] == 3.0f);
  CATCH_REQUIRE(p[7] == 4.0f);
  CATCH_REQUIRE(p[8] == 40.0f);
  CATCH_REQUIRE(p[9] == 50.0f);
  CATCH_REQUIRE(p[10] == 60.0f);
  CATCH_REQUIRE(p[11] == 200.0f);
}


CATCH_TEST_CASE("float: concat 1D tensors", "[ams][tensor][float][concat]")
{
  AMSInit();
  std::vector<float> a = {1.5f, 2.5f, 3.5f};
  std::vector<float> b = {4.5f, 5.5f};

  std::vector<AMSTensor::IntDimType> shapeA =  {3};
  std::vector<AMSTensor::IntDimType> strides = {1};
  std::vector<AMSTensor::IntDimType> shapeB = {2};

  auto tA = AMSTensor::view<float>(
      a.data(), shapeA, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<float>(
      b.data(), shapeB, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(result.dim() == 1);
  CATCH_REQUIRE(result.shape()[0] == 5);

  auto* p = result.data<float>();
  CATCH_REQUIRE(p[0] == 1.5f);
  CATCH_REQUIRE(p[4] == 5.5f);
}


CATCH_TEST_CASE("float: concat result independent of source",
                "[ams][tensor][float][concat]")
{
  AMSInit();
  std::vector<float> a = {1.0f, 2.0f};
  std::vector<float> b = {3.0f, 4.0f};

  std::vector<AMSTensor::IntDimType> shape = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tA = AMSTensor::view<float>(
      a.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<float>(
      b.data(), shape, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);
  auto* p = result.data<float>();

  a[0] = 999.0f;
  b[0] = 888.0f;
  CATCH_REQUIRE(p[0] == 1.0f);
  CATCH_REQUIRE(p[2] == 3.0f);
}


// =========================================================================
// double — create
// =========================================================================

CATCH_TEST_CASE("double: create 1D tensor", "[ams][tensor][double][create]")
{
  AMSInit();
  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);
  if (device == AMSResourceType::AMS_DEVICE) {
#if !defined(__AMS_ENABLE_CUDA__) && !defined(__AMS_ENABLE_HIP__)
    CATCH_SKIP("GPU device not available");
#endif
  }

  std::vector<AMSTensor::IntDimType> shape = {7};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor = AMSTensor::create<double>(shape, strides, device);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(tensor.elements() == 7);
  CATCH_REQUIRE(tensor.element_size() == sizeof(double));
  CATCH_REQUIRE(tensor.dim() == 1);
  CATCH_REQUIRE(tensor.nbytes() == 7 * sizeof(double));
  CATCH_REQUIRE(tensor.location() == device);
}


CATCH_TEST_CASE("double: create 2D tensor", "[ams][tensor][double][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {5, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto tensor =
      AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(tensor.elements() == 15);
  CATCH_REQUIRE(tensor.dim() == 2);
  CATCH_REQUIRE(tensor.nbytes() == 15 * sizeof(double));
}


// =========================================================================
// double — view
// =========================================================================

CATCH_TEST_CASE("double: view from existing buffer",
                "[ams][tensor][double][view]")
{
  AMSInit();
  std::vector<double> data = {1.11, 2.22, 3.33, 4.44};

  std::vector<AMSTensor::IntDimType> shape = {4};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto v = AMSTensor::view<double>(
      data.data(), shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(v.elements() == 4);
  CATCH_REQUIRE(v.data<double>() == data.data());
  CATCH_REQUIRE(v.data<double>()[0] == 1.11);
  CATCH_REQUIRE(v.data<double>()[3] == 4.44);
}


// =========================================================================
// double — data access
// =========================================================================

CATCH_TEST_CASE("double: write and read data",
                "[ams][tensor][double][data]")
{
  AMSInit();

  std::vector<AMSTensor::IntDimType> shape = {3};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<double>();

  data[0] = 1.0e-15;
  data[1] = 3.141592653589793;
  data[2] = 1.0e+15;

  CATCH_REQUIRE(data[0] == 1.0e-15);
  CATCH_REQUIRE(data[1] == 3.141592653589793);
  CATCH_REQUIRE(data[2] == 1.0e+15);
}


// =========================================================================
// double — transpose
// =========================================================================

CATCH_TEST_CASE("double: transpose 2D tensor",
                "[ams][tensor][double][transpose]")
{
  AMSInit();

  std::vector<AMSTensor::IntDimType> shape = {4, 7};
  std::vector<AMSTensor::IntDimType> strides = {7, 1};

  auto tensor =
      AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = tensor.transpose(0, 1);

  CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(transposed.shape()[0] == 7);
  CATCH_REQUIRE(transposed.shape()[1] == 4);
  CATCH_REQUIRE(transposed.elements() == 28);
  CATCH_REQUIRE(!transposed.contiguous());
}


// =========================================================================
// double — move
// =========================================================================

CATCH_TEST_CASE("double: move constructor", "[ams][tensor][double][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {6};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto t1 = AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = t1.data<double>();

  auto t2 = std::move(t1);

  CATCH_REQUIRE(t2.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(t2.elements() == 6);
  CATCH_REQUIRE(t2.nbytes() == 6 * sizeof(double));
  CATCH_REQUIRE(t2.data<double>() == ptr);
}


// =========================================================================
// double — clone
// =========================================================================

CATCH_TEST_CASE("double: clone contiguous 2D tensor",
                "[ams][tensor][double][clone]")
{
  AMSInit();
  std::vector<double> src = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto view = AMSTensor::view<double>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(cloned.elements() == 6);
  CATCH_REQUIRE(cloned.nbytes() == 6 * sizeof(double));
  CATCH_REQUIRE(cloned.contiguous());

  auto* p = cloned.data<double>();
  CATCH_REQUIRE(p != src.data());
  for (int i = 0; i < 6; ++i) CATCH_REQUIRE(p[i] == src[i]);
}


CATCH_TEST_CASE("double: clone non-contiguous (transposed) tensor",
                "[ams][tensor][double][clone][transpose]")
{
  AMSInit();
  // 2x4 row-major → transpose to 4x2
  std::vector<double> src = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<AMSTensor::IntDimType> shape = {2, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto original = AMSTensor::view<double>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = original.transpose(0, 1);
  CATCH_REQUIRE(!transposed.contiguous());

  auto cloned = transposed.clone();

  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.shape()[0] == 4);
  CATCH_REQUIRE(cloned.shape()[1] == 2);
  CATCH_REQUIRE(cloned.strides()[0] == 2);
  CATCH_REQUIRE(cloned.strides()[1] == 1);

  // Logical [i,j] of transposed = src[j*4 + i]
  auto* p = cloned.data<double>();
  CATCH_REQUIRE(p[0] == 10.0);  // [0,0] = src[0*4+0]
  CATCH_REQUIRE(p[1] == 50.0);  // [0,1] = src[1*4+0]
  CATCH_REQUIRE(p[2] == 20.0);  // [1,0] = src[0*4+1]
  CATCH_REQUIRE(p[3] == 60.0);  // [1,1] = src[1*4+1]
  CATCH_REQUIRE(p[4] == 30.0);  // [2,0] = src[0*4+2]
  CATCH_REQUIRE(p[5] == 70.0);  // [2,1] = src[1*4+2]
  CATCH_REQUIRE(p[6] == 40.0);  // [3,0] = src[0*4+3]
  CATCH_REQUIRE(p[7] == 80.0);  // [3,1] = src[1*4+3]
}


CATCH_TEST_CASE("double: clone is independent of source",
                "[ams][tensor][double][clone]")
{
  AMSInit();
  std::vector<double> src = {1.0, 2.0, 3.0};
  std::vector<AMSTensor::IntDimType> shape = {3};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto view = AMSTensor::view<double>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  src[0] = 999.0;
  CATCH_REQUIRE(cloned.data<double>()[0] == 1.0);
}


// =========================================================================
// double — concat
// =========================================================================

CATCH_TEST_CASE("double: concat two 2D tensors",
                "[ams][tensor][double][concat]")
{
  AMSInit();
  std::vector<double> a = {1.1, 2.2, 3.3, 4.4};
  std::vector<double> b = {5.5, 6.6, 7.7, 8.8};

  std::vector<AMSTensor::IntDimType> shape = {2, 2};
  std::vector<AMSTensor::IntDimType> strides = {2, 1};

  auto tA = AMSTensor::view<double>(
      a.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<double>(
      b.data(), shape, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_DOUBLE);

  CATCH_REQUIRE(result.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 4);

  auto* p = result.data<double>();
  // Row 0: [1.1, 2.2, 5.5, 6.6]
  CATCH_REQUIRE(p[0] == 1.1);
  CATCH_REQUIRE(p[1] == 2.2);
  CATCH_REQUIRE(p[2] == 5.5);
  CATCH_REQUIRE(p[3] == 6.6);
  // Row 1: [3.3, 4.4, 7.7, 8.8]
  CATCH_REQUIRE(p[4] == 3.3);
  CATCH_REQUIRE(p[5] == 4.4);
  CATCH_REQUIRE(p[6] == 7.7);
  CATCH_REQUIRE(p[7] == 8.8);
}


CATCH_TEST_CASE("double: concat 1D tensors",
                "[ams][tensor][double][concat]")
{
  AMSInit();
  std::vector<double> a = {1.0, 2.0};
  std::vector<double> b = {3.0, 4.0, 5.0};
  std::vector<AMSTensor::IntDimType> shapeA = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};
  std::vector<AMSTensor::IntDimType> shapeB = {3};

  auto tA = AMSTensor::view<double>(
      a.data(), shapeA, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<double>(
      b.data(), shapeB, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_DOUBLE);

  CATCH_REQUIRE(result.dim() == 1);
  CATCH_REQUIRE(result.shape()[0] == 5);

  auto* p = result.data<double>();
  CATCH_REQUIRE(p[0] == 1.0);
  CATCH_REQUIRE(p[4] == 5.0);
}

// =========================================================================
// dtype_to_size utility
// =========================================================================

CATCH_TEST_CASE("dtype_to_size: float types", "[ams][utils][float]")
{
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_SINGLE) == sizeof(float));
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_DOUBLE) == sizeof(double));
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_DOUBLE) ==
                2 * dtype_to_size(AMSDType::AMS_SINGLE));
}
