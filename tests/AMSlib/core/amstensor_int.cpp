/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "ams_test_device.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

// =========================================================================
// int32_t — create
// =========================================================================

CATCH_TEST_CASE("int32: create 1D tensor", "[ams][tensor][int32][create]")
{
  AMSInit();
  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);
  if (device == AMSResourceType::AMS_DEVICE) {
    if (!ams::test::hasRuntimeDevice()) CATCH_SKIP("GPU device not available");
  }

  std::vector<AMSTensor::IntDimType> shape = {10};
  std::vector<AMSTensor::IntDimType> strides = {1};
  auto tensor = AMSTensor::create<int32_t>(shape, strides, device);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(tensor.elements() == 10);
  CATCH_REQUIRE(tensor.element_size() == sizeof(int32_t));
  CATCH_REQUIRE(tensor.dim() == 1);
  CATCH_REQUIRE(tensor.nbytes() == 10 * sizeof(int32_t));
  CATCH_REQUIRE(tensor.location() == device);
  CATCH_REQUIRE(tensor.shape()[0] == 10);
}


CATCH_TEST_CASE("int32: create 2D tensor", "[ams][tensor][int32][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {5, 8};
  std::vector<AMSTensor::IntDimType> strides = {8, 1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(tensor.elements() == 40);
  CATCH_REQUIRE(tensor.dim() == 2);
  CATCH_REQUIRE(tensor.shape()[0] == 5);
  CATCH_REQUIRE(tensor.shape()[1] == 8);
  CATCH_REQUIRE(tensor.nbytes() == 40 * sizeof(int32_t));
}


CATCH_TEST_CASE("int32: create 3D tensor", "[ams][tensor][int32][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {4, 3, 2};
  std::vector<AMSTensor::IntDimType> strides = {6, 2, 1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(tensor.elements() == 24);
  CATCH_REQUIRE(tensor.dim() == 3);
}


// =========================================================================
// int32_t — view
// =========================================================================

CATCH_TEST_CASE("int32: view 1D from existing buffer",
                "[ams][tensor][int32][view]")
{
  AMSInit();
  std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<AMSTensor::IntDimType> shape = {10};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto v = AMSTensor::view<int32_t>(data.data(),
                                    shape,
                                    strides,
                                    AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(v.elements() == 10);
  CATCH_REQUIRE(v.data<int32_t>() == data.data());  // shares memory
  CATCH_REQUIRE(v.data<int32_t>()[0] == 1);
  CATCH_REQUIRE(v.data<int32_t>()[9] == 10);
}


CATCH_TEST_CASE("int32: view 2D from existing buffer",
                "[ams][tensor][int32][view]")
{
  AMSInit();
  std::vector<int32_t> data(20, 42);

  std::vector<AMSTensor::IntDimType> shape = {4, 5};
  std::vector<AMSTensor::IntDimType> strides = {5, 1};

  auto v = AMSTensor::view<int32_t>(data.data(),
                                    shape,
                                    strides,
                                    AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.elements() == 20);
  CATCH_REQUIRE(v.shape()[0] == 4);
  CATCH_REQUIRE(v.shape()[1] == 5);
  CATCH_REQUIRE(v.data<int32_t>()[0] == 42);
  CATCH_REQUIRE(v.data<int32_t>()[19] == 42);
}


CATCH_TEST_CASE("int32: view from AMSTensor alias",
                "[ams][tensor][int32][view]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {6};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = tensor.data<int32_t>();
  for (int i = 0; i < 6; ++i)
    ptr[i] = i + 1;

  auto alias = AMSTensor::view(tensor);

  CATCH_REQUIRE(alias.data<int32_t>() == ptr);
  CATCH_REQUIRE(alias.elements() == 6);
  CATCH_REQUIRE(alias.data<int32_t>()[5] == 6);
}


// =========================================================================
// int32_t — data access
// =========================================================================

CATCH_TEST_CASE("int32: write and read 1D data", "[ams][tensor][int32][data]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<int32_t>();

  for (int i = 0; i < 5; ++i)
    data[i] = i * 10;

  CATCH_REQUIRE(data[0] == 0);
  CATCH_REQUIRE(data[1] == 10);
  CATCH_REQUIRE(data[4] == 40);
}


CATCH_TEST_CASE("int32: write and read 2D data", "[ams][tensor][int32][data]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<int32_t>();

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      data[i * 4 + j] = i * 10 + j;

  CATCH_REQUIRE(data[0] == 0);    // [0,0]
  CATCH_REQUIRE(data[3] == 3);    // [0,3]
  CATCH_REQUIRE(data[4] == 10);   // [1,0]
  CATCH_REQUIRE(data[11] == 23);  // [2,3]
}


// =========================================================================
// int32_t — transpose
// =========================================================================

CATCH_TEST_CASE("int32: transpose 2D tensor", "[ams][tensor][int32][transpose]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto tensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = tensor.transpose(0, 1);

  CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(transposed.shape()[0] == 4);
  CATCH_REQUIRE(transposed.shape()[1] == 3);
  CATCH_REQUIRE(transposed.elements() == 12);
  CATCH_REQUIRE(!transposed.contiguous());
}


// =========================================================================
// int32_t — move
// =========================================================================

CATCH_TEST_CASE("int32: move constructor", "[ams][tensor][int32][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {10};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto t1 =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = t1.data<int32_t>();

  auto t2 = std::move(t1);

  CATCH_REQUIRE(t2.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(t2.elements() == 10);
  CATCH_REQUIRE(t2.dim() == 1);
  CATCH_REQUIRE(t2.nbytes() == 10 * sizeof(int32_t));
  CATCH_REQUIRE(t2.data<int32_t>() == ptr);
}


CATCH_TEST_CASE("int32: move assignment", "[ams][tensor][int32][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape1 = {10};
  std::vector<AMSTensor::IntDimType> shape2 = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto t1 =
      AMSTensor::create<int32_t>(shape1, strides, AMSResourceType::AMS_HOST);
  auto t2 =
      AMSTensor::create<int32_t>(shape2, strides, AMSResourceType::AMS_HOST);
  auto* ptr1 = t1.data<int32_t>();

  t2 = std::move(t1);

  CATCH_REQUIRE(t2.elements() == 10);
  CATCH_REQUIRE(t2.data<int32_t>() == ptr1);
}


// =========================================================================
// int32_t — clone
// =========================================================================

CATCH_TEST_CASE("int32: clone contiguous 2D tensor",
                "[ams][tensor][int32][clone]")
{
  AMSInit();
  std::vector<int32_t> src(12);
  for (int i = 0; i < 12; ++i)
    src[i] = i * 7;

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto view = AMSTensor::view<int32_t>(src.data(),
                                       shape,
                                       strides,
                                       AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(cloned.dim() == 2);
  CATCH_REQUIRE(cloned.shape()[0] == 3);
  CATCH_REQUIRE(cloned.shape()[1] == 4);
  CATCH_REQUIRE(cloned.elements() == 12);
  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.strides()[0] == 4);
  CATCH_REQUIRE(cloned.strides()[1] == 1);

  auto* clonedPtr = cloned.data<int32_t>();
  CATCH_REQUIRE(clonedPtr != src.data());
  for (int i = 0; i < 12; ++i) {
    CATCH_REQUIRE(clonedPtr[i] == src[i]);
  }
}


CATCH_TEST_CASE("int32: clone non-contiguous (transposed) tensor",
                "[ams][tensor][int32][clone][transpose]")
{
  AMSInit();
  // 3x4 row-major, transposed to 4x3 with strides [1,4]
  std::vector<int32_t> src(12);
  for (int i = 0; i < 12; ++i)
    src[i] = i;

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto original = AMSTensor::view<int32_t>(src.data(),
                                           shape,
                                           strides,
                                           AMSResourceType::AMS_HOST);
  auto transposed = original.transpose(0, 1);
  CATCH_REQUIRE(!transposed.contiguous());

  auto cloned = transposed.clone();

  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.shape()[0] == 4);
  CATCH_REQUIRE(cloned.shape()[1] == 3);
  CATCH_REQUIRE(cloned.strides()[0] == 3);
  CATCH_REQUIRE(cloned.strides()[1] == 1);

  // Logical element [i,j] of transposed is src[j*4 + i]
  auto* p = cloned.data<int32_t>();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      CATCH_REQUIRE(p[i * 3 + j] == static_cast<int32_t>(j * 4 + i));
}


CATCH_TEST_CASE("int32: clone is independent of source",
                "[ams][tensor][int32][clone]")
{
  AMSInit();
  std::vector<int32_t> src = {10, 20, 30, 40};

  std::vector<AMSTensor::IntDimType> shape = {4};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto view = AMSTensor::view<int32_t>(src.data(),
                                       shape,
                                       strides,
                                       AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  src[0] = 999;
  CATCH_REQUIRE(cloned.data<int32_t>()[0] == 10);
}


// =========================================================================
// int32_t — concat
// =========================================================================

CATCH_TEST_CASE("int32: concat single tensor", "[ams][tensor][int32][concat]")
{
  AMSInit();
  std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto tA = AMSTensor::view<int32_t>(a.data(),
                                     shape,
                                     strides,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 3);
  auto* p = result.data<int32_t>();
  for (int i = 0; i < 6; ++i)
    CATCH_REQUIRE(p[i] == a[i]);
}


CATCH_TEST_CASE("int32: concat two 2D tensors", "[ams][tensor][int32][concat]")
{
  AMSInit();
  //  A:[3,2]      B:[3,3]
  //  [1, 2]       [10, 20, 30]
  //  [3, 4]       [40, 50, 60]
  //  [5, 6]       [70, 80, 90]
  //
  //  Result: [3,5]
  //  [1, 2, 10, 20, 30]
  //  [3, 4, 40, 50, 60]
  //  [5, 6, 70, 80, 90]
  std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> b = {10, 20, 30, 40, 50, 60, 70, 80, 90};
  std::vector<AMSTensor::IntDimType> shapeA = {3, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};

  auto tA = AMSTensor::view<int32_t>(a.data(),
                                     shapeA,
                                     stridesA,
                                     AMSResourceType::AMS_HOST);

  std::vector<AMSTensor::IntDimType> shapeB = {3, 3};
  std::vector<AMSTensor::IntDimType> stridesB = {3, 1};

  auto tB = AMSTensor::view<int32_t>(b.data(),
                                     shapeB,
                                     stridesB,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

  CATCH_REQUIRE(result.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(result.shape()[0] == 3);
  CATCH_REQUIRE(result.shape()[1] == 5);
  CATCH_REQUIRE(result.contiguous());

  auto* p = result.data<int32_t>();
  std::vector<int32_t> expected = {
      1, 2, 10, 20, 30, 3, 4, 40, 50, 60, 5, 6, 70, 80, 90};
  for (int i = 0; i < 15; ++i) {
    CATCH_INFO("index " << i);
    CATCH_REQUIRE(p[i] == expected[i]);
  }
}


CATCH_TEST_CASE("int32: concat three tensors", "[ams][tensor][int32][concat]")
{
  AMSInit();
  // A:[2,2]  B:[2,1]  C:[2,3]  →  [2,6]
  std::vector<int32_t> a = {1, 2, 3, 4};
  std::vector<int32_t> b = {10, 20};
  std::vector<int32_t> c = {100, 200, 300, 400, 500, 600};

  std::vector<AMSTensor::IntDimType> shapeA = {2, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};

  std::vector<AMSTensor::IntDimType> shapeB = {2, 1};
  std::vector<AMSTensor::IntDimType> stridesB = {1, 1};

  std::vector<AMSTensor::IntDimType> shapeC = {2, 3};
  std::vector<AMSTensor::IntDimType> stridesC = {3, 1};

  auto tA = AMSTensor::view<int32_t>(a.data(),
                                     shapeA,
                                     stridesA,
                                     AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<int32_t>(b.data(),
                                     shapeB,
                                     stridesB,
                                     AMSResourceType::AMS_HOST);
  auto tC = AMSTensor::view<int32_t>(c.data(),
                                     shapeC,
                                     stridesC,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));
  tensors.push_back(AMSTensor::view(tC));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 6);

  auto* p = result.data<int32_t>();
  // Row 0: [1, 2, 10, 100, 200, 300]
  CATCH_REQUIRE(p[0] == 1);
  CATCH_REQUIRE(p[1] == 2);
  CATCH_REQUIRE(p[2] == 10);
  CATCH_REQUIRE(p[3] == 100);
  CATCH_REQUIRE(p[4] == 200);
  CATCH_REQUIRE(p[5] == 300);
  // Row 1: [3, 4, 20, 400, 500, 600]
  CATCH_REQUIRE(p[6] == 3);
  CATCH_REQUIRE(p[7] == 4);
  CATCH_REQUIRE(p[8] == 20);
  CATCH_REQUIRE(p[9] == 400);
  CATCH_REQUIRE(p[10] == 500);
  CATCH_REQUIRE(p[11] == 600);
}


CATCH_TEST_CASE("int32: concat 1D tensors", "[ams][tensor][int32][concat]")
{
  AMSInit();
  std::vector<int32_t> a = {1, 2, 3};
  std::vector<int32_t> b = {4, 5};

  std::vector<AMSTensor::IntDimType> shapeA = {3};
  std::vector<AMSTensor::IntDimType> shapeB = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tA = AMSTensor::view<int32_t>(a.data(),
                                     shapeA,
                                     strides,
                                     AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<int32_t>(b.data(),
                                     shapeB,
                                     strides,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

  CATCH_REQUIRE(result.dim() == 1);
  CATCH_REQUIRE(result.shape()[0] == 5);

  auto* p = result.data<int32_t>();
  CATCH_REQUIRE(p[0] == 1);
  CATCH_REQUIRE(p[1] == 2);
  CATCH_REQUIRE(p[2] == 3);
  CATCH_REQUIRE(p[3] == 4);
  CATCH_REQUIRE(p[4] == 5);
}


CATCH_TEST_CASE("int32: concat result independent of source",
                "[ams][tensor][int32][concat]")
{
  AMSInit();
  std::vector<int32_t> a = {1, 2};
  std::vector<int32_t> b = {3, 4};

  std::vector<AMSTensor::IntDimType> shape = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tA = AMSTensor::view<int32_t>(a.data(),
                                     shape,
                                     strides,
                                     AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<int32_t>(b.data(),
                                     shape,
                                     strides,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);
  auto* p = result.data<int32_t>();

  a[0] = 999;
  b[0] = 888;
  CATCH_REQUIRE(p[0] == 1);
  CATCH_REQUIRE(p[2] == 3);
}


// =========================================================================
// int64_t — create
// =========================================================================

CATCH_TEST_CASE("int64: create 1D tensor", "[ams][tensor][int64][create]")
{
  AMSInit();
  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);
  if (device == AMSResourceType::AMS_DEVICE) {
    if (!ams::test::hasRuntimeDevice()) CATCH_SKIP("GPU device not available");
  }

  std::vector<AMSTensor::IntDimType> shape = {15};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor = AMSTensor::create<int64_t>(shape, strides, device);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(tensor.elements() == 15);
  CATCH_REQUIRE(tensor.element_size() == sizeof(int64_t));
  CATCH_REQUIRE(tensor.dim() == 1);
  CATCH_REQUIRE(tensor.nbytes() == 15 * sizeof(int64_t));
  CATCH_REQUIRE(tensor.location() == device);
}


CATCH_TEST_CASE("int64: create 2D tensor", "[ams][tensor][int64][create]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {6, 7};
  std::vector<AMSTensor::IntDimType> strides = {7, 1};

  auto tensor =
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(tensor.elements() == 42);
  CATCH_REQUIRE(tensor.dim() == 2);
  CATCH_REQUIRE(tensor.nbytes() == 42 * sizeof(int64_t));
}


// =========================================================================
// int64_t — view
// =========================================================================

CATCH_TEST_CASE("int64: view from existing buffer",
                "[ams][tensor][int64][view]")
{
  AMSInit();
  std::vector<int64_t> data = {100, 200, 300, 400, 500};

  std::vector<AMSTensor::IntDimType> shape = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto v = AMSTensor::view<int64_t>(data.data(),
                                    shape,
                                    strides,
                                    AMSResourceType::AMS_HOST);

  CATCH_REQUIRE(v.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(v.elements() == 5);
  CATCH_REQUIRE(v.data<int64_t>()[0] == 100);
  CATCH_REQUIRE(v.data<int64_t>()[4] == 500);
}


// =========================================================================
// int64_t — data access
// =========================================================================

CATCH_TEST_CASE("int64: write and read large values",
                "[ams][tensor][int64][data]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {3};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tensor =
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* data = tensor.data<int64_t>();

  data[0] = 1000000000LL;
  data[1] = 2000000000LL;
  data[2] = 3000000000LL;

  CATCH_REQUIRE(data[0] == 1000000000LL);
  CATCH_REQUIRE(data[1] == 2000000000LL);
  CATCH_REQUIRE(data[2] == 3000000000LL);
}


// =========================================================================
// int64_t — transpose
// =========================================================================

CATCH_TEST_CASE("int64: transpose 2D tensor", "[ams][tensor][int64][transpose]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {5, 6};
  std::vector<AMSTensor::IntDimType> strides = {6, 1};

  auto tensor =
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = tensor.transpose(0, 1);

  CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(transposed.shape()[0] == 6);
  CATCH_REQUIRE(transposed.shape()[1] == 5);
  CATCH_REQUIRE(transposed.elements() == 30);
}


// =========================================================================
// int64_t — move
// =========================================================================

CATCH_TEST_CASE("int64: move constructor", "[ams][tensor][int64][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {20};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto t1 =
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto* ptr = t1.data<int64_t>();

  auto t2 = std::move(t1);

  CATCH_REQUIRE(t2.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(t2.elements() == 20);
  CATCH_REQUIRE(t2.nbytes() == 20 * sizeof(int64_t));
  CATCH_REQUIRE(t2.data<int64_t>() == ptr);
}


// =========================================================================
// int64_t — clone
// =========================================================================

CATCH_TEST_CASE("int64: clone contiguous tensor", "[ams][tensor][int64][clone]")
{
  AMSInit();
  std::vector<int64_t> src = {100, 200, 300, 400, 500, 600};
  std::vector<AMSTensor::IntDimType> shape = {3, 2};
  std::vector<AMSTensor::IntDimType> strides = {2, 1};

  auto view = AMSTensor::view<int64_t>(src.data(),
                                       shape,
                                       strides,
                                       AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(cloned.elements() == 6);
  CATCH_REQUIRE(cloned.nbytes() == 6 * sizeof(int64_t));
  CATCH_REQUIRE(cloned.contiguous());

  auto* p = cloned.data<int64_t>();
  CATCH_REQUIRE(p != src.data());
  for (int i = 0; i < 6; ++i)
    CATCH_REQUIRE(p[i] == src[i]);
}


CATCH_TEST_CASE("int64: clone non-contiguous tensor",
                "[ams][tensor][int64][clone][transpose]")
{
  AMSInit();
  // 2x3 row-major, transposed to 3x2
  std::vector<int64_t> src = {10, 20, 30, 40, 50, 60};
  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto original = AMSTensor::view<int64_t>(src.data(),
                                           shape,
                                           strides,
                                           AMSResourceType::AMS_HOST);
  auto transposed = original.transpose(0, 1);
  auto cloned = transposed.clone();

  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.shape()[0] == 3);
  CATCH_REQUIRE(cloned.shape()[1] == 2);
  CATCH_REQUIRE(cloned.strides()[0] == 2);
  CATCH_REQUIRE(cloned.strides()[1] == 1);

  // Logical [i,j] of transposed = src[j*3 + i]
  auto* p = cloned.data<int64_t>();
  CATCH_REQUIRE(p[0] == 10);  // [0,0] = src[0*3+0]
  CATCH_REQUIRE(p[1] == 40);  // [0,1] = src[1*3+0]
  CATCH_REQUIRE(p[2] == 20);  // [1,0] = src[0*3+1]
  CATCH_REQUIRE(p[3] == 50);  // [1,1] = src[1*3+1]
  CATCH_REQUIRE(p[4] == 30);  // [2,0] = src[0*3+2]
  CATCH_REQUIRE(p[5] == 60);  // [2,1] = src[1*3+2]
}


// =========================================================================
// int64_t — concat
// =========================================================================

CATCH_TEST_CASE("int64: concat two 2D tensors", "[ams][tensor][int64][concat]")
{
  AMSInit();
  std::vector<int64_t> a = {1, 2, 3, 4};
  std::vector<int64_t> b = {10, 20, 30, 40, 50, 60};
  std::vector<AMSTensor::IntDimType> shapeA = {2, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};
  std::vector<AMSTensor::IntDimType> shapeB = {2, 3};
  std::vector<AMSTensor::IntDimType> stridesB = {3, 1};

  auto tA = AMSTensor::view<int64_t>(a.data(),
                                     shapeA,
                                     stridesA,
                                     AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<int64_t>(b.data(),
                                     shapeB,
                                     stridesB,
                                     AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT64);

  CATCH_REQUIRE(result.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 5);

  auto* p = result.data<int64_t>();
  // Row 0: [1, 2, 10, 20, 30]
  CATCH_REQUIRE(p[0] == 1);
  CATCH_REQUIRE(p[1] == 2);
  CATCH_REQUIRE(p[2] == 10);
  CATCH_REQUIRE(p[3] == 20);
  CATCH_REQUIRE(p[4] == 30);
  // Row 1: [3, 4, 40, 50, 60]
  CATCH_REQUIRE(p[5] == 3);
  CATCH_REQUIRE(p[6] == 4);
  CATCH_REQUIRE(p[7] == 40);
  CATCH_REQUIRE(p[8] == 50);
  CATCH_REQUIRE(p[9] == 60);
}


// =========================================================================
// dtype_to_size utility
// =========================================================================

CATCH_TEST_CASE("dtype_to_size: integer types", "[ams][utils][int]")
{
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_INT32) == sizeof(int32_t));
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_INT64) == sizeof(int64_t));
  CATCH_REQUIRE(dtype_to_size(AMSDType::AMS_INT64) ==
                2 * dtype_to_size(AMSDType::AMS_INT32));
}
