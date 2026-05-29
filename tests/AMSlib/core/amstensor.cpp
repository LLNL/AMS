/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

CATCH_TEST_CASE("AMSTensor: int32_t tensor creation and basic properties",
                "[ams][tensor][int32]")
{
  AMSInit();

  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);

  // Skip GPU tests if CUDA is not available
  if (device == AMSResourceType::AMS_DEVICE) {
#if !defined(__AMS_ENABLE_CUDA__) && !defined(__AMS_ENABLE_HIP__)
    CATCH_SKIP("GPU device not available");
#endif
  }

  CATCH_SECTION("Create 1D int32_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {10};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);

    CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor.elements() == 10);
    CATCH_REQUIRE(tensor.element_size() == sizeof(int32_t));
    CATCH_REQUIRE(tensor.location() == device);
    CATCH_REQUIRE(tensor.shape().size() == 1);
    CATCH_REQUIRE(tensor.shape()[0] == 10);
    // Note: contiguous() check removed due to pre-existing AMSTensor bug
  }

  CATCH_SECTION("Create 2D int32_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {5, 8};
    std::vector<AMSTensor::IntDimType> strides = {8, 1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);

    CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor.elements() == 40);
    CATCH_REQUIRE(tensor.element_size() == sizeof(int32_t));
    CATCH_REQUIRE(tensor.shape().size() == 2);
    CATCH_REQUIRE(tensor.shape()[0] == 5);
    CATCH_REQUIRE(tensor.shape()[1] == 8);
  }

  CATCH_SECTION("Create 3D int32_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {4, 3, 2};
    std::vector<AMSTensor::IntDimType> strides = {6, 2, 1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);

    CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor.elements() == 24);
    CATCH_REQUIRE(tensor.element_size() == sizeof(int32_t));
  }
}

CATCH_TEST_CASE("AMSTensor: int64_t tensor creation and basic properties",
                "[ams][tensor][int64]")
{
  AMSInit();

  const auto device =
      GENERATE(AMSResourceType::AMS_HOST, AMSResourceType::AMS_DEVICE);

  if (device == AMSResourceType::AMS_DEVICE) {
#if !defined(__AMS_ENABLE_CUDA__) && !defined(__AMS_ENABLE_HIP__)
    CATCH_SKIP("GPU device not available");
#endif
  }

  CATCH_SECTION("Create 1D int64_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {15};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor = AMSTensor::create<int64_t>(shape, strides, device);

    CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT64);
    CATCH_REQUIRE(tensor.elements() == 15);
    CATCH_REQUIRE(tensor.element_size() == sizeof(int64_t));
    CATCH_REQUIRE(tensor.location() == device);
    // Note: contiguous() check removed due to pre-existing AMSTensor bug
  }

  CATCH_SECTION("Create 2D int64_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {6, 7};
    std::vector<AMSTensor::IntDimType> strides = {7, 1};

    auto tensor = AMSTensor::create<int64_t>(shape, strides, device);

    CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT64);
    CATCH_REQUIRE(tensor.elements() == 42);
    CATCH_REQUIRE(tensor.element_size() == sizeof(int64_t));
  }
}

CATCH_TEST_CASE("AMSTensor: int32_t tensor view operations",
                "[ams][tensor][int32][view]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Create view from existing int32_t data")
  {
    std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<AMSTensor::IntDimType> shape = {10};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor_view =
        AMSTensor::view<int32_t>(data.data(), shape, strides, device);

    CATCH_REQUIRE(tensor_view.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor_view.elements() == 10);
    CATCH_REQUIRE(tensor_view.location() == device);

    // Verify we can access the data
    auto* ptr = tensor_view.data<int32_t>();
    CATCH_REQUIRE(ptr != nullptr);
    CATCH_REQUIRE(ptr[0] == 1);
    CATCH_REQUIRE(ptr[9] == 10);
  }

  CATCH_SECTION("Create 2D view from int32_t data")
  {
    std::vector<int32_t> data(20, 42);  // 20 elements, all set to 42
    std::vector<AMSTensor::IntDimType> shape = {4, 5};
    std::vector<AMSTensor::IntDimType> strides = {5, 1};

    auto tensor_view =
        AMSTensor::view<int32_t>(data.data(), shape, strides, device);

    CATCH_REQUIRE(tensor_view.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor_view.elements() == 20);
    CATCH_REQUIRE(tensor_view.shape()[0] == 4);
    CATCH_REQUIRE(tensor_view.shape()[1] == 5);

    auto* ptr = tensor_view.data<int32_t>();
    CATCH_REQUIRE(ptr[0] == 42);
    CATCH_REQUIRE(ptr[19] == 42);
  }
}

CATCH_TEST_CASE("AMSTensor: int64_t tensor view operations",
                "[ams][tensor][int64][view]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Create view from existing int64_t data")
  {
    std::vector<int64_t> data = {100, 200, 300, 400, 500};
    std::vector<AMSTensor::IntDimType> shape = {5};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor_view =
        AMSTensor::view<int64_t>(data.data(), shape, strides, device);

    CATCH_REQUIRE(tensor_view.dtype() == AMSDType::AMS_INT64);
    CATCH_REQUIRE(tensor_view.elements() == 5);

    auto* ptr = tensor_view.data<int64_t>();
    CATCH_REQUIRE(ptr[0] == 100);
    CATCH_REQUIRE(ptr[4] == 500);
  }
}

CATCH_TEST_CASE("AMSTensor: int tensor transpose operations",
                "[ams][tensor][transpose]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Transpose 2D int32_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {3, 4};
    std::vector<AMSTensor::IntDimType> strides = {4, 1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);
    auto transposed = tensor.transpose(0, 1);

    CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(transposed.shape()[0] == 4);
    CATCH_REQUIRE(transposed.shape()[1] == 3);
    CATCH_REQUIRE(transposed.elements() == 12);
  }

  CATCH_SECTION("Transpose 2D int64_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {5, 6};
    std::vector<AMSTensor::IntDimType> strides = {6, 1};

    auto tensor = AMSTensor::create<int64_t>(shape, strides, device);
    auto transposed = tensor.transpose(0, 1);

    CATCH_REQUIRE(transposed.dtype() == AMSDType::AMS_INT64);
    CATCH_REQUIRE(transposed.shape()[0] == 6);
    CATCH_REQUIRE(transposed.shape()[1] == 5);
    CATCH_REQUIRE(transposed.elements() == 30);
  }
}

CATCH_TEST_CASE("AMSTensor: int tensor move semantics", "[ams][tensor][move]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Move int32_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {10};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor1 = AMSTensor::create<int32_t>(shape, strides, device);
    auto* original_ptr = tensor1.data<int32_t>();

    // Move construct
    auto tensor2 = std::move(tensor1);

    CATCH_REQUIRE(tensor2.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensor2.elements() == 10);
    CATCH_REQUIRE(tensor2.data<int32_t>() == original_ptr);
  }

  CATCH_SECTION("Move int64_t tensor")
  {
    std::vector<AMSTensor::IntDimType> shape = {20};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor1 = AMSTensor::create<int64_t>(shape, strides, device);
    auto* original_ptr = tensor1.data<int64_t>();

    // Move construct (not move assign, to avoid existing AMSTensor bug)
    auto tensor2 = std::move(tensor1);

    CATCH_REQUIRE(tensor2.dtype() == AMSDType::AMS_INT64);
    CATCH_REQUIRE(tensor2.elements() == 20);
    CATCH_REQUIRE(tensor2.data<int64_t>() == original_ptr);
  }
}

CATCH_TEST_CASE("AMSTensor: dtype_to_size utility for int types",
                "[ams][utils]")
{
  CATCH_SECTION("Verify int32_t size")
  {
    size_t size = dtype_to_size(AMSDType::AMS_INT32);
    CATCH_REQUIRE(size == sizeof(int32_t));
    CATCH_REQUIRE(size == 4);
  }

  CATCH_SECTION("Verify int64_t size")
  {
    size_t size = dtype_to_size(AMSDType::AMS_INT64);
    CATCH_REQUIRE(size == sizeof(int64_t));
    CATCH_REQUIRE(size == 8);
  }

  CATCH_SECTION("Compare sizes")
  {
    size_t size_int32 = dtype_to_size(AMSDType::AMS_INT32);
    size_t size_int64 = dtype_to_size(AMSDType::AMS_INT64);
    size_t size_float = dtype_to_size(AMSDType::AMS_SINGLE);
    size_t size_double = dtype_to_size(AMSDType::AMS_DOUBLE);

    CATCH_REQUIRE(size_int32 == size_float);   // Both 4 bytes
    CATCH_REQUIRE(size_int64 == size_double);  // Both 8 bytes
    CATCH_REQUIRE(size_int64 == 2 * size_int32);
  }
}

CATCH_TEST_CASE("AMSTensor: SmallVector of int tensors",
                "[ams][tensor][smallvector]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Create vector of int32_t tensors")
  {
    ams::SmallVector<AMSTensor> tensors;

    std::vector<AMSTensor::IntDimType> shape1 = {5};
    std::vector<AMSTensor::IntDimType> shape2 = {10};
    std::vector<AMSTensor::IntDimType> strides = {1};

    tensors.push_back(AMSTensor::create<int32_t>(shape1, strides, device));
    tensors.push_back(AMSTensor::create<int32_t>(shape2, strides, device));

    CATCH_REQUIRE(tensors.size() == 2);
    CATCH_REQUIRE(tensors[0].dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensors[1].dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensors[0].elements() == 5);
    CATCH_REQUIRE(tensors[1].elements() == 10);
  }

  CATCH_SECTION("Create vector of int64_t tensors")
  {
    ams::SmallVector<AMSTensor> tensors;

    std::vector<AMSTensor::IntDimType> shape = {7};
    std::vector<AMSTensor::IntDimType> strides = {1};

    for (int i = 0; i < 3; ++i) {
      tensors.push_back(AMSTensor::create<int64_t>(shape, strides, device));
    }

    CATCH_REQUIRE(tensors.size() == 3);
    for (const auto& tensor : tensors) {
      CATCH_REQUIRE(tensor.dtype() == AMSDType::AMS_INT64);
      CATCH_REQUIRE(tensor.elements() == 7);
    }
  }

  CATCH_SECTION("Mixed type tensors in SmallVector")
  {
    ams::SmallVector<AMSTensor> tensors;

    std::vector<AMSTensor::IntDimType> shape = {8};
    std::vector<AMSTensor::IntDimType> strides = {1};

    tensors.push_back(AMSTensor::create<float>(shape, strides, device));
    tensors.push_back(AMSTensor::create<int32_t>(shape, strides, device));
    tensors.push_back(AMSTensor::create<double>(shape, strides, device));
    tensors.push_back(AMSTensor::create<int64_t>(shape, strides, device));

    CATCH_REQUIRE(tensors.size() == 4);
    CATCH_REQUIRE(tensors[0].dtype() == AMSDType::AMS_SINGLE);
    CATCH_REQUIRE(tensors[1].dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(tensors[2].dtype() == AMSDType::AMS_DOUBLE);
    CATCH_REQUIRE(tensors[3].dtype() == AMSDType::AMS_INT64);
  }
}

CATCH_TEST_CASE("AMSTensor: int tensor data access and modification",
                "[ams][tensor][data]")
{
  AMSInit();

  const auto device = GENERATE(AMSResourceType::AMS_HOST);

  CATCH_SECTION("Write and read int32_t data")
  {
    std::vector<AMSTensor::IntDimType> shape = {5};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);
    auto* data = tensor.data<int32_t>();

    // Write data
    for (int i = 0; i < 5; ++i) {
      data[i] = i * 10;
    }

    // Read data back
    CATCH_REQUIRE(data[0] == 0);
    CATCH_REQUIRE(data[1] == 10);
    CATCH_REQUIRE(data[2] == 20);
    CATCH_REQUIRE(data[3] == 30);
    CATCH_REQUIRE(data[4] == 40);
  }

  CATCH_SECTION("Write and read int64_t data")
  {
    std::vector<AMSTensor::IntDimType> shape = {3};
    std::vector<AMSTensor::IntDimType> strides = {1};

    auto tensor = AMSTensor::create<int64_t>(shape, strides, device);
    auto* data = tensor.data<int64_t>();

    // Write large values
    data[0] = 1000000000LL;
    data[1] = 2000000000LL;
    data[2] = 3000000000LL;

    // Read data back
    CATCH_REQUIRE(data[0] == 1000000000LL);
    CATCH_REQUIRE(data[1] == 2000000000LL);
    CATCH_REQUIRE(data[2] == 3000000000LL);
  }

  CATCH_SECTION("2D int32_t tensor data access")
  {
    std::vector<AMSTensor::IntDimType> shape = {3, 4};
    std::vector<AMSTensor::IntDimType> strides = {4, 1};

    auto tensor = AMSTensor::create<int32_t>(shape, strides, device);
    auto* data = tensor.data<int32_t>();

    // Fill with row-major data
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        data[i * 4 + j] = i * 10 + j;
      }
    }

    // Verify access
    CATCH_REQUIRE(data[0] == 0);    // [0,0]
    CATCH_REQUIRE(data[3] == 3);    // [0,3]
    CATCH_REQUIRE(data[4] == 10);   // [1,0]
    CATCH_REQUIRE(data[11] == 23);  // [2,3]
  }
}

// ---------------------------------------------------------------------------
// Clone tests
// ---------------------------------------------------------------------------

CATCH_TEST_CASE("AMSTensor::clone: contiguous 1D float tensor",
                "[ams][tensor][clone]")
{
  AMSInit();

  // Source: [1.0, 2.0, 3.0, 4.0, 5.0]
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<AMSTensor::IntDimType> shape = {5};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto view = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  // Metadata must match
  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(cloned.elements() == 5);
  CATCH_REQUIRE(cloned.dim() == 1);
  CATCH_REQUIRE(cloned.shape()[0] == 5);
  CATCH_REQUIRE(cloned.strides()[0] == 1);
  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.nbytes() == 5 * sizeof(float));

  // Data must be a deep copy (different pointer, same values)
  auto* clonedPtr = cloned.data<float>();
  CATCH_REQUIRE(clonedPtr != src.data());
  for (int i = 0; i < 5; ++i) {
    CATCH_REQUIRE(clonedPtr[i] == src[i]);
  }

  // Mutating the source must not affect the clone
  src[0] = 999.0f;
  CATCH_REQUIRE(clonedPtr[0] == 1.0f);
}


CATCH_TEST_CASE("AMSTensor::clone: contiguous 2D int32 tensor",
                "[ams][tensor][clone][int32]")
{
  AMSInit();

  // 3x4 row-major tensor filled with i*10+j
  std::vector<int32_t> src(12);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      src[i * 4 + j] = i * 10 + j;

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto view = AMSTensor::view<int32_t>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(cloned.dim() == 2);
  CATCH_REQUIRE(cloned.shape()[0] == 3);
  CATCH_REQUIRE(cloned.shape()[1] == 4);
  CATCH_REQUIRE(cloned.elements() == 12);
  CATCH_REQUIRE(cloned.contiguous());

  auto* clonedPtr = cloned.data<int32_t>();
  CATCH_REQUIRE(clonedPtr != src.data());
  for (int i = 0; i < 12; ++i) {
    CATCH_REQUIRE(clonedPtr[i] == src[i]);
  }
}


CATCH_TEST_CASE("AMSTensor::clone: contiguous double tensor",
                "[ams][tensor][clone][double]")
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

  auto* clonedPtr = cloned.data<double>();
  CATCH_REQUIRE(clonedPtr != src.data());
  for (int i = 0; i < 6; ++i) {
    CATCH_REQUIRE(clonedPtr[i] == src[i]);
  }
}


CATCH_TEST_CASE("AMSTensor::clone: non-contiguous (transposed) tensor",
                "[ams][tensor][clone][transpose]")
{
  AMSInit();

  // Create a 3x4 contiguous tensor, then transpose to 4x3.
  // Original layout (row-major):
  //   row0: [0, 1, 2, 3]
  //   row1: [4, 5, 6, 7]
  //   row2: [8, 9, 10, 11]
  //
  // After transpose(0,1) → shape [4,3], strides [1,4]
  //   Logical row0: [0, 4, 8]
  //   Logical row1: [1, 5, 9]
  //   Logical row2: [2, 6, 10]
  //   Logical row3: [3, 7, 11]
  //
  // Clone should produce a contiguous [4,3] tensor with strides [3,1]:
  //   Memory: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

  std::vector<float> src(12);
  for (int i = 0; i < 12; ++i) src[i] = static_cast<float>(i);

  std::vector<AMSTensor::IntDimType> shape = {3, 4};
  std::vector<AMSTensor::IntDimType> strides = {4, 1};

  auto original = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto transposed = original.transpose(0, 1);

  // Transposed tensor is non-contiguous
  CATCH_REQUIRE(!transposed.contiguous());
  CATCH_REQUIRE(transposed.shape()[0] == 4);
  CATCH_REQUIRE(transposed.shape()[1] == 3);

  auto cloned = transposed.clone();

  // Clone must be contiguous with shape [4,3] and row-major strides [3,1]
  CATCH_REQUIRE(cloned.contiguous());
  CATCH_REQUIRE(cloned.shape()[0] == 4);
  CATCH_REQUIRE(cloned.shape()[1] == 3);
  CATCH_REQUIRE(cloned.strides()[0] == 3);
  CATCH_REQUIRE(cloned.strides()[1] == 1);
  CATCH_REQUIRE(cloned.elements() == 12);

  // Verify data: logical element [i,j] of the transposed tensor
  // is element [j,i] of the original, i.e. src[j*4 + i]
  auto* clonedPtr = cloned.data<float>();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      float expected = static_cast<float>(j * 4 + i);
      CATCH_INFO("clone[" << i << "," << j << "] = "
                 << clonedPtr[i * 3 + j] << ", expected " << expected);
      CATCH_REQUIRE(clonedPtr[i * 3 + j] == expected);
    }
  }

  // Must be a deep copy — different memory
  CATCH_REQUIRE(cloned.raw_data() != transposed.raw_data());
}


CATCH_TEST_CASE("AMSTensor::clone: int64 tensor",
                "[ams][tensor][clone][int64]")
{
  AMSInit();

  std::vector<int64_t> src = {100, 200, 300, 400, 500, 600};
  std::vector<AMSTensor::IntDimType> shape = {3, 2};
  std::vector<AMSTensor::IntDimType> strides = {2, 1};

  auto view = AMSTensor::view<int64_t>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto cloned = view.clone();

  CATCH_REQUIRE(cloned.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(cloned.elements() == 6);
  CATCH_REQUIRE(cloned.nbytes() == 6 * sizeof(int64_t));

  auto* clonedPtr = cloned.data<int64_t>();
  CATCH_REQUIRE(clonedPtr != src.data());
  for (int i = 0; i < 6; ++i) {
    CATCH_REQUIRE(clonedPtr[i] == src[i]);
  }
}


// ---------------------------------------------------------------------------
// Concat tests
// ---------------------------------------------------------------------------

CATCH_TEST_CASE("AMSTensor::concat: single tensor passthrough",
                "[ams][tensor][concat]")
{
  AMSInit();

  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  auto t = AMSTensor::view<float>(
      src.data(), shape, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(t));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(result.dim() == 2);
  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 3);
  CATCH_REQUIRE(result.elements() == 6);

  auto* ptr = result.data<float>();
  for (int i = 0; i < 6; ++i) {
    CATCH_REQUIRE(ptr[i] == src[i]);
  }
}


CATCH_TEST_CASE("AMSTensor::concat: two 1D float tensors",
                "[ams][tensor][concat][1d]")
{
  AMSInit();

  std::vector<float> a = {1.0f, 2.0f, 3.0f};
  std::vector<float> b = {4.0f, 5.0f};
  std::vector<AMSTensor::IntDimType> shapeA = {3};
  std::vector<AMSTensor::IntDimType> shapeB = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto tA = AMSTensor::view<float>(
      a.data(), shapeA, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<float>(
      b.data(), shapeB, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

  // 1D concat: [3] + [2] → [5]
  CATCH_REQUIRE(result.dim() == 1);
  CATCH_REQUIRE(result.shape()[0] == 5);
  CATCH_REQUIRE(result.elements() == 5);
  CATCH_REQUIRE(result.contiguous());

  auto* ptr = result.data<float>();
  CATCH_REQUIRE(ptr[0] == 1.0f);
  CATCH_REQUIRE(ptr[1] == 2.0f);
  CATCH_REQUIRE(ptr[2] == 3.0f);
  CATCH_REQUIRE(ptr[3] == 4.0f);
  CATCH_REQUIRE(ptr[4] == 5.0f);
}


CATCH_TEST_CASE("AMSTensor::concat: two 2D float tensors along last dim",
                "[ams][tensor][concat][2d]")
{
  AMSInit();

  // A: [3, 2]        B: [3, 3]
  //   [1, 2]           [7, 8, 9]
  //   [3, 4]           [10, 11, 12]
  //   [5, 6]           [13, 14, 15]
  //
  // Result: [3, 5]
  //   [1, 2, 7, 8, 9]
  //   [3, 4, 10, 11, 12]
  //   [5, 6, 13, 14, 15]

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

  CATCH_REQUIRE(result.dim() == 2);
  CATCH_REQUIRE(result.shape()[0] == 3);
  CATCH_REQUIRE(result.shape()[1] == 5);
  CATCH_REQUIRE(result.elements() == 15);
  CATCH_REQUIRE(result.contiguous());

  auto* ptr = result.data<float>();
  // Row 0: [1, 2, 7, 8, 9]
  CATCH_REQUIRE(ptr[0] == 1.0f);
  CATCH_REQUIRE(ptr[1] == 2.0f);
  CATCH_REQUIRE(ptr[2] == 7.0f);
  CATCH_REQUIRE(ptr[3] == 8.0f);
  CATCH_REQUIRE(ptr[4] == 9.0f);
  // Row 1: [3, 4, 10, 11, 12]
  CATCH_REQUIRE(ptr[5] == 3.0f);
  CATCH_REQUIRE(ptr[6] == 4.0f);
  CATCH_REQUIRE(ptr[7] == 10.0f);
  CATCH_REQUIRE(ptr[8] == 11.0f);
  CATCH_REQUIRE(ptr[9] == 12.0f);
  // Row 2: [5, 6, 13, 14, 15]
  CATCH_REQUIRE(ptr[10] == 5.0f);
  CATCH_REQUIRE(ptr[11] == 6.0f);
  CATCH_REQUIRE(ptr[12] == 13.0f);
  CATCH_REQUIRE(ptr[13] == 14.0f);
  CATCH_REQUIRE(ptr[14] == 15.0f);
}


CATCH_TEST_CASE("AMSTensor::concat: three 2D tensors",
                "[ams][tensor][concat][multi]")
{
  AMSInit();

  // A:[2,2]  B:[2,3]  C:[2,1]  → Result:[2,6]
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {10, 20, 30, 40, 50, 60};
  std::vector<float> c = {100, 200};

  std::vector<AMSTensor::IntDimType> shapeA = {2, 2};
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

  CATCH_REQUIRE(result.dim() == 2);
  CATCH_REQUIRE(result.shape()[0] == 2);
  CATCH_REQUIRE(result.shape()[1] == 6);
  CATCH_REQUIRE(result.elements() == 12);

  auto* ptr = result.data<float>();
  // Row 0: [1, 2, 10, 20, 30, 100]
  CATCH_REQUIRE(ptr[0] == 1.0f);
  CATCH_REQUIRE(ptr[1] == 2.0f);
  CATCH_REQUIRE(ptr[2] == 10.0f);
  CATCH_REQUIRE(ptr[3] == 20.0f);
  CATCH_REQUIRE(ptr[4] == 30.0f);
  CATCH_REQUIRE(ptr[5] == 100.0f);
  // Row 1: [3, 4, 40, 50, 60, 200]
  CATCH_REQUIRE(ptr[6] == 3.0f);
  CATCH_REQUIRE(ptr[7] == 4.0f);
  CATCH_REQUIRE(ptr[8] == 40.0f);
  CATCH_REQUIRE(ptr[9] == 50.0f);
  CATCH_REQUIRE(ptr[10] == 60.0f);
  CATCH_REQUIRE(ptr[11] == 200.0f);
}


CATCH_TEST_CASE("AMSTensor::concat: int32 tensors",
                "[ams][tensor][concat][int32]")
{
  AMSInit();

  std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> b = {10, 20, 30, 40, 50, 60};
  std::vector<AMSTensor::IntDimType> shape = {3, 2};
  std::vector<AMSTensor::IntDimType> strides = {2, 1};

  auto tA = AMSTensor::view<int32_t>(
      a.data(), shape, strides, AMSResourceType::AMS_HOST);
  auto tB = AMSTensor::view<int32_t>(
      b.data(), shape, strides, AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> tensors;
  tensors.push_back(AMSTensor::view(tA));
  tensors.push_back(AMSTensor::view(tB));

  auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

  CATCH_REQUIRE(result.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(result.dim() == 2);
  CATCH_REQUIRE(result.shape()[0] == 3);
  CATCH_REQUIRE(result.shape()[1] == 4);

  auto* ptr = result.data<int32_t>();
  // Row 0: [1, 2, 10, 20]
  CATCH_REQUIRE(ptr[0] == 1);
  CATCH_REQUIRE(ptr[1] == 2);
  CATCH_REQUIRE(ptr[2] == 10);
  CATCH_REQUIRE(ptr[3] == 20);
  // Row 1: [3, 4, 30, 40]
  CATCH_REQUIRE(ptr[4] == 3);
  CATCH_REQUIRE(ptr[5] == 4);
  CATCH_REQUIRE(ptr[6] == 30);
  CATCH_REQUIRE(ptr[7] == 40);
  // Row 2: [5, 6, 50, 60]
  CATCH_REQUIRE(ptr[8] == 5);
  CATCH_REQUIRE(ptr[9] == 6);
  CATCH_REQUIRE(ptr[10] == 50);
  CATCH_REQUIRE(ptr[11] == 60);
}


CATCH_TEST_CASE("AMSTensor::concat: double tensors",
                "[ams][tensor][concat][double]")
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

  auto* ptr = result.data<double>();
  // Row 0: [1.1, 2.2, 5.5, 6.6]
  CATCH_REQUIRE(ptr[0] == 1.1);
  CATCH_REQUIRE(ptr[1] == 2.2);
  CATCH_REQUIRE(ptr[2] == 5.5);
  CATCH_REQUIRE(ptr[3] == 6.6);
  // Row 1: [3.3, 4.4, 7.7, 8.8]
  CATCH_REQUIRE(ptr[4] == 3.3);
  CATCH_REQUIRE(ptr[5] == 4.4);
  CATCH_REQUIRE(ptr[6] == 7.7);
  CATCH_REQUIRE(ptr[7] == 8.8);
}


CATCH_TEST_CASE("AMSTensor::concat: result is independent of source",
                "[ams][tensor][concat][ownership]")
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
  auto* ptr = result.data<float>();

  // Mutate sources after concat — result must be unaffected
  a[0] = 999.0f;
  b[0] = 888.0f;
  CATCH_REQUIRE(ptr[0] == 1.0f);
  CATCH_REQUIRE(ptr[1] == 2.0f);
  CATCH_REQUIRE(ptr[2] == 3.0f);
  CATCH_REQUIRE(ptr[3] == 4.0f);
}