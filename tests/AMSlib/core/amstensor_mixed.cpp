/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

// =========================================================================
// SmallVector holding mixed-dtype tensors
// =========================================================================

CATCH_TEST_CASE("mixed: SmallVector of all four dtypes",
                "[ams][tensor][mixed][smallvector]")
{
  AMSInit();
  ams::SmallVector<AMSTensor> tensors;
  std::vector<AMSTensor::IntDimType> shape = {8};
  std::vector<AMSTensor::IntDimType> strides = {1};

  tensors.push_back(
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST));
  tensors.push_back(
      AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST));
  tensors.push_back(
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST));
  tensors.push_back(
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST));

  CATCH_REQUIRE(tensors.size() == 4);
  CATCH_REQUIRE(tensors[0].dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(tensors[1].dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(tensors[2].dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(tensors[3].dtype() == AMSDType::AMS_INT64);

  // All have the same number of elements
  for (auto& t : tensors)
    CATCH_REQUIRE(t.elements() == 8);

  // But different byte sizes
  CATCH_REQUIRE(tensors[0].nbytes() == 8 * sizeof(float));
  CATCH_REQUIRE(tensors[1].nbytes() == 8 * sizeof(double));
  CATCH_REQUIRE(tensors[2].nbytes() == 8 * sizeof(int32_t));
  CATCH_REQUIRE(tensors[3].nbytes() == 8 * sizeof(int64_t));

  // element_size matches each type
  CATCH_REQUIRE(tensors[0].element_size() == sizeof(float));
  CATCH_REQUIRE(tensors[1].element_size() == sizeof(double));
  CATCH_REQUIRE(tensors[2].element_size() == sizeof(int32_t));
  CATCH_REQUIRE(tensors[3].element_size() == sizeof(int64_t));
}


CATCH_TEST_CASE("mixed: SmallVector with different shapes per dtype",
                "[ams][tensor][mixed][smallvector]")
{
  AMSInit();
  ams::SmallVector<AMSTensor> tensors;

  std::vector<AMSTensor::IntDimType> shapeA = {3, 4};
  std::vector<AMSTensor::IntDimType> stridesA = {4, 1};

  std::vector<AMSTensor::IntDimType> shapeB = {5};
  std::vector<AMSTensor::IntDimType> stridesB = {1};

  std::vector<AMSTensor::IntDimType> shapeC = {2, 2, 2};
  std::vector<AMSTensor::IntDimType> stridesC = {4, 2, 1};

  tensors.push_back(
      AMSTensor::create<float>(shapeA, stridesA, AMSResourceType::AMS_HOST));
  tensors.push_back(
      AMSTensor::create<int32_t>(shapeB, stridesB, AMSResourceType::AMS_HOST));
  tensors.push_back(
      AMSTensor::create<double>(shapeC, stridesC, AMSResourceType::AMS_HOST));

  CATCH_REQUIRE(tensors[0].dim() == 2);
  CATCH_REQUIRE(tensors[0].elements() == 12);
  CATCH_REQUIRE(tensors[1].dim() == 1);
  CATCH_REQUIRE(tensors[1].elements() == 5);
  CATCH_REQUIRE(tensors[2].dim() == 3);
  CATCH_REQUIRE(tensors[2].elements() == 8);
}


// =========================================================================
// Clone preserves dtype across all types
// =========================================================================

CATCH_TEST_CASE("mixed: clone preserves dtype for all four types",
                "[ams][tensor][mixed][clone]")
{
  AMSInit();

  // float
  std::vector<float> fData = {1.0f, 2.0f, 3.0f};
  std::vector<AMSTensor::IntDimType> shape = {3};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto fView = AMSTensor::view<float>(fData.data(),
                                      shape,
                                      strides,
                                      AMSResourceType::AMS_HOST);
  auto fClone = fView.clone();
  CATCH_REQUIRE(fClone.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(fClone.data<float>()[2] == 3.0f);

  // double
  std::vector<double> dData = {10.0, 20.0, 30.0};
  auto dView = AMSTensor::view<double>(dData.data(),
                                       shape,
                                       strides,
                                       AMSResourceType::AMS_HOST);
  auto dClone = dView.clone();
  CATCH_REQUIRE(dClone.dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(dClone.data<double>()[2] == 30.0);

  // int32
  std::vector<int32_t> i32Data = {100, 200, 300};
  auto i32View = AMSTensor::view<int32_t>(i32Data.data(),
                                          shape,
                                          strides,
                                          AMSResourceType::AMS_HOST);
  auto i32Clone = i32View.clone();
  CATCH_REQUIRE(i32Clone.dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(i32Clone.data<int32_t>()[2] == 300);

  // int64
  std::vector<int64_t> i64Data = {1000, 2000, 3000};
  auto i64View = AMSTensor::view<int64_t>(i64Data.data(),
                                          shape,
                                          strides,
                                          AMSResourceType::AMS_HOST);
  auto i64Clone = i64View.clone();
  CATCH_REQUIRE(i64Clone.dtype() == AMSDType::AMS_INT64);
  CATCH_REQUIRE(i64Clone.data<int64_t>()[2] == 3000);
}

CATCH_TEST_CASE("mixed: clone all types in a SmallVector",
                "[ams][tensor][mixed][clone]")
{
  AMSInit();
  std::vector<float> fData = {1.0f, 2.0f};
  std::vector<double> dData = {3.0, 4.0};
  std::vector<int32_t> i32Data = {5, 6};
  std::vector<int64_t> i64Data = {7, 8};

  std::vector<AMSTensor::IntDimType> shape = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  ams::SmallVector<AMSTensor> originals;
  originals.push_back(AMSTensor::view<float>(
      fData.data(), shape, strides, AMSResourceType::AMS_HOST));
  originals.push_back(AMSTensor::view<double>(
      dData.data(), shape, strides, AMSResourceType::AMS_HOST));
  originals.push_back(AMSTensor::view<int32_t>(
      i32Data.data(), shape, strides, AMSResourceType::AMS_HOST));
  originals.push_back(AMSTensor::view<int64_t>(
      i64Data.data(), shape, strides, AMSResourceType::AMS_HOST));

  ams::SmallVector<AMSTensor> clones;
  for (auto& t : originals)
    clones.push_back(t.clone());

  // Mutate all source buffers
  fData[0] = 999.0f;
  dData[0] = 999.0;
  i32Data[0] = 999;
  i64Data[0] = 999;

  // Clones must be unaffected
  CATCH_REQUIRE(clones[0].data<float>()[0] == 1.0f);
  CATCH_REQUIRE(clones[1].data<double>()[0] == 3.0);
  CATCH_REQUIRE(clones[2].data<int32_t>()[0] == 5);
  CATCH_REQUIRE(clones[3].data<int64_t>()[0] == 7);

  // Dtype preserved
  CATCH_REQUIRE(clones[0].dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(clones[1].dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(clones[2].dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(clones[3].dtype() == AMSDType::AMS_INT64);
}


// =========================================================================
// Concat: same operation across all four types
// =========================================================================

CATCH_TEST_CASE("mixed: concat two 2D tensors for each dtype",
                "[ams][tensor][mixed][concat]")
{
  AMSInit();

  // Pattern: A:[2,2] + B:[2,3] → [2,5]

  std::vector<AMSTensor::IntDimType> shapeA = {2, 2};
  std::vector<AMSTensor::IntDimType> stridesA = {2, 1};

  std::vector<AMSTensor::IntDimType> shapeB = {2, 3};
  std::vector<AMSTensor::IntDimType> stridesB = {3, 1};

  CATCH_SECTION("float")
  {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40, 50, 60};

    auto tA = AMSTensor::view<float>(a.data(),
                                     shapeA,
                                     stridesA,
                                     AMSResourceType::AMS_HOST);
    auto tB = AMSTensor::view<float>(b.data(),
                                     shapeB,
                                     stridesB,
                                     AMSResourceType::AMS_HOST);

    ams::SmallVector<AMSTensor> tensors;
    tensors.push_back(AMSTensor::view(tA));
    tensors.push_back(AMSTensor::view(tB));

    auto result = AMSTensor::concat(tensors, AMSDType::AMS_SINGLE);

    CATCH_REQUIRE(result.dtype() == AMSDType::AMS_SINGLE);
    CATCH_REQUIRE(result.shape()[0] == 2);
    CATCH_REQUIRE(result.shape()[1] == 5);
    auto* p = result.data<float>();
    CATCH_REQUIRE(p[0] == 1.0f);
    CATCH_REQUIRE(p[2] == 10.0f);
    CATCH_REQUIRE(p[5] == 3.0f);
    CATCH_REQUIRE(p[7] == 40.0f);
  }

  CATCH_SECTION("double")
  {
    std::vector<double> a = {1, 2, 3, 4};
    std::vector<double> b = {10, 20, 30, 40, 50, 60};

    auto tA = AMSTensor::view<double>(a.data(),
                                      shapeA,
                                      stridesA,
                                      AMSResourceType::AMS_HOST);
    auto tB = AMSTensor::view<double>(b.data(),
                                      shapeB,
                                      stridesB,
                                      AMSResourceType::AMS_HOST);

    ams::SmallVector<AMSTensor> tensors;
    tensors.push_back(AMSTensor::view(tA));
    tensors.push_back(AMSTensor::view(tB));

    auto result = AMSTensor::concat(tensors, AMSDType::AMS_DOUBLE);

    CATCH_REQUIRE(result.dtype() == AMSDType::AMS_DOUBLE);
    CATCH_REQUIRE(result.shape()[0] == 2);
    CATCH_REQUIRE(result.shape()[1] == 5);
    auto* p = result.data<double>();
    CATCH_REQUIRE(p[0] == 1.0);
    CATCH_REQUIRE(p[2] == 10.0);
    CATCH_REQUIRE(p[5] == 3.0);
    CATCH_REQUIRE(p[7] == 40.0);
  }

  CATCH_SECTION("int32")
  {
    std::vector<int32_t> a = {1, 2, 3, 4};
    std::vector<int32_t> b = {10, 20, 30, 40, 50, 60};

    auto tA = AMSTensor::view<int32_t>(a.data(),
                                       shapeA,
                                       stridesA,
                                       AMSResourceType::AMS_HOST);
    auto tB = AMSTensor::view<int32_t>(b.data(),
                                       shapeB,
                                       stridesB,
                                       AMSResourceType::AMS_HOST);

    ams::SmallVector<AMSTensor> tensors;
    tensors.push_back(AMSTensor::view(tA));
    tensors.push_back(AMSTensor::view(tB));

    auto result = AMSTensor::concat(tensors, AMSDType::AMS_INT32);

    CATCH_REQUIRE(result.dtype() == AMSDType::AMS_INT32);
    CATCH_REQUIRE(result.shape()[0] == 2);
    CATCH_REQUIRE(result.shape()[1] == 5);
    auto* p = result.data<int32_t>();
    CATCH_REQUIRE(p[0] == 1);
    CATCH_REQUIRE(p[2] == 10);
    CATCH_REQUIRE(p[5] == 3);
    CATCH_REQUIRE(p[7] == 40);
  }

  CATCH_SECTION("int64")
  {
    std::vector<int64_t> a = {1, 2, 3, 4};
    std::vector<int64_t> b = {10, 20, 30, 40, 50, 60};

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
    CATCH_REQUIRE(p[0] == 1);
    CATCH_REQUIRE(p[2] == 10);
    CATCH_REQUIRE(p[5] == 3);
    CATCH_REQUIRE(p[7] == 40);
  }
}


// =========================================================================
// Interleaved operations across types
// =========================================================================

CATCH_TEST_CASE("mixed: create, write, clone, verify across types",
                "[ams][tensor][mixed][interleaved]")
{
  AMSInit();

  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  // Create a float tensor and an int32 tensor with the same shape
  auto fTensor =
      AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto iTensor =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);

  // Write the same logical values
  auto* fPtr = fTensor.data<float>();
  auto* iPtr = iTensor.data<int32_t>();
  for (int i = 0; i < 6; ++i) {
    fPtr[i] = static_cast<float>(i + 1);
    iPtr[i] = i + 1;
  }

  // Clone both
  auto fClone = fTensor.clone();
  auto iClone = iTensor.clone();

  // Verify dtypes didn't get mixed up
  CATCH_REQUIRE(fClone.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(iClone.dtype() == AMSDType::AMS_INT32);

  // Verify data
  for (int i = 0; i < 6; ++i) {
    CATCH_REQUIRE(fClone.data<float>()[i] == static_cast<float>(i + 1));
    CATCH_REQUIRE(iClone.data<int32_t>()[i] == i + 1);
  }

  // Clones are independent
  fPtr[0] = 999.0f;
  iPtr[0] = 999;
  CATCH_REQUIRE(fClone.data<float>()[0] == 1.0f);
  CATCH_REQUIRE(iClone.data<int32_t>()[0] == 1);
}


CATCH_TEST_CASE("mixed: concat float then concat int32 independently",
                "[ams][tensor][mixed][interleaved][concat]")
{
  AMSInit();
  // Two float tensors
  std::vector<float> fA = {1.0f, 2.0f};
  std::vector<float> fB = {3.0f, 4.0f};

  // Two int32 tensors with same values
  std::vector<int32_t> iA = {1, 2};
  std::vector<int32_t> iB = {3, 4};

  std::vector<AMSTensor::IntDimType> shape = {2};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto ftA = AMSTensor::view<float>(fA.data(),
                                    shape,
                                    strides,
                                    AMSResourceType::AMS_HOST);
  auto ftB = AMSTensor::view<float>(fB.data(),
                                    shape,
                                    strides,
                                    AMSResourceType::AMS_HOST);
  auto itA = AMSTensor::view<int32_t>(iA.data(),
                                      shape,
                                      strides,
                                      AMSResourceType::AMS_HOST);
  auto itB = AMSTensor::view<int32_t>(iB.data(),
                                      shape,
                                      strides,
                                      AMSResourceType::AMS_HOST);

  ams::SmallVector<AMSTensor> fTensors;
  fTensors.push_back(AMSTensor::view(ftA));
  fTensors.push_back(AMSTensor::view(ftB));

  ams::SmallVector<AMSTensor> iTensors;
  iTensors.push_back(AMSTensor::view(itA));
  iTensors.push_back(AMSTensor::view(itB));

  auto fResult = AMSTensor::concat(fTensors, AMSDType::AMS_SINGLE);
  auto iResult = AMSTensor::concat(iTensors, AMSDType::AMS_INT32);

  // Both produce [4] shaped results with values [1, 2, 3, 4]
  CATCH_REQUIRE(fResult.shape()[0] == 4);
  CATCH_REQUIRE(iResult.shape()[0] == 4);
  CATCH_REQUIRE(fResult.dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(iResult.dtype() == AMSDType::AMS_INT32);

  for (int i = 0; i < 4; ++i) {
    CATCH_REQUIRE(fResult.data<float>()[i] == static_cast<float>(i + 1));
    CATCH_REQUIRE(iResult.data<int32_t>()[i] == i + 1);
  }
}


// =========================================================================
// Move across types in a SmallVector
// =========================================================================

CATCH_TEST_CASE("mixed: move tensors into SmallVector preserves types",
                "[ams][tensor][mixed][move]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {4};
  std::vector<AMSTensor::IntDimType> strides = {1};

  auto f = AMSTensor::create<float>(shape, strides, AMSResourceType::AMS_HOST);
  auto d = AMSTensor::create<double>(shape, strides, AMSResourceType::AMS_HOST);
  auto i32 =
      AMSTensor::create<int32_t>(shape, strides, AMSResourceType::AMS_HOST);
  auto i64 =
      AMSTensor::create<int64_t>(shape, strides, AMSResourceType::AMS_HOST);

  auto* fPtr = f.data<float>();
  auto* dPtr = d.data<double>();
  auto* i32Ptr = i32.data<int32_t>();
  auto* i64Ptr = i64.data<int64_t>();

  ams::SmallVector<AMSTensor> vec;
  vec.push_back(std::move(f));
  vec.push_back(std::move(d));
  vec.push_back(std::move(i32));
  vec.push_back(std::move(i64));

  // Types preserved after move
  CATCH_REQUIRE(vec[0].dtype() == AMSDType::AMS_SINGLE);
  CATCH_REQUIRE(vec[1].dtype() == AMSDType::AMS_DOUBLE);
  CATCH_REQUIRE(vec[2].dtype() == AMSDType::AMS_INT32);
  CATCH_REQUIRE(vec[3].dtype() == AMSDType::AMS_INT64);

  // Pointers transferred (not copied)
  CATCH_REQUIRE(vec[0].data<float>() == fPtr);
  CATCH_REQUIRE(vec[1].data<double>() == dPtr);
  CATCH_REQUIRE(vec[2].data<int32_t>() == i32Ptr);
  CATCH_REQUIRE(vec[3].data<int64_t>() == i64Ptr);

  // Sizes preserved
  for (auto& t : vec) {
    CATCH_REQUIRE(t.elements() == 4);
    CATCH_REQUIRE(t.dim() == 1);
  }

  // Byte sizes differ
  CATCH_REQUIRE(vec[0].nbytes() == 4 * sizeof(float));
  CATCH_REQUIRE(vec[1].nbytes() == 4 * sizeof(double));
  CATCH_REQUIRE(vec[2].nbytes() == 4 * sizeof(int32_t));
  CATCH_REQUIRE(vec[3].nbytes() == 4 * sizeof(int64_t));
}


// =========================================================================
// Transpose + clone across types
// =========================================================================

CATCH_TEST_CASE("mixed: transpose then clone preserves dtype",
                "[ams][tensor][mixed][transpose][clone]")
{
  AMSInit();
  std::vector<AMSTensor::IntDimType> shape = {2, 3};
  std::vector<AMSTensor::IntDimType> strides = {3, 1};

  // float: 2x3 → transpose → 3x2 → clone
  std::vector<float> fSrc = {1, 2, 3, 4, 5, 6};
  auto fOrig = AMSTensor::view<float>(fSrc.data(),
                                      shape,
                                      strides,
                                      AMSResourceType::AMS_HOST);
  auto fTransposed = fOrig.transpose(0, 1);
  auto fClone = fTransposed.clone();

  // int32: same layout
  std::vector<int32_t> iSrc = {1, 2, 3, 4, 5, 6};
  auto iOrig = AMSTensor::view<int32_t>(iSrc.data(),
                                        shape,
                                        strides,
                                        AMSResourceType::AMS_HOST);
  auto iTransposed = iOrig.transpose(0, 1);
  auto iClone = iTransposed.clone();

  // Both clones: shape [3,2], contiguous
  CATCH_REQUIRE(fClone.shape()[0] == 3);
  CATCH_REQUIRE(fClone.shape()[1] == 2);
  CATCH_REQUIRE(fClone.contiguous());
  CATCH_REQUIRE(fClone.dtype() == AMSDType::AMS_SINGLE);

  CATCH_REQUIRE(iClone.shape()[0] == 3);
  CATCH_REQUIRE(iClone.shape()[1] == 2);
  CATCH_REQUIRE(iClone.contiguous());
  CATCH_REQUIRE(iClone.dtype() == AMSDType::AMS_INT32);

  // Logical element [i,j] of transposed = src[j*3 + i]
  // Both should have identical values (just different types)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      float fExpected = static_cast<float>(j * 3 + i + 1);
      int32_t iExpected = j * 3 + i + 1;
      CATCH_INFO("[" << i << "," << j << "]");
      CATCH_REQUIRE(fClone.data<float>()[i * 2 + j] == fExpected);
      CATCH_REQUIRE(iClone.data<int32_t>()[i * 2 + j] == iExpected);
    }
  }
}


// =========================================================================
// dtype_to_size: cross-type relationships
// =========================================================================

CATCH_TEST_CASE("dtype_to_size: cross-type size relationships",
                "[ams][utils][mixed]")
{
  size_t sFloat = dtype_to_size(AMSDType::AMS_SINGLE);
  size_t sDouble = dtype_to_size(AMSDType::AMS_DOUBLE);
  size_t sInt32 = dtype_to_size(AMSDType::AMS_INT32);
  size_t sInt64 = dtype_to_size(AMSDType::AMS_INT64);

  // float == int32 == 4 bytes
  CATCH_REQUIRE(sFloat == sInt32);
  CATCH_REQUIRE(sFloat == 4);

  // double == int64 == 8 bytes
  CATCH_REQUIRE(sDouble == sInt64);
  CATCH_REQUIRE(sDouble == 8);

  // 8-byte types are twice the 4-byte types
  CATCH_REQUIRE(sDouble == 2 * sFloat);
  CATCH_REQUIRE(sInt64 == 2 * sInt32);
}
