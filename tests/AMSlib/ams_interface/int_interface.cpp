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

using namespace ams;

// Global test fixture to initialize AMS once for all tests
struct AMSGlobalFixture {
  AMSGlobalFixture() { AMSInit(); }
};

// This creates a single global instance that will initialize AMS before any tests run
static AMSGlobalFixture amsGlobalFixture;

// Simple computation function for int32_t
void compute_int32(int32_t* input, int32_t* output, int num_elements)
{
  for (int i = 0; i < num_elements; ++i) {
    // Simple computation: output = input * 2 + 1
    output[i] = input[i] * 2 + 1;
  }
}

// Simple computation function for int64_t
void compute_int64(int64_t* input, int64_t* output, int num_elements)
{
  for (int i = 0; i < num_elements; ++i) {
    // Simple computation: output = input * 3 + 10
    output[i] = input[i] * 3 + 10;
  }
}

CATCH_TEST_CASE("AMS API: int32_t tensor execution without model",
                "[ams][api][int32]")
{
  const auto resource = GENERATE(AMSResourceType::AMS_HOST);
  constexpr int num_elements = 100;

  CATCH_SECTION("Execute with int32_t inputs and outputs")
  {
    // Allocate and initialize input data
    std::vector<int32_t> input_data(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      input_data[i] = i;
    }

    // Allocate output data
    std::vector<int32_t> output_data(num_elements, 0);

    // Create AMS tensors
    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;

    SmallVector<AMSTensor::IntDimType> shape_1d{num_elements};
    SmallVector<AMSTensor::IntDimType> strides_1d{1};

    inputs.push_back(AMSTensor::view<int32_t>(
        input_data.data(), shape_1d, strides_1d, resource));

    outputs.push_back(AMSTensor::view<int32_t>(
        output_data.data(), shape_1d, strides_1d, resource));

    // Define computation lambda
    DomainLambda computation = [&](const SmallVector<AMSTensor>& ins,
                                   SmallVector<AMSTensor>& io,
                                   SmallVector<AMSTensor>& outs) {
      CATCH_REQUIRE(ins.size() == 1);
      CATCH_REQUIRE(outs.size() == 1);
      CATCH_REQUIRE(ins[0].dtype() == AMSDType::AMS_INT32);
      CATCH_REQUIRE(outs[0].dtype() == AMSDType::AMS_INT32);

      int32_t* in_ptr = ins[0].data<int32_t>();
      int32_t* out_ptr = outs[0].data<int32_t>();
      int count = ins[0].elements();

      compute_int32(in_ptr, out_ptr, count);
    };

    // Create model and executor (with threshold = 1.0, always use physics)
    AMSCAbstrModel model = AMSRegisterAbstractModel(
        "int32_test", 1.0, "", false);  // No model path, no storage
    CATCH_REQUIRE(model >= 0);

    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    CATCH_REQUIRE(executor >= 0);

    // Execute
    AMSExecute(executor, computation, inputs, inouts, outputs);

    // Verify results
    for (int i = 0; i < num_elements; ++i) {
      int32_t expected = i * 2 + 1;
      CATCH_REQUIRE(output_data[i] == expected);
    }

    // Note: Not destroying executor to avoid triggering AMSFinalize between tests
    // The executor will be cleaned up at program exit
  }
}

CATCH_TEST_CASE("AMS API: int64_t tensor execution without model",
                "[ams][api][int64]")
{
  const auto resource = GENERATE(AMSResourceType::AMS_HOST);
  constexpr int num_elements = 150;

  CATCH_SECTION("Execute with int64_t inputs and outputs")
  {
    // Allocate and initialize input data with large values
    std::vector<int64_t> input_data(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      input_data[i] = static_cast<int64_t>(i) * 1000000;
    }

    // Allocate output data
    std::vector<int64_t> output_data(num_elements, 0);

    // Create AMS tensors
    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;

    SmallVector<AMSTensor::IntDimType> shape_1d{num_elements};
    SmallVector<AMSTensor::IntDimType> strides_1d{1};

    inputs.push_back(AMSTensor::view<int64_t>(
        input_data.data(), shape_1d, strides_1d, resource));

    outputs.push_back(AMSTensor::view<int64_t>(
        output_data.data(), shape_1d, strides_1d, resource));

    // Define computation lambda
    DomainLambda computation = [&](const SmallVector<AMSTensor>& ins,
                                   SmallVector<AMSTensor>& io,
                                   SmallVector<AMSTensor>& outs) {
      CATCH_REQUIRE(ins.size() == 1);
      CATCH_REQUIRE(outs.size() == 1);
      CATCH_REQUIRE(ins[0].dtype() == AMSDType::AMS_INT64);
      CATCH_REQUIRE(outs[0].dtype() == AMSDType::AMS_INT64);

      int64_t* in_ptr = ins[0].data<int64_t>();
      int64_t* out_ptr = outs[0].data<int64_t>();
      int count = ins[0].elements();

      compute_int64(in_ptr, out_ptr, count);
    };

    // Create model and executor (with threshold = 1.0, always use physics)
    AMSCAbstrModel model = AMSRegisterAbstractModel(
        "int64_test", 1.0, "", false);  // No model path, no storage
    CATCH_REQUIRE(model >= 0);

    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    CATCH_REQUIRE(executor >= 0);

    // Execute
    AMSExecute(executor, computation, inputs, inouts, outputs);

    // Verify results
    for (int i = 0; i < num_elements; ++i) {
      int64_t expected = static_cast<int64_t>(i) * 1000000 * 3 + 10;
      CATCH_REQUIRE(output_data[i] == expected);
    }

    // Note: Not destroying executor to avoid triggering AMSFinalize between tests
    // The executor will be cleaned up at program exit
  }
}

CATCH_TEST_CASE("AMS API: 2D int32_t tensor execution", "[ams][api][int32][2d]")
{
  const auto resource = GENERATE(AMSResourceType::AMS_HOST);
  constexpr int rows = 10;
  constexpr int cols = 8;
  constexpr int num_elements = rows * cols;

  CATCH_SECTION("Execute with 2D int32_t tensors")
  {
    // Allocate 2D input data
    std::vector<int32_t> input_data(num_elements);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_data[i * cols + j] = i * 10 + j;
      }
    }

    std::vector<int32_t> output_data(num_elements, 0);

    // Create 2D tensors
    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;

    SmallVector<AMSTensor::IntDimType> shape_2d{rows, cols};
    SmallVector<AMSTensor::IntDimType> strides_2d{cols, 1};

    inputs.push_back(AMSTensor::view<int32_t>(
        input_data.data(), shape_2d, strides_2d, resource));

    outputs.push_back(AMSTensor::view<int32_t>(
        output_data.data(), shape_2d, strides_2d, resource));

    // Computation: element-wise doubling
    DomainLambda computation = [&](const SmallVector<AMSTensor>& ins,
                                   SmallVector<AMSTensor>& io,
                                   SmallVector<AMSTensor>& outs) {
      CATCH_REQUIRE(ins[0].shape().size() == 2);
      CATCH_REQUIRE(ins[0].shape()[0] == rows);
      CATCH_REQUIRE(ins[0].shape()[1] == cols);

      int32_t* in_ptr = ins[0].data<int32_t>();
      int32_t* out_ptr = outs[0].data<int32_t>();

      for (int i = 0; i < num_elements; ++i) {
        out_ptr[i] = in_ptr[i] * 2;
      }
    };

    AMSCAbstrModel model =
        AMSRegisterAbstractModel("int32_2d_test", 1.0, "", false);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

    AMSExecute(executor, computation, inputs, inouts, outputs);

    // Verify
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i * cols + j;
        int32_t expected = (i * 10 + j) * 2;
        CATCH_REQUIRE(output_data[idx] == expected);
      }
    }

    // Note: Not destroying executor to avoid triggering AMSFinalize between tests
    // The executor will be cleaned up at program exit
  }
}

CATCH_TEST_CASE("AMS API: Mixed type tensors", "[ams][api][mixed]")
{
  const auto resource = GENERATE(AMSResourceType::AMS_HOST);
  constexpr int num_elements = 50;

  CATCH_SECTION("Execute with mixed float and int32_t tensors")
  {
    // Float input
    std::vector<float> float_input(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      float_input[i] = static_cast<float>(i) * 1.5f;
    }

    // Int32 input
    std::vector<int32_t> int_input(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      int_input[i] = i * 2;
    }

    // Int32 output
    std::vector<int32_t> output_data(num_elements, 0);

    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;

    SmallVector<AMSTensor::IntDimType> shape_1d{num_elements};
    SmallVector<AMSTensor::IntDimType> strides_1d{1};

    inputs.push_back(AMSTensor::view<float>(
        float_input.data(), shape_1d, strides_1d, resource));

    inputs.push_back(AMSTensor::view<int32_t>(
        int_input.data(), shape_1d, strides_1d, resource));

    outputs.push_back(AMSTensor::view<int32_t>(
        output_data.data(), shape_1d, strides_1d, resource));

    DomainLambda computation = [&](const SmallVector<AMSTensor>& ins,
                                   SmallVector<AMSTensor>& io,
                                   SmallVector<AMSTensor>& outs) {
      CATCH_REQUIRE(ins.size() == 2);
      CATCH_REQUIRE(ins[0].dtype() == AMSDType::AMS_SINGLE);
      CATCH_REQUIRE(ins[1].dtype() == AMSDType::AMS_INT32);
      CATCH_REQUIRE(outs[0].dtype() == AMSDType::AMS_INT32);

      float* float_ptr = ins[0].data<float>();
      int32_t* int_ptr = ins[1].data<int32_t>();
      int32_t* out_ptr = outs[0].data<int32_t>();

      for (int i = 0; i < num_elements; ++i) {
        // Convert float to int and add to int input
        out_ptr[i] = static_cast<int32_t>(float_ptr[i]) + int_ptr[i];
      }
    };

    AMSCAbstrModel model =
        AMSRegisterAbstractModel("mixed_test", 1.0, "", false);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

    AMSExecute(executor, computation, inputs, inouts, outputs);

    // Verify
    for (int i = 0; i < num_elements; ++i) {
      int32_t expected =
          static_cast<int32_t>(static_cast<float>(i) * 1.5f) + i * 2;
      CATCH_REQUIRE(output_data[i] == expected);
    }

    // Note: Not destroying executor to avoid triggering AMSFinalize between tests
    // The executor will be cleaned up at program exit
  }
}
