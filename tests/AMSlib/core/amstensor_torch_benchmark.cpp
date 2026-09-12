/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <torch/torch.h>

#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "AMSTorchInterop.hpp"
#include "ams_catch_main.hpp"

namespace
{
using ams::AMSTensor;
using Dim = AMSTensor::IntDimType;

constexpr int64_t benchmarkSizes[] = {16, 128, 1024};

std::string utcTimestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  if (!::gmtime_r(&now, &utc))
    throw std::runtime_error("Unable to construct benchmark UTC timestamp");
  std::ostringstream result;
  result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return result.str();
}

std::string csvField(const std::string& value)
{
  if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
  std::string result = "\"";
  for (const char character : value) {
    result += character;
    if (character == '\"') result += '\"';
  }
  result += '\"';
  return result;
}

bool hasContent(const std::string& path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  return input && input.tellg() > 0;
}

class CsvBenchmarkListener : public Catch::EventListenerBase
{
public:
  using Catch::EventListenerBase::EventListenerBase;

  void testRunStarting(const Catch::TestRunInfo&) override
  {
    const char* configuredPath = std::getenv("AMS_TENSOR_BENCHMARK_CSV");
    _path = configuredPath && *configuredPath
                ? configuredPath
                : AMS_TENSOR_BENCHMARK_CSV_DEFAULT;
    const bool writeHeader = !hasContent(_path);
    _output.open(_path, std::ios::out | std::ios::app);
    if (!_output)
      throw std::runtime_error("Unable to open benchmark CSV: " + _path);
    if (writeHeader) {
      _output << "timestamp_utc,ams_commit,source_state,benchmark,samples,"
                 "iterations_per_sample,mean_ns,mean_lower_ns,mean_upper_ns,"
                 "stddev_ns,stddev_lower_ns,stddev_upper_ns,outliers,"
                 "confidence_interval\n";
    }
    _timestamp = utcTimestamp();
  }

  void benchmarkEnded(const Catch::BenchmarkStats<>& stats) override
  {
    _output << csvField(_timestamp) << ',' << csvField(AMS_BENCHMARK_GIT_COMMIT)
            << ',' << csvField(AMS_BENCHMARK_SOURCE_STATE) << ','
            << csvField(stats.info.name) << ',' << stats.info.samples << ','
            << stats.info.iterations << ',' << std::setprecision(17)
            << stats.mean.point.count() << ',' << stats.mean.lower_bound.count()
            << ',' << stats.mean.upper_bound.count() << ','
            << stats.standardDeviation.point.count() << ','
            << stats.standardDeviation.lower_bound.count() << ','
            << stats.standardDeviation.upper_bound.count() << ','
            << stats.outliers.total() << ',' << stats.mean.confidence_interval
            << '\n';
    _output.flush();
    if (!_output)
      throw std::runtime_error("Unable to write benchmark CSV: " + _path);
  }

private:
  std::ofstream _output;
  std::string _path;
  std::string _timestamp;
};

CATCH_REGISTER_LISTENER(CsvBenchmarkListener)

template <typename T>
struct TypeInfo;

template <>
struct TypeInfo<float> {
  static constexpr const char* name = "float32";
  static constexpr ams::AMSDType amsDtype = ams::AMS_SINGLE;
  static constexpr c10::ScalarType torchDtype = torch::kFloat32;
};

template <>
struct TypeInfo<double> {
  static constexpr const char* name = "float64";
  static constexpr ams::AMSDType amsDtype = ams::AMS_DOUBLE;
  static constexpr c10::ScalarType torchDtype = torch::kFloat64;
};

template <>
struct TypeInfo<int32_t> {
  static constexpr const char* name = "int32";
  static constexpr ams::AMSDType amsDtype = ams::AMS_INT32;
  static constexpr c10::ScalarType torchDtype = torch::kInt32;
};

template <>
struct TypeInfo<int64_t> {
  static constexpr const char* name = "int64";
  static constexpr ams::AMSDType amsDtype = ams::AMS_INT64;
  static constexpr c10::ScalarType torchDtype = torch::kInt64;
};

std::string benchmarkName(const char* implementation,
                          const char* operation,
                          const char* dtype,
                          const char* layout,
                          int64_t rows,
                          int64_t columns)
{
  return std::string(implementation) + "/" + operation + "/" + dtype + "/" +
         layout + "/" + std::to_string(rows) + "x" + std::to_string(columns);
}

std::vector<Dim> shape(int64_t rows, int64_t columns)
{
  return {static_cast<Dim>(rows), static_cast<Dim>(columns)};
}

std::vector<Dim> strides(int64_t columns)
{
  return {static_cast<Dim>(columns), 1};
}

template <typename T>
AMSTensor makeAmsTensor(int64_t rows, int64_t columns)
{
  auto result = AMSTensor::create<T>(shape(rows, columns),
                                     strides(columns),
                                     ams::AMS_HOST);
  std::fill_n(result.template data<T>(), result.elements(), T{1});
  return result;
}

template <typename T>
torch::Tensor makeTorchTensor(int64_t rows, int64_t columns)
{
  auto options =
      torch::TensorOptions().dtype(TypeInfo<T>::torchDtype).device(torch::kCPU);
  auto result = torch::empty({rows, columns}, options);
  result.fill_(1);
  return result;
}

template <typename T>
void benchmarkAllocation(int64_t side)
{
  const auto amsName = benchmarkName(
      "AMS", "allocation", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsName))
  {
    size_t bytes = 0;
    {
      auto result =
          AMSTensor::create<T>(shape(side, side), strides(side), ams::AMS_HOST);
      bytes = result.nbytes();
    }
    return bytes;
  };

  const auto torchName = benchmarkName(
      "Torch", "allocation", TypeInfo<T>::name, "contiguous", side, side);
  const auto options =
      torch::TensorOptions().dtype(TypeInfo<T>::torchDtype).device(torch::kCPU);
  CATCH_BENCHMARK(std::string(torchName))
  {
    int64_t elements = 0;
    {
      auto result = torch::empty({side, side}, options);
      elements = result.numel();
    }
    return elements;
  };
}

template <typename T>
void benchmarkViews(int64_t side)
{
  const size_t elements = static_cast<size_t>(side * side);
  std::vector<T> raw(elements, T{1});
  const auto options =
      torch::TensorOptions().dtype(TypeInfo<T>::torchDtype).device(torch::kCPU);

  const auto amsRawName = benchmarkName(
      "AMS", "raw_view", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsRawName))
  {
    const void* pointer = nullptr;
    {
      auto result = AMSTensor::view(raw.data(),
                                    shape(side, side),
                                    strides(side),
                                    ams::AMS_HOST);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  const auto torchRawName = benchmarkName(
      "Torch", "raw_view", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(torchRawName))
  {
    const void* pointer = nullptr;
    {
      auto result = torch::from_blob(raw.data(), {side, side}, options);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  auto amsSource = makeAmsTensor<T>(side, side);
  auto torchSource = makeTorchTensor<T>(side, side);
  const auto amsAliasName = benchmarkName(
      "AMS", "retained_alias", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsAliasName))
  {
    const void* pointer = nullptr;
    {
      auto result = AMSTensor::view(amsSource);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  const auto torchAliasName = benchmarkName(
      "Torch", "retained_alias", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(torchAliasName))
  {
    const void* pointer = nullptr;
    {
      auto result = torchSource.alias();
      pointer = result.data_ptr();
    }
    return pointer;
  };
}

template <typename T>
void benchmarkMetadataAndTranspose(int64_t side)
{
  auto amsSource = makeAmsTensor<T>(side, side);
  auto torchSource = makeTorchTensor<T>(side, side);

  const auto amsMetadataName = benchmarkName(
      "AMS", "metadata_batch16", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsMetadataName))
  {
    size_t total = 0;
    for (int batch = 0; batch < 16; ++batch) {
      total += static_cast<size_t>(amsSource.elements());
      total += amsSource.nbytes();
      total += amsSource.dim();
      total += static_cast<size_t>(amsSource.shape()[0] + amsSource.shape()[1]);
      total +=
          static_cast<size_t>(amsSource.strides()[0] + amsSource.strides()[1]);
      total += static_cast<size_t>(amsSource.contiguous());
    }
    return total;
  };

  const auto torchMetadataName = benchmarkName(
      "Torch", "metadata_batch16", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(torchMetadataName))
  {
    size_t total = 0;
    for (int batch = 0; batch < 16; ++batch) {
      total += static_cast<size_t>(torchSource.numel());
      total +=
          static_cast<size_t>(torchSource.numel() * torchSource.element_size());
      total += static_cast<size_t>(torchSource.dim());
      total += static_cast<size_t>(torchSource.size(0) + torchSource.size(1));
      total +=
          static_cast<size_t>(torchSource.stride(0) + torchSource.stride(1));
      total += static_cast<size_t>(torchSource.is_contiguous());
    }
    return total;
  };

  const auto amsTransposeName = benchmarkName(
      "AMS", "transpose", TypeInfo<T>::name, "strided", side, side);
  CATCH_BENCHMARK(std::string(amsTransposeName))
  {
    const void* pointer = nullptr;
    {
      auto result = amsSource.transpose(0, 1);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  const auto torchTransposeName = benchmarkName(
      "Torch", "transpose", TypeInfo<T>::name, "strided", side, side);
  CATCH_BENCHMARK(std::string(torchTransposeName))
  {
    const void* pointer = nullptr;
    {
      auto result = torchSource.transpose(0, 1);
      pointer = result.data_ptr();
    }
    return pointer;
  };
}

template <typename T>
void benchmarkClone(int64_t side)
{
  auto amsBase = makeAmsTensor<T>(side, side);
  auto amsTransposed = amsBase.transpose(0, 1);
  auto torchBase = makeTorchTensor<T>(side, side);
  auto torchTransposed = torchBase.transpose(0, 1);

  const auto amsContiguousName = benchmarkName(
      "AMS", "clone", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsContiguousName))
  {
    size_t bytes = 0;
    {
      auto result = amsBase.clone();
      bytes = result.nbytes();
    }
    return bytes;
  };

  const auto torchContiguousName = benchmarkName(
      "Torch", "clone", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(torchContiguousName))
  {
    int64_t elements = 0;
    {
      auto result = torchBase.clone(torch::MemoryFormat::Contiguous);
      elements = result.numel();
    }
    return elements;
  };

  const auto amsStridedName = benchmarkName(
      "AMS", "clone", TypeInfo<T>::name, "transposed", side, side);
  CATCH_BENCHMARK(std::string(amsStridedName))
  {
    size_t bytes = 0;
    {
      auto result = amsTransposed.clone();
      bytes = result.nbytes();
    }
    return bytes;
  };

  const auto torchStridedName = benchmarkName(
      "Torch", "clone", TypeInfo<T>::name, "transposed", side, side);
  CATCH_BENCHMARK(std::string(torchStridedName))
  {
    int64_t elements = 0;
    {
      auto result = torchTransposed.clone(torch::MemoryFormat::Contiguous);
      elements = result.numel();
    }
    return elements;
  };
}

template <typename T>
void benchmarkConcat(int64_t side, int inputCount)
{
  const int64_t inputColumns = side / inputCount;
  ams::SmallVector<AMSTensor> amsInputs;
  std::vector<torch::Tensor> torchInputs;
  amsInputs.reserve(static_cast<size_t>(inputCount));
  torchInputs.reserve(static_cast<size_t>(inputCount));
  for (int input = 0; input < inputCount; ++input) {
    amsInputs.push_back(makeAmsTensor<T>(side, inputColumns));
    torchInputs.push_back(makeTorchTensor<T>(side, inputColumns));
  }

  const std::string operation = "concat" + std::to_string(inputCount);
  const auto amsName = benchmarkName(
      "AMS", operation.c_str(), TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(amsName))
  {
    size_t bytes = 0;
    {
      auto result = AMSTensor::concat(amsInputs, TypeInfo<T>::amsDtype);
      bytes = result.nbytes();
    }
    return bytes;
  };

  const auto torchName = benchmarkName(
      "Torch", operation.c_str(), TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(torchName))
  {
    int64_t elements = 0;
    {
      auto result = torch::cat(torchInputs, 1);
      elements = result.numel();
    }
    return elements;
  };
}

template <typename T>
void benchmarkInterop(int64_t side)
{
  auto amsSource = makeAmsTensor<T>(side, side);
  auto torchSource = makeTorchTensor<T>(side, side);

  const auto fromViewName = benchmarkName(
      "Interop", "fromTorchView", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(fromViewName))
  {
    const void* pointer = nullptr;
    {
      auto result = ams::fromTorchView(torchSource);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  const auto toViewName = benchmarkName(
      "Interop", "toTorchView", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(toViewName))
  {
    const void* pointer = nullptr;
    {
      auto result = ams::toTorchView(amsSource);
      pointer = result.data_ptr();
    }
    return pointer;
  };

  const auto fromCopyName = benchmarkName(
      "Interop", "fromTorchCopy", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(fromCopyName))
  {
    size_t bytes = 0;
    {
      auto result = ams::fromTorchCopy(torchSource);
      bytes = result.nbytes();
    }
    return bytes;
  };

  const auto toCopyName = benchmarkName(
      "Interop", "toTorchCopy", TypeInfo<T>::name, "contiguous", side, side);
  CATCH_BENCHMARK(std::string(toCopyName))
  {
    int64_t elements = 0;
    {
      auto result = ams::toTorchCopy(amsSource);
      elements = result.numel();
    }
    return elements;
  };
}

template <typename T>
void verifyOperations()
{
  constexpr int64_t side = 4;
  auto amsBase = makeAmsTensor<T>(side, side);
  auto torchBase = makeTorchTensor<T>(side, side);

  auto amsAlias = AMSTensor::view(amsBase);
  auto torchAlias = torchBase.alias();
  CATCH_REQUIRE(amsAlias.data_ptr() == amsBase.data_ptr());
  CATCH_REQUIRE(torchAlias.data_ptr() == torchBase.data_ptr());

  auto amsTransposed = amsBase.transpose(0, 1);
  auto torchTransposed = torchBase.transpose(0, 1);
  auto amsClone = amsTransposed.clone();
  auto torchClone = torchTransposed.clone(torch::MemoryFormat::Contiguous);
  CATCH_REQUIRE(amsClone.contiguous());
  CATCH_REQUIRE(torchClone.is_contiguous());
  CATCH_REQUIRE(amsClone.template data<T>()[0] == T{1});
  CATCH_REQUIRE(torchClone.template data_ptr<T>()[0] == T{1});

  ams::SmallVector<AMSTensor> amsInputs;
  std::vector<torch::Tensor> torchInputs;
  for (int input = 0; input < 2; ++input) {
    amsInputs.push_back(makeAmsTensor<T>(side, side / 2));
    torchInputs.push_back(makeTorchTensor<T>(side, side / 2));
  }
  auto amsConcat = AMSTensor::concat(amsInputs, TypeInfo<T>::amsDtype);
  auto torchConcat = torch::cat(torchInputs, 1);
  CATCH_REQUIRE(amsConcat.shape()[0] == side);
  CATCH_REQUIRE(amsConcat.shape()[1] == side);
  CATCH_REQUIRE(torchConcat.size(0) == side);
  CATCH_REQUIRE(torchConcat.size(1) == side);

  auto fromView = ams::fromTorchView(torchBase);
  auto fromCopy = ams::fromTorchCopy(torchBase);
  auto toView = ams::toTorchView(amsBase);
  auto toCopy = ams::toTorchCopy(amsBase);
  CATCH_REQUIRE(fromView.data_ptr() == torchBase.data_ptr());
  CATCH_REQUIRE(fromCopy.data_ptr() != torchBase.data_ptr());
  CATCH_REQUIRE(toView.data_ptr() == amsBase.data_ptr());
  CATCH_REQUIRE(toCopy.data_ptr() != amsBase.data_ptr());
}

template <typename T>
void benchmarkType()
{
  verifyOperations<T>();
  for (const int64_t side : benchmarkSizes)
    benchmarkAllocation<T>(side);
  benchmarkViews<T>(128);
  benchmarkMetadataAndTranspose<T>(128);
  for (const int64_t side : benchmarkSizes)
    benchmarkClone<T>(side);
  for (const int64_t side : benchmarkSizes) {
    benchmarkConcat<T>(side, 2);
    benchmarkConcat<T>(side, 4);
  }
  benchmarkInterop<T>(128);
}
}  // namespace

CATCH_TEST_CASE("AMSTensor and Torch CPU tensor-container primitives",
                "[benchmark][performance][amstensor][torch]")
{
  CATCH_REQUIRE(torch::get_num_threads() == 1);
  CATCH_REQUIRE(torch::get_num_interop_threads() == 1);
  ams::AMSInit();

  benchmarkType<float>();
  benchmarkType<double>();
  benchmarkType<int32_t>();
  benchmarkType<int64_t>();
}

int main(int argc, char** argv)
{
  ::setenv("OMP_NUM_THREADS", "1", 1);
  ::setenv("MKL_NUM_THREADS", "1", 1);
  ::setenv("OPENBLAS_NUM_THREADS", "1", 1);
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  return ams::test::runCatchSession(argc, argv);
}
