/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define CATCH_CONFIG_PREFIX_ALL
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "ams_test_device.hpp"
#include "nlohmann/json.hpp"
#include "wf/jsondb.hpp"
#include "wf/resource_manager.hpp"

#if defined(__AMS_ENABLE_TORCH__)
#include <torch/torch.h>

#include "AMSTorchInterop.hpp"
#endif

namespace fs = std::filesystem;
using Dim = ams::AMSTensor::IntDimType;

namespace
{
std::vector<uint8_t> decodeBase64(const std::string& encoded)
{
  std::vector<uint8_t> decoded;
  uint32_t accumulator = 0;
  int bits = 0;
  for (unsigned char character : encoded) {
    if (character == '=') break;
    int value = -1;
    if (character >= 'A' && character <= 'Z')
      value = character - 'A';
    else if (character >= 'a' && character <= 'z')
      value = character - 'a' + 26;
    else if (character >= '0' && character <= '9')
      value = character - '0' + 52;
    else if (character == '+')
      value = 62;
    else if (character == '/')
      value = 63;
    CATCH_REQUIRE(value >= 0);
    accumulator = (accumulator << 6) | static_cast<uint32_t>(value);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      decoded.push_back(static_cast<uint8_t>((accumulator >> bits) & 0xffU));
    }
  }
  return decoded;
}

std::vector<uint8_t> readBytes(const fs::path& path)
{
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  CATCH_REQUIRE(stream.is_open());
  const auto size = stream.tellg();
  CATCH_REQUIRE(size >= 0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.seekg(0);
  if (!bytes.empty())
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  return bytes;
}

nlohmann::json readManifest(const fs::path& path)
{
  std::ifstream stream(path);
  CATCH_REQUIRE(stream.is_open());
  nlohmann::json manifest;
  stream >> manifest;
  return manifest;
}

template <typename T>
std::vector<uint8_t> asBytes(const std::vector<T>& values)
{
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void requireTensor(const fs::path& root,
                   const nlohmann::json& descriptor,
                   const std::vector<uint8_t>& expected,
                   const std::string& dtype,
                   const nlohmann::json& shape,
                   const std::string& mode)
{
  CATCH_REQUIRE(descriptor["dtype"] == dtype);
  CATCH_REQUIRE(descriptor["shape"] == shape);
  CATCH_REQUIRE(descriptor["byte_size"] == expected.size());
  std::vector<uint8_t> actual;
  if (mode == "binary") {
    CATCH_REQUIRE(descriptor.contains("path"));
    actual = readBytes(root / descriptor["path"].get<std::string>());
  } else {
    CATCH_REQUIRE(descriptor["encoding"] == "base64");
    actual = decodeBase64(descriptor["data"].get<std::string>());
  }
  CATCH_REQUIRE(actual == expected);
}

void exerciseHostStorage(const std::string& mode)
{
  const fs::path root =
      fs::temp_directory_path() / ("ams_jsondb_amstensor_" + mode);
  fs::remove_all(root);
  fs::create_directories(root);

  std::vector<float> floats = {1.25f, -2.5f, 3.75f, 4.5f};
  std::vector<double> doubles = {-1.0, 2.0};
  std::vector<int32_t> int32s = {-3, 4, 5};
  std::vector<int64_t> int64s = {6, -7};
  double scalar = 9.5;
  std::vector<int32_t> empty;
  std::vector<float> transpose_storage = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> strided_storage = {10, 11, -1, 20, 21};

  std::vector<ams::AMSTensor> inputs;
  inputs.emplace_back(ams::AMSTensor::view<float>(floats.data(),
                                                  std::vector<Dim>{2, 2},
                                                  std::vector<Dim>{2, 1},
                                                  ams::AMS_HOST));
  inputs.emplace_back(ams::AMSTensor::view<double>(
      doubles.data(), std::vector<Dim>{2}, std::vector<Dim>{1}, ams::AMS_HOST));
  inputs.emplace_back(ams::AMSTensor::view<int32_t>(
      int32s.data(), std::vector<Dim>{3}, std::vector<Dim>{1}, ams::AMS_HOST));
  inputs.emplace_back(ams::AMSTensor::view<int64_t>(
      int64s.data(), std::vector<Dim>{2}, std::vector<Dim>{1}, ams::AMS_HOST));
  inputs.emplace_back(ams::AMSTensor::view<double>(
      &scalar, std::vector<Dim>{}, std::vector<Dim>{}, ams::AMS_HOST));
  inputs.emplace_back(ams::AMSTensor::view<int32_t>(empty.data(),
                                                    std::vector<Dim>{0, 3},
                                                    std::vector<Dim>{3, 1},
                                                    ams::AMS_HOST));
  auto matrix = ams::AMSTensor::view<float>(transpose_storage.data(),
                                            std::vector<Dim>{2, 3},
                                            std::vector<Dim>{3, 1},
                                            ams::AMS_HOST);
  inputs.emplace_back(matrix.transpose(0, 1));
  inputs.emplace_back(ams::AMSTensor::view<int64_t>(strided_storage.data(),
                                                    std::vector<Dim>{2, 2},
                                                    std::vector<Dim>{3, 1},
                                                    ams::AMS_HOST));

  fs::path manifest_path;
  {
    ams::db::JSONDB db(root.string(), "direct", 4, mode);
    manifest_path = db.getFilename();
    db.store(inputs, {});
    db.close();
  }

  const auto tensors = readManifest(manifest_path)["cases"][0]["tensors"];
  requireTensor(
      root, tensors["input_0"], asBytes(floats), "float32", {2, 2}, mode);
  requireTensor(
      root, tensors["input_1"], asBytes(doubles), "float64", {2}, mode);
  requireTensor(root, tensors["input_2"], asBytes(int32s), "int32", {3}, mode);
  requireTensor(root, tensors["input_3"], asBytes(int64s), "int64", {2}, mode);
  requireTensor(root,
                tensors["input_4"],
                asBytes(std::vector<double>{scalar}),
                "float64",
                nlohmann::json::array(),
                mode);
  requireTensor(root,
                tensors["input_5"],
                {},
                "int32",
                nlohmann::json::array({0, 3}),
                mode);
  requireTensor(root,
                tensors["input_6"],
                asBytes(std::vector<float>{1, 4, 2, 5, 3, 6}),
                "float32",
                nlohmann::json::array({3, 2}),
                mode);
  requireTensor(root,
                tensors["input_7"],
                asBytes(std::vector<int64_t>{10, 11, 20, 21}),
                "int64",
                nlohmann::json::array({2, 2}),
                mode);
  fs::remove_all(root);
}
}  // namespace

CATCH_TEST_CASE("JSONDB stores AMSTensors in binary and base64 modes",
                "[ams][db][json]")
{
  exerciseHostStorage("binary");
  exerciseHostStorage("json");
}

CATCH_TEST_CASE("JSONDB rejects heterogeneous graph storage", "[ams][db][json]")
{
  const fs::path root = fs::temp_directory_path() / "ams_jsondb_heterogeneous";
  fs::remove_all(root);
  fs::create_directories(root);
  ams::AMSHeterogeneousGraph graph;
  ams::AMSHeterogeneousGraphFields outputs;
  {
    ams::db::JSONDB db(root.string(), "heterogeneous", 0, "json");
    CATCH_REQUIRE_THROWS_WITH(db.store(graph, outputs),
                              "Heterogeneous graph storage not yet implemented "
                              "in JSONDB");
    db.close();
  }
  fs::remove_all(root);
}

#if defined(__AMS_ENABLE_TORCH__)
CATCH_TEST_CASE("JSONDB Torch adapter matches AMSTensor serialization",
                "[ams][db][json][torch]")
{
  const fs::path ams_root = fs::temp_directory_path() / "ams_jsondb_ams_schema";
  const fs::path torch_root =
      fs::temp_directory_path() / "ams_jsondb_torch_schema";
  fs::remove_all(ams_root);
  fs::remove_all(torch_root);
  fs::create_directories(ams_root);
  fs::create_directories(torch_root);

  torch::Tensor source =
      torch::tensor({1, 2, 3, 4, 5, 6}, torch::dtype(torch::kInt32))
          .reshape({2, 3})
          .transpose(0, 1);
  std::vector<torch::Tensor> torch_inputs{source};
  std::vector<ams::AMSTensor> ams_inputs;
  ams_inputs.emplace_back(ams::fromTorchView(source));
  fs::path ams_manifest;
  fs::path torch_manifest;
  {
    ams::db::JSONDB db(ams_root.string(), "ams", 0, "json");
    ams_manifest = db.getFilename();
    db.store(ams_inputs, {});
    db.close();
  }
  {
    ams::db::JSONDB db(torch_root.string(), "torch", 0, "json");
    torch_manifest = db.getFilename();
    db.store(torch_inputs, {});
    db.close();
  }

  CATCH_REQUIRE(readManifest(ams_manifest)["cases"] ==
                readManifest(torch_manifest)["cases"]);
  fs::remove_all(ams_root);
  fs::remove_all(torch_root);
}

CATCH_TEST_CASE("JSONDB Torch adapter rejects unsupported dtypes",
                "[ams][db][json][torch]")
{
  const fs::path root = fs::temp_directory_path() / "ams_jsondb_torch_dtype";
  fs::remove_all(root);
  fs::create_directories(root);
  std::vector<torch::Tensor> inputs{torch::ones({2}, torch::kBool)};
  std::vector<torch::Tensor> outputs;
  {
    ams::db::JSONDB db(root.string(), "dtype", 0, "json");
    CATCH_REQUIRE_THROWS_AS(db.store(inputs, outputs), std::invalid_argument);
    db.close();
  }
  fs::remove_all(root);
}
#endif

#if defined(__AMS_ENABLE_HIP__) || defined(__AMS_ENABLE_CUDA__)
CATCH_TEST_CASE("JSONDB stages pinned and device AMSTensors on the host",
                "[ams][db][json][device]")
{
  if (!ams::test::hasRuntimeDevice()) CATCH_SKIP("GPU device not available");
  ams::AMSInit();
  const fs::path root = fs::temp_directory_path() / "ams_jsondb_device_staging";
  fs::remove_all(root);
  fs::create_directories(root);

  const std::vector<float> expected = {1, 2, 3, 4};
  const std::vector<Dim> shape = {2, 2};
  const std::vector<Dim> strides = {2, 1};
  auto device = ams::AMSTensor::create<float>(shape, strides, ams::AMS_DEVICE);
  auto pinned = ams::AMSTensor::create<float>(shape, strides, ams::AMS_PINNED);
  ams::internal::_raw_copy(const_cast<float*>(expected.data()),
                           ams::AMS_HOST,
                           device.data_ptr(),
                           ams::AMS_DEVICE,
                           device.nbytes());
  ams::internal::_raw_copy(const_cast<float*>(expected.data()),
                           ams::AMS_HOST,
                           pinned.data_ptr(),
                           ams::AMS_PINNED,
                           pinned.nbytes());
  std::vector<ams::AMSTensor> inputs;
  inputs.emplace_back(std::move(device));
  inputs.emplace_back(std::move(pinned));

  fs::path manifest_path;
  {
    ams::db::JSONDB db(root.string(), "device", 0, "json");
    manifest_path = db.getFilename();
    db.store(inputs, {});
    db.close();
  }
  const auto tensors = readManifest(manifest_path)["cases"][0]["tensors"];
  requireTensor(
      root, tensors["input_0"], asBytes(expected), "float32", {2, 2}, "json");
  requireTensor(
      root, tensors["input_1"], asBytes(expected), "float32", {2, 2}, "json");
  fs::remove_all(root);
}
#endif
