/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#define CATCH_CONFIG_PREFIX_ALL
#include <H5Ipublic.h>
#include <hdf5.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Temp directory
// ---------------------------------------------------------------------------

static inline std::filesystem::path makeTempDir()
{
  auto db_dir = std::filesystem::temp_directory_path() / "ams_workflow_tests";
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) perror("mkdtemp");
  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);
  return db_dir;
}

// ---------------------------------------------------------------------------
// HDF5 dataset readers
// ---------------------------------------------------------------------------

template <typename T>
static std::vector<T> readHDF5Dataset(const std::string& filePath,
                                      const std::string& datasetName,
                                      hid_t expectedNativeType)
{
  hid_t file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CATCH_REQUIRE(file_id >= 0);

  hid_t dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    H5Fclose(file_id);
    CATCH_FAIL("Failed to open dataset: " << datasetName);
  }

  hid_t space_id = H5Dget_space(dset_id);
  CATCH_REQUIRE(space_id >= 0);

  int ndims = H5Sget_simple_extent_ndims(space_id);
  CATCH_REQUIRE(ndims >= 0);

  std::vector<hsize_t> dims(ndims, 0);
  CATCH_REQUIRE(H5Sget_simple_extent_dims(space_id, dims.data(), nullptr) >= 0);

  size_t total_elems = 1;
  for (auto d : dims)
    total_elems *= static_cast<size_t>(d);

  hid_t dtype_id = H5Dget_type(dset_id);
  CATCH_REQUIRE(dtype_id >= 0);
  CATCH_REQUIRE(H5Tget_size(dtype_id) == sizeof(T));

  std::vector<T> out(total_elems);
  CATCH_REQUIRE(H5Dread(dset_id,
                        expectedNativeType,
                        H5S_ALL,
                        H5S_ALL,
                        H5P_DEFAULT,
                        out.data()) >= 0);

  H5Tclose(dtype_id);
  H5Sclose(space_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  return out;
}

template <typename T>
static std::vector<T> readVectorDataset(const std::string& filePath,
                                        const std::string& datasetName,
                                        hid_t DataType)
{
  hid_t file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    throw std::runtime_error("Failed to open HDF5 file: " + filePath);

  hid_t dataset_id = H5Dopen(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Fclose(file_id);
    throw std::runtime_error("Failed to open dataset: " + datasetName);
  }

  hid_t dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0) {
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataspace for dataset: " +
                             datasetName);
  }

  int ndims = H5Sget_simple_extent_ndims(dataspace_id);
  std::vector<hsize_t> dims(ndims);
  H5Sget_simple_extent_dims(dataspace_id, dims.data(), nullptr);

  hid_t datatype_id = H5Dget_type(dataset_id);
  size_t datatype_size = H5Tget_size(datatype_id);

  std::vector<T> data;
  if (datatype_size == sizeof(T)) {
    size_t total = 1;
    for (auto d : dims)
      total *= d;
    data.resize(total);
    if (H5Dread(
            dataset_id, DataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) <
        0) {
      H5Tclose(datatype_id);
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Fclose(file_id);
      throw std::runtime_error("Failed to read dataset: " + datasetName);
    }
  } else {
    H5Tclose(datatype_id);
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Unsupported data type for dataset: " +
                             datasetName);
  }

  H5Tclose(datatype_id);
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  return data;
}

// ---------------------------------------------------------------------------
// Random data generation and comparison
// ---------------------------------------------------------------------------

static inline void fillRandom(float* data, size_t count, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < count; ++i)
    data[i] = dist(gen);
}

static inline bool allClose(const std::vector<float>& a,
                            const std::vector<float>& b,
                            float rtol = 1e-5f,
                            float atol = 1e-8f)
{
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > atol + rtol * std::fabs(b[i])) return false;
  }
  return true;
}

/// Verify that an HDF5 dataset contains the expected float data.
/// Each buffer in expectedBuffers is one store() call's flat float data,
/// concatenated along the first dimension to form the expected HDF5 content.
static inline bool verifyDatasetContents_f32(
    const std::string& fileName,
    const std::string& datasetName,
    const std::vector<std::vector<float>>& expectedBuffers)
{
  std::vector<float> expected;
  for (auto& buf : expectedBuffers)
    expected.insert(expected.end(), buf.begin(), buf.end());

  auto actual = readHDF5Dataset<float>(fileName, datasetName, H5T_NATIVE_FLOAT);
  return allClose(actual, expected);
}