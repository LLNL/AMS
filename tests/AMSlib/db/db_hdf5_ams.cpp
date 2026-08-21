/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "AMS.h"
#include "db_hdf5_helpers.hpp"
#include "wf/basedb.hpp"

using IDT = ams::AMSTensor::IntDimType;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

CATCH_TEST_CASE("DBManager tracks instances and materializes files",
                "[ams][db][instances]")
{
  auto db_dir = makeTempDir();
  auto& db = ams::db::DBManager::getInstance();
  db.instantiate_fs_db(ams::AMSDBType::AMS_HDF5, db_dir.string() + "/");

  std::vector<std::string> domains = {"domain_1",
                                      "domain_2",
                                      "domain_1",
                                      "domain_2"};
  for (auto& dn : domains) {
    auto file_db = db.getDB(dn);
    (void)file_db;
  }

  CATCH_REQUIRE(db.getNumInstances() == 2);

  db.clean();
  CATCH_REQUIRE(db.getNumInstances() == 0);

  for (auto const& dn : {"domain_1", "domain_2"}) {
    const std::filesystem::path fn = db_dir / (std::string(dn) + "_0.h5");
    CATCH_INFO("Checking file exists: " << fn.string());
    CATCH_REQUIRE(std::filesystem::exists(fn));
  }

  std::error_code ec;
  db.clean();
  std::filesystem::remove_all(db_dir, ec);
}


CATCH_TEST_CASE("hdf5DB creates file and stores domain_name dataset",
                "[ams][db][hdf5]")
{
  auto db_dir = makeTempDir();
  const std::string domain_name = "domain_1";
  std::string filename;
  {
    ams::db::hdf5DB db(db_dir.string() + "/", domain_name, 0);
    filename = db.getFilename();
  }

  CATCH_REQUIRE(std::filesystem::exists(filename));

  {
    ams::db::hdf5DB db(db_dir.string() + "/", domain_name, 0);
    CATCH_REQUIRE(std::filesystem::exists(db.getFilename()));
  }

  auto data = readHDF5Dataset<char>(filename, "domain_name", H5T_NATIVE_CHAR);
  std::string read_str(data.begin(), data.end());
  if (!read_str.empty() && read_str.back() == '\0') read_str.pop_back();

  CATCH_INFO("HDF5 file: " << filename);
  CATCH_INFO("Read domain_name dataset: '" << read_str << "'");
  CATCH_REQUIRE(read_str == domain_name);

  std::filesystem::remove_all(db_dir);
}


CATCH_TEST_CASE("HDF5 DB: 'domain_name' dataset matches provided name",
                "[ams][db][hdf5][metadata]")
{
  auto db_dir = makeTempDir();
  const std::string directory = db_dir.string() + "/";
  const std::string domain_name = "domain_bar";
  std::string filename;

  {
    ams::db::hdf5DB db(directory, domain_name, 0);
    filename = db.getFilename();
  }
  CATCH_REQUIRE(std::filesystem::exists(filename));

  std::vector<char> expected(domain_name.begin(), domain_name.end());
  auto vec = readVectorDataset<char>(filename, "domain_name", H5T_NATIVE_CHAR);
  CATCH_REQUIRE(vec == expected);
  std::filesystem::remove_all(db_dir);
}


CATCH_TEST_CASE("HDF5 DB: append and verify input/output datasets",
                "[ams][db][hdf5]")
{
  auto db_dir = makeTempDir();
  const std::string directory = db_dir.string() + "/";
  const std::string domain_name = "domain_foo";
  std::string filename;

  const IDT nRows = 21;
  const IDT nCols = 4;
  const size_t nElems = static_cast<size_t>(nRows * nCols);
  std::vector<IDT> shape = {nRows, nCols};
  std::vector<IDT> strides = {nCols, 1};

  std::vector<std::vector<float>> expectedInputs;
  std::vector<std::vector<float>> expectedOutputs;
  unsigned seed = 42;

  // Two iterations: create then reopen+append; verify after each
  for (int iter = 0; iter < 2; ++iter) {
    std::vector<float> inputBuf(nElems);
    std::vector<float> outputBuf(nElems);
    fillRandom(inputBuf.data(), nElems, seed++);
    fillRandom(outputBuf.data(), nElems, seed++);

    {
      ams::db::hdf5DB db(directory, domain_name, 0);
      filename = db.getFilename();

      auto IData = ams::AMSTensor::view<float>(inputBuf.data(),
                                               shape,
                                               strides,
                                               ams::AMSResourceType::AMS_HOST);
      auto OData = ams::AMSTensor::view<float>(outputBuf.data(),
                                               shape,
                                               strides,
                                               ams::AMSResourceType::AMS_HOST);

      db.store(IData, OData);
    }

    expectedInputs.push_back(inputBuf);
    expectedOutputs.push_back(outputBuf);

    CATCH_CAPTURE(filename);
    CATCH_REQUIRE(std::filesystem::exists(filename));
    CATCH_REQUIRE(
        verifyDatasetContents_f32(filename, "input_data", expectedInputs));
    CATCH_REQUIRE(
        verifyDatasetContents_f32(filename, "output_data", expectedOutputs));
  }
  std::filesystem::remove_all(db_dir);
}


CATCH_TEST_CASE("HDF5 DB: collects multiple AMSTensors as flat rows",
                "[ams][db][hdf5][collection]")
{
  ams::AMSInit();
  auto db_dir = makeTempDir();
  const std::string domain_name = "multi_tensor_collection";
  std::string filename;

  std::vector<float> input_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input_b = {5.0f, 6.0f};
  std::vector<float> output_a = {7.0f, 8.0f};
  std::vector<float> output_b = {9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<IDT> two_columns = {2, 2};
  std::vector<IDT> one_column = {2, 1};
  std::vector<IDT> strides_two = {2, 1};
  std::vector<IDT> strides_one = {1, 1};

  {
    ams::db::hdf5DB db(db_dir.string() + "/", domain_name, 0);
    filename = db.getFilename();

    ams::SmallVector<ams::AMSTensor> inputs;
    inputs.push_back(ams::AMSTensor::view<float>(
        input_a.data(), two_columns, strides_two, ams::AMS_HOST));
    inputs.push_back(ams::AMSTensor::view<float>(
        input_b.data(), one_column, strides_one, ams::AMS_HOST));

    ams::SmallVector<ams::AMSTensor> outputs;
    outputs.push_back(ams::AMSTensor::view<float>(
        output_a.data(), one_column, strides_one, ams::AMS_HOST));
    outputs.push_back(ams::AMSTensor::view<float>(
        output_b.data(), two_columns, strides_two, ams::AMS_HOST));

    db.store(inputs, outputs);
  }

  const std::vector<float> expected_inputs = {
      1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f};
  const std::vector<float> expected_outputs = {
      7.0f, 9.0f, 10.0f, 8.0f, 11.0f, 12.0f};
  CATCH_REQUIRE(readVectorDataset<float>(filename,
                                         "input_data",
                                         H5T_NATIVE_FLOAT) == expected_inputs);
  CATCH_REQUIRE(readVectorDataset<float>(filename,
                                         "output_data",
                                         H5T_NATIVE_FLOAT) == expected_outputs);

  std::filesystem::remove_all(db_dir);
}
