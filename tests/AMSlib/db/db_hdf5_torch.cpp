/*
 * Copyright 2021-2026 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <torch/torch.h>

#include "AMS.h"
#include "db_hdf5_helpers.hpp"
#include "wf/basedb.hpp"

using IDT = ams::AMSTensor::IntDimType;


/// Create an AMSTensor view over a CPU-contiguous float32 torch::Tensor.
static ams::AMSTensor torchToAMSView(torch::Tensor& t)
{
  std::vector<IDT> shape(t.sizes().begin(), t.sizes().end());
  std::vector<IDT> strides(t.strides().begin(), t.strides().end());
  return ams::AMSTensor::view<float>(t.data_ptr<float>(),
                                     shape,
                                     strides,
                                     ams::AMSResourceType::AMS_HOST);
}


CATCH_TEST_CASE("DBManager tracks instances and materializes files (torch)",
                "[ams][db][instances][torch]")
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


CATCH_TEST_CASE("hdf5DB creates file and stores domain_name dataset (torch)",
                "[ams][db][hdf5][torch]")
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


CATCH_TEST_CASE("HDF5 DB: 'domain_name' dataset matches provided name (torch)",
                "[ams][db][hdf5][metadata][torch]")
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


CATCH_TEST_CASE("HDF5 DB (Torch): append and verify input/output datasets",
                "[ams][db][hdf5][torch]")
{
  auto db_dir = makeTempDir();
  const std::string directory = db_dir.string() + "/";
  const std::string domain_name = "domain_torch";
  std::string filename;

  std::vector<std::vector<float>> expectedInputs;
  std::vector<std::vector<float>> expectedOutputs;

  for (int iter = 0; iter < 2; ++iter) {
    torch::Tensor IData =
        torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor OData =
        torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    {
      ams::db::hdf5DB db(directory, domain_name, 0);
      filename = db.getFilename();

      auto amsI = torchToAMSView(IData);
      auto amsO = torchToAMSView(OData);
      db.store(amsI, amsO);
    }

    // Capture expected data as flat float vectors
    {
      auto* iPtr = IData.data_ptr<float>();
      expectedInputs.emplace_back(iPtr, iPtr + IData.numel());
      auto* oPtr = OData.data_ptr<float>();
      expectedOutputs.emplace_back(oPtr, oPtr + OData.numel());
    }

    CATCH_CAPTURE(filename);
    CATCH_REQUIRE(std::filesystem::exists(filename));
    CATCH_REQUIRE(
        verifyDatasetContents_f32(filename, "input_data", expectedInputs));
    CATCH_REQUIRE(
        verifyDatasetContents_f32(filename, "output_data", expectedOutputs));
  }
  std::filesystem::remove_all(db_dir);
}