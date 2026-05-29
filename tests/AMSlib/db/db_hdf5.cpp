#define CATCH_CONFIG_PREFIX_ALL
#include <catch2/catch_all.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "AMSTypes.hpp"
#include "wf/basedb.hpp"
#include "wf/interface.hpp"

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

CATCH_TEST_CASE("DBManager tracks instances and materializes files",
                "[ams][db][instances]")
{
  namespace fs = std::filesystem;

  auto db_dir = (std::filesystem::temp_directory_path() / "ams_workflow_tests");
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) {
    perror("mkdtemp");
  }

  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);


  auto& db = ams::db::DBManager::getInstance();
  db.instantiate_fs_db(ams::AMSDBType::AMS_HDF5, db_dir.string() + "/");

  // Touch two domains twice each
  std::vector<std::string> domains = {"domain_1",
                                      "domain_2",
                                      "domain_1",
                                      "domain_2"};
  for (auto& dn : domains) {
    auto file_db = db.getDB(dn);
    (void)file_db;
  }

  // Only two unique instances should exist
  CATCH_REQUIRE(db.getNumInstances() == 2);

  // Clean triggers destructors & resets instance tracking
  db.clean();
  CATCH_REQUIRE(db.getNumInstances() == 0);

  // Files must exist on disk even after clean()
  for (auto const& dn : {"domain_1", "domain_2"}) {
    const fs::path fn = db_dir / (std::string(dn) + "_0.h5");
    CATCH_INFO("Checking file exists: " << fn.string());
    CATCH_REQUIRE(fs::exists(fn));
  }

  // Best-effort cleanup of temp artifacts
  std::error_code ec;
  db.clean();
  fs::remove_all(db_dir, ec);
}


// --- Helper to read an HDF5 dataset into a std::vector<T> ---
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

  // (Optional) sanity check the dataset type size vs T
  hid_t dtype_id = H5Dget_type(dset_id);
  CATCH_REQUIRE(dtype_id >= 0);
  const size_t dtype_size = H5Tget_size(dtype_id);
  CATCH_REQUIRE(dtype_size == sizeof(T));

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

CATCH_TEST_CASE("hdf5DB creates file and stores domain_name dataset",
                "[ams][db][hdf5]")
{
  // Choose test inputs (you can parameterize if you like)
  const std::string domain_name = "domain_1";

  auto db_dir = (std::filesystem::temp_directory_path() / "ams_workflow_tests");
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) {
    perror("mkdtemp");
  }

  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);
  // Create DB (rid=0), like your original code
  std::string filename;
  {
    ams::db::hdf5DB db(db_dir.string() + "/", domain_name, /*rid*/ 0);
    filename = db.getFilename();
  }

  CATCH_REQUIRE(std::filesystem::exists(filename));

  {
    ams::db::hdf5DB db(db_dir.string() + "/", domain_name, /*rid*/ 0);
    CATCH_REQUIRE(std::filesystem::exists(db.getFilename()));
  }

  // Read dataset "domain_name" as chars and compare to domain_name
  const std::string dataset = "domain_name";
  auto data = readHDF5Dataset<char>(filename, dataset, H5T_NATIVE_CHAR);

  // Some files might store a trailing '\0'; accept either exact or '\0'-terminated.
  std::string read_str(data.begin(), data.end());
  // Trim a single trailing NUL if present
  if (!read_str.empty() && read_str.back() == '\0') read_str.pop_back();

  CATCH_INFO("HDF5 file: " << filename);
  CATCH_INFO("Read domain_name dataset: '" << read_str << "'");
  CATCH_REQUIRE(read_str == domain_name);

  std::filesystem::remove_all(db_dir);
}

static bool verifyDatasetContents_f32_flat(
    const std::string& fileName,
    const std::string& datasetName,
    const std::vector<torch::Tensor>& expectedTensors)
{
  hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
    throw std::runtime_error("Failed to open HDF5 file: " + fileName);

  hid_t dset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
  if (dset_id < 0) {
    H5Fclose(file_id);
    throw std::runtime_error("Failed to open dataset: " + datasetName);
  }

  hid_t space_id = H5Dget_space(dset_id);
  if (space_id < 0) {
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Failed to get dataspace.");
  }

  int ndims = H5Sget_simple_extent_ndims(space_id);
  if (ndims < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Bad ndims");
  }

  std::vector<hsize_t> dims(ndims);
  if (H5Sget_simple_extent_dims(space_id, dims.data(), nullptr) < 0) {
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    throw std::runtime_error("Bad dims");
  }
  H5Sclose(space_id);

  size_t total = 1;
  for (auto d : dims)
    total *= d;

  torch::Tensor readTensor =
      torch::empty({static_cast<int64_t>(total)}, torch::kFloat32);
  herr_t st = H5Dread(dset_id,
                      H5T_NATIVE_FLOAT,
                      H5S_ALL,
                      H5S_ALL,
                      H5P_DEFAULT,
                      readTensor.data_ptr());
  H5Dclose(dset_id);
  H5Fclose(file_id);
  if (st < 0) throw std::runtime_error("H5Dread failed");

  auto expectedTensor =
      torch::cat(expectedTensors).flatten().to(torch::kFloat32).cpu();
  return torch::allclose(readTensor,
                         expectedTensor,
                         /*rtol=*/1e-5,
                         /*atol=*/1e-8);
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

// ---------- Tests ----------

CATCH_TEST_CASE("HDF5 DB: append and verify input/output datasets",
                "[ams][db][hdf5]")
{
  auto db_dir = (std::filesystem::temp_directory_path() / "ams_workflow_tests");
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) {
    perror("mkdtemp");
  }

  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);

  const std::string directory = db_dir.string() + "/";
  const std::string domain_name = "domain_foo";
  std::string filename;

  std::vector<torch::Tensor> inputTensors, outputTensors;

  // Two iterations: create then reopen+append; verify after each
  for (int iter = 0; iter < 2; ++iter) {
    {
      ams::db::hdf5DB db(directory, domain_name, /*rid=*/0);
      filename = db.getFilename();

      torch::Tensor IData =
          torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat32));
      torch::Tensor OData =
          torch::rand({21, 4}, torch::TensorOptions().dtype(torch::kFloat32));

      auto amsIData = torchToAMSTensors(IData);
      auto amsOData = torchToAMSTensors(OData);
      db.store(amsIData, amsOData);

      inputTensors.emplace_back(std::move(IData));
      outputTensors.emplace_back(std::move(OData));
    }

    CATCH_CAPTURE(filename);
    CATCH_REQUIRE(std::filesystem::exists(filename));
    CATCH_REQUIRE(
        verifyDatasetContents_f32_flat(filename, "input_data", inputTensors));
    CATCH_REQUIRE(
        verifyDatasetContents_f32_flat(filename, "output_data", outputTensors));
  }
  std::filesystem::remove_all(db_dir);
}

CATCH_TEST_CASE("HDF5 DB: 'domain_name' dataset matches provided name",
                "[ams][db][hdf5][metadata]")
{
  auto db_dir = (std::filesystem::temp_directory_path() / "ams_workflow_tests");
  std::filesystem::create_directories(db_dir);

  std::string tmp_dir = db_dir / "ams-test-XXXXXX";
  std::vector<char> tmp(tmp_dir.begin(), tmp_dir.end());
  tmp.push_back('\0');
  char* dirname = mkdtemp(tmp.data());
  if (!dirname) {
    perror("mkdtemp");
  }

  db_dir = std::filesystem::path(dirname);
  std::filesystem::create_directories(db_dir);

  const std::string directory = db_dir.string() + "/";
  const std::string domain_name = "domain_bar";
  std::string filename;

  {
    ams::db::hdf5DB db(directory, domain_name, /*rid=*/0);
    filename = db.getFilename();
  }
  CATCH_REQUIRE(std::filesystem::exists(filename));

  // Expect dataset named "domain_name" with the char contents of domain_name
  const std::string dataset = "domain_name";
  std::vector<char> expected(domain_name.begin(), domain_name.end());

  auto vec = readVectorDataset<char>(filename, dataset, H5T_NATIVE_CHAR);
  // Helpful diagnostic if it ever differs
  CATCH_REQUIRE(vec == expected);
  std::filesystem::remove_all(db_dir);
}
