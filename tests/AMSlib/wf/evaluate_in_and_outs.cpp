#include <hdf5.h>
#include <torch/torch.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/interfaces/catch_interfaces_reporter.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "AMS.h"
#include "models/simple_models.hpp"
#include "wf/workflow.hpp"

using namespace ams;

// Register it so it runs before any test cases

static std::atomic<int> g_test_counter{0};

static std::string next_test_name()
{
  int id = g_test_counter.fetch_add(1, std::memory_order_relaxed);
  return "test_" + std::to_string(id);
}

// ---------- Pretty printers so Catch can show readable names ----------
struct PhysicsCfg {
  std::string Prec;  // "float" | "double"
  std::string Dev;   // "cpu" | "cuda" (rocm maps to cuda device type)
  int NumInOuts;     // e.g., 0,1,...
};
inline std::ostream& operator<<(std::ostream& os, PhysicsCfg const& p)
{
  return os << "ph.prec=" << p.Prec << " | ph.dev=" << p.Dev
            << " | inouts=" << p.NumInOuts;
}


static bool verifyDatasetContents(const std::string& fileName,
                                  const std::string& datasetName,
                                  const torch::Tensor& expectedFlatCpuFloat)
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
    throw std::runtime_error("Failed to get dataspace");
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

  // Read as float32 on CPU
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

  auto expected = expectedFlatCpuFloat.reshape({static_cast<int64_t>(total)})
                      .to(torch::kFloat32)
                      .cpu();
  if (!torch::allclose(readTensor, expected, /*rtol=*/1e-5, /*atol=*/1e-8)) {
    CATCH_INFO("HDF5 mismatch in dataset '" << datasetName
                                            << "' total=" << total);
    CATCH_INFO("read[0:10]="
               << readTensor.index({torch::indexing::Slice(
                      0, std::min<int64_t>(10, readTensor.size(0)))})
               << " expected[0:10]="
               << expected.index({torch::indexing::Slice(
                      0, std::min<int64_t>(10, expected.size(0)))}));
    return false;
  }
  return true;
}

// compile-time mapping like your original (CUDA/HIP -> CUDA dev type)
#if defined(__AMS_ENABLE_CUDA__) || defined(__AMS_ENABLE_HIP__)
constexpr c10::DeviceType AMS_TEST_CTYPE = c10::DeviceType::CUDA;
#else
constexpr c10::DeviceType AMS_TEST_CTYPE =
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
#endif

// ---------- Core compute (your callback logic, templated) ----------
template <typename T, torch::Dtype DType, torch::DeviceType DeviceType>
static void compute(ams::AMSWorkflow& wf,
                    std::vector<torch::Tensor>& orig_in,
                    std::vector<torch::Tensor>& orig_inout,
                    std::vector<torch::Tensor>& orig_out,
                    T& /*broadcastVal*/,
                    bool /*has_broadcast*/ = false)
{
  auto callBack = [&](const ams::SmallVector<ams::AMSTensor>& pruned_ins,
                      ams::SmallVector<ams::AMSTensor>& pruned_inouts,
                      ams::SmallVector<ams::AMSTensor>& pruned_outs) {
    const int numIn = pruned_ins.size();
    const int numInOut = pruned_inouts.size();
    const int numOut = pruned_outs.size();
    int numElements = 0;
    if (!pruned_ins.empty())
      numElements = pruned_ins[0].shape()[0];
    else if (!pruned_inouts.empty())
      numElements = pruned_inouts[0].shape()[0];
    else
      CATCH_FAIL("Callback received empty batch");

    std::vector<torch::Tensor> in;
    in.reserve(numIn);
    std::vector<torch::Tensor> inout;
    inout.reserve(numInOut);
    std::vector<torch::Tensor> out;
    out.reserve(numOut);

    for (auto& V : pruned_ins) {
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      in.push_back(torch::from_blob(const_cast<void*>(V.data_ptr()),
                                    shape,
                                    torch::TensorOptions().dtype(DType).device(
                                        DeviceType)));
    }
    for (auto& V : pruned_inouts) {
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      inout.push_back(torch::from_blob(
          V.data_ptr(),
          shape,
          torch::TensorOptions().dtype(DType).device(DeviceType)));
    }
    for (auto& V : pruned_outs) {
      c10::IntArrayRef shape(V.shape().begin(), V.shape().size());
      out.push_back(torch::from_blob(V.data_ptr(),
                                     shape,
                                     torch::TensorOptions().dtype(DType).device(
                                         DeviceType)));
    }

    torch::Tensor identity_matrix =
        torch::eye(out.size() + inout.size(),
                   torch::TensorOptions().dtype(DType).device(DeviceType));

    for (int i = 0; i < numElements; i++) {
      torch::Tensor aggregate =
          torch::zeros({1, numIn + numInOut},
                       torch::TensorOptions().dtype(DType).device(DeviceType));

      for (int j = 0; j < numIn; j++)
        aggregate[0][j] = in[j][i][0];
      for (int j = 0; j < numInOut; j++)
        aggregate[0][numIn + j] = inout[j][i][0];

      auto res = aggregate.matmul(identity_matrix) * 13.0;

      for (int j = 0; j < numOut; j++)
        out[j][i][0] = res[0][j];
      for (int j = 0; j < numInOut; j++)
        inout[j][i][0] = res[0][numOut + j];
    }
  };

  wf.evaluate(callBack, orig_in, orig_inout, orig_out);
}

// ---------- The test ----------
CATCH_TEST_CASE("Workflow Evaluate: In/Out/InOut + HDF5 verification",
                "[ams][workflow][db]")
{
  // ----- Generators -----
  auto model_desc = GENERATE_COPY(from_range(simple_models));
  auto threshold = GENERATE(Catch::Generators::values({0.0, 0.5, 1.0}));

  // Physics (host) precision/device cartesian product
  auto phys = GENERATE_REF(Catch::Generators::values<PhysicsCfg>({
      {"float", "cpu", 0},
      {"double", "cpu", 0},
      {"float", "cpu", 1},
      {"double", "cpu", 1},
      {"float", "cpu", 8},
      {"double", "cpu", 8},
      // add more NumInOuts here if you want to cover >0: {"float","cuda",1}, ...
  }));

  // Skip GPU physics if CUDA not available
  if (phys.Dev == "cuda" && !torch::cuda::is_available()) {
    CATCH_SKIP("CUDA not available; skipping " << phys << " with "
                                               << model_desc);
  }

  if (threshold == 0.5 && model_desc.UQType != "duq_max")
    CATCH_SKIP("These tests only support duq_max " << phys << " with "
                                                   << model_desc);


  // Fixed shapes you pass in your example ("1,8") -> we only need the last dim here
  constexpr int SIZE = 32;
  const std::vector<int64_t> iShape = {1, 8};
  const std::vector<int64_t> oShape = {1, 8};

  // Prepare DB
  auto& db_instance = ams::db::DBManager::getInstance();

  // Types & devices
  torch::Dtype DType =
      (phys.Prec == "double") ? torch::kFloat64 : torch::kFloat32;
  c10::DeviceType dev =
      (phys.Dev == "cuda") ? c10::DeviceType::CUDA : c10::DeviceType::CPU;

  std::string domain{next_test_name()};
  std::string filename;
  {
    ams::AMSWorkflow wf(model_desc.ModelPath,
                        /*domain*/ domain,
                        /*threshold*/ threshold,
                        0,
                        1);
    filename = std::string(wf.getDBFilename());

    torch::TensorOptions tOptions =
        torch::TensorOptions().dtype(DType).device(dev);

    // Build inputs based on last dims and NumInOuts
    const int numInOuts = phys.NumInOuts;
    const int numIn = static_cast<int>(iShape.back()) - numInOuts;
    const int numOut = static_cast<int>(oShape.back()) - numInOuts;

    std::vector<torch::Tensor> in, inout, out;
    in.reserve(numIn);
    inout.reserve(numInOuts);
    out.reserve(numOut);

    CATCH_CAPTURE(phys, model_desc, threshold, filename);
    for (int i = 0; i < numIn; ++i)
      in.push_back(torch::ones({SIZE, 1}, tOptions));
    for (int i = 0; i < numInOuts; ++i)
      inout.push_back(torch::ones({SIZE, 1}, tOptions));
    for (int i = 0; i < numOut; ++i)
      out.push_back(torch::zeros({SIZE, 1}, tOptions));

    float fb = 0.0f;
    double db = 0.0;

    // Dispatch compute like your original (respecting compile-time AMS_TEST_CTYPE)
    if (DType == torch::kFloat64 && dev == AMS_TEST_CTYPE)
      compute<double, torch::kFloat64, AMS_TEST_CTYPE>(
          wf, in, inout, out, db, false);
    else if (DType == torch::kFloat32 && dev == AMS_TEST_CTYPE)
      compute<float, torch::kFloat32, AMS_TEST_CTYPE>(
          wf, in, inout, out, fb, false);
    else if (DType == torch::kFloat64 && dev == c10::DeviceType::CPU)
      compute<double, torch::kFloat64, c10::DeviceType::CPU>(
          wf, in, inout, out, db, false);
    else if (DType == torch::kFloat32 && dev == c10::DeviceType::CPU)
      compute<float, torch::kFloat32, c10::DeviceType::CPU>(
          wf, in, inout, out, fb, false);
    else
      CATCH_FAIL("Unsupported (dtype,dev) combo");

    // If there is no model, AMS should ignore threshold -> treat like 0.0
    double effThreshold = model_desc.ModelPath.empty() ? 0.0 : threshold;

    // Check out & inout values like your binary
    for (auto& V : {std::ref(inout), std::ref(out)}) {
      for (size_t i = 0; i < V.get().size(); ++i) {
        auto data = V.get()[i];
        if (effThreshold == 0.0) {
          auto correct = torch::ones(data.sizes(), data.options()) * 13;
          CATCH_REQUIRE(torch::allclose(correct, data, 1e-5, 1e-8));
        } else if (effThreshold == 0.5) {
          auto correct = torch::ones(data.sizes(), data.options());
          auto indices = torch::arange(data.sizes()[0],
                                       correct.options().dtype(torch::kLong));
          auto alt = (indices % 2).to(correct.dtype()) * 12;
          alt = alt.reshape({data.sizes()[0], 1});
          correct += alt;
          CATCH_REQUIRE(torch::allclose(correct, data, 1e-5, 1e-8));
        } else if (effThreshold == 1.0) {
          auto correct = torch::ones(data.sizes(), data.options());
          CATCH_REQUIRE(torch::allclose(correct, data, 1e-5, 1e-8));
        } else {
          CATCH_FAIL("Unknown threshold value");
        }
      }
    }
  }

  db_instance.clean();
  double collectFrac = 1.0 - threshold;
  if (collectFrac > 0.0) {
    const int nin = static_cast<int>(iShape.back());
    const int nout = static_cast<int>(oShape.back());

    // NOTE: HDF5 stored as float32 CPU in verifier
    auto expectedInput =
        torch::ones({static_cast<int64_t>(SIZE * collectFrac), nin},
                    torch::TensorOptions().dtype(torch::kFloat32));
    auto expectedOutput =
        torch::ones({static_cast<int64_t>(SIZE * collectFrac), nout},
                    torch::TensorOptions().dtype(torch::kFloat32)) *
        13.0f;

    if (threshold != 1.0) {
      CATCH_REQUIRE(
          verifyDatasetContents(filename, "input_data", expectedInput));
      CATCH_REQUIRE(
          verifyDatasetContents(filename, "output_data", expectedOutput));
    }
  }
}

int main(int argc, char** argv)
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

  AMSInit();
  // Use a temp file in the build tree (std::filesystem temp)
  auto& db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(AMSDBType::AMS_HDF5, db_dir.string());


  Catch::Session session;

  if (int rc = session.applyCommandLine(argc, argv))
    return rc;  // bad CLI -> propagate

  int rc = session.run();  // run tests
  std::cout << "RC:" << rc << "\n";

  // Treat "warnings only" (e.g., due to skipped tests) as success for CTest.
  // Catch2 commonly uses 4 for warnings; adjust if your config differs.
  AMSFinalize();
  std::filesystem::remove_all(db_dir);
  std::fflush(stdout);
  std::fflush(stderr);
  // Skip late HIP/Torch global finalizers after explicit AMS cleanup.
  std::_Exit((rc == 0 || rc == 4) ? EXIT_SUCCESS : rc);
}
